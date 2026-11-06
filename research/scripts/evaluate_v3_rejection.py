# -*- coding: utf-8 -*-
"""第三阶段：拒识（未知/易混指令）评估与阈值选择。

协议
----
* 阈值只在 **验证集已知命令** + **未知校准集（unknown_v3 calibration 12 文本/36 条）**
  上选择；测试集（已知 483 条 + 未知测试 36 条）只做一次最终评估。
* 与已知命令同文本的未知样本不存在（数据审计已保证跨子集文本无重叠）。
* 分数 = 温度标定后的最大类概率（max_prob，预注册主方案）；
  margin（top1-top2）作为补充指标同时报告，但不参与选择。

产物
----
* ``data/models/v3_thresholds.json``：选中阈值与元数据（供控制服务/演示加载）；
* ``outputs/v3_experiments/rejection.json``：验证集曲线、测试集指标、
  硬负例逐文本误接受表、自助法置信区间。

注意：未知集只有 36 条测试样本，FAR 的 95% 置信上界须按 rule-of-three
（3/36≈8.3%）解读，不能把观测到的 0% 当作真实安全保证。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.v3_model import V3Predictor, THRESHOLDS_FILENAME  # noqa: E402

INTENTS = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_audio(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    cand = BASE / p
    if cand.exists():
        return cand
    return BASE / 'data' / 'tts_corpus' / p


def load_known(cfg: dict):
    data = cfg['data']
    manifest = BASE / data['manifest']
    split = BASE / data['split_file']
    with open(split, encoding='utf-8-sig', newline='') as f:
        group2subset = {r['expression_group_id']: r['subset'] for r in csv.DictReader(f)}
    recs = {'train': [], 'val': [], 'test': []}
    with open(manifest, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            gid = row.get('expression_group_id', '')
            if gid not in group2subset:
                continue
            if (row.get('quality_status') or 'ok') != 'ok':
                continue
            if row['command'] not in INTENTS:
                continue
            recs[group2subset[gid]].append({
                'utterance_id': row['utterance_id'],
                'audio_path': str(resolve_audio(row['audio_path'])),
                'command': row['command'],
                'expression_group_id': gid,
                'register': row.get('register', ''),
                'condition': row.get('condition', ''),
                'voice': row.get('voice_identity', row.get('speaker_id', '')),
                'text': row.get('text', ''),
            })
    return recs


def load_unknown():
    manifest = BASE / 'data' / 'tts_corpus' / 'unknown_v3.csv'
    if not manifest.exists():
        raise FileNotFoundError(f'未知指令清单不存在: {manifest}')
    recs = {'calibration': [], 'test': []}
    with open(manifest, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            recs[row['text_split']].append({
                'utterance_id': row['utterance_id'],
                'audio_path': str(resolve_audio(row['audio_path'])),
                'label': row.get('label', 'unknown'),
                'negative_type': row.get('negative_type', ''),
                'text': row.get('text', ''),
                'text_group_id': row.get('text_group_id', ''),
                'voice': row.get('voice_id', ''),
            })
    return recs


def score_all(model: V3Predictor, recs, log=print):
    out = []
    for i, r in enumerate(recs):
        pred = model.predict(r['audio_path'])
        out.append({**{k: r[k] for k in r if k != 'audio_path'},
                    'audio_path': r['audio_path'],
                    'pred_command': pred.command,
                    'max_prob': float(pred.confidence),
                    'margin': float(pred.margin),
                    'entropy': float(pred.entropy)})
        if (i + 1) % 200 == 0:
            log(f'  打分 {i + 1}/{len(recs)}')
    return out


def threshold_curve(known, unknown, thresholds):
    rows = []
    for t in thresholds:
        ka = [s for s in known if s['max_prob'] >= t]
        ua = [s for s in unknown if s['max_prob'] >= t]
        rows.append({
            'threshold': float(t),
            'known_coverage': len(ka) / len(known) if known else 0.0,
            'known_false_reject_rate': 1.0 - (len(ka) / len(known) if known else 0.0),
            'accepted_accuracy': float(np.mean([s['pred_command'] == s['command'] for s in ka]))
            if ka else 0.0,
            'n_known_accepted': len(ka),
            'unknown_false_accept_rate': len(ua) / len(unknown) if unknown else None,
            'n_unknown_accepted': len(ua),
        })
    return rows


def select_threshold(curve, min_coverage, target_far):
    """在满足覆盖率约束下最小化未知误接受率；无法满足 FAR 时取 FAR 最小。"""
    feasible = [r for r in curve if r['known_coverage'] >= min_coverage]
    within_far = [r for r in feasible
                  if r['unknown_false_accept_rate'] is not None
                  and r['unknown_false_accept_rate'] <= target_far]
    if within_far:
        # 约束内：优先覆盖率（更可用），其次接受准确率
        return max(within_far, key=lambda r: (r['known_coverage'], r['accepted_accuracy']))
    if feasible:
        return min(feasible, key=lambda r: (r['unknown_false_accept_rate'], -r['known_coverage']))
    return max(curve, key=lambda r: r['known_coverage'])


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def group_bootstrap_coverage(known, accepted_flags, n_boot=2000, seed=42):
    by_group = defaultdict(list)
    for r, a in zip(known, accepted_flags):
        by_group[r['expression_group_id']].append(a)
    groups = sorted(by_group)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        s = [x for i in pick for x in by_group[groups[i]]]
        vals.append(float(np.mean(s)))
    return {'coverage_ci95': [float(np.percentile(vals, 2.5)),
                              float(np.percentile(vals, 97.5))],
            'n_groups': len(groups)}


def accepted_accuracy_bootstrap(accepted, n_boot=2000, seed=42):
    if not accepted:
        return None
    correct = np.array([s['pred_command'] == s['command'] for s in accepted], dtype=float)
    rng = np.random.RandomState(seed)
    vals = [float(np.mean(rng.choice(correct, size=len(correct), replace=True)))
            for _ in range(n_boot)]
    return {'accepted_accuracy_ci95': [float(np.percentile(vals, 2.5)),
                                       float(np.percentile(vals, 97.5))]}


def main():
    ap = argparse.ArgumentParser(description='v3 拒识评估（阈值仅验证集选择）')
    ap.add_argument('--model', default=str(BASE / 'data/models/v3_cnn_ensemble.npz'))
    ap.add_argument('--config', default=str(BASE / 'config_v3.json'))
    ap.add_argument('--out', default=str(BASE / 'outputs/v3_experiments/rejection.json'))
    ap.add_argument('--thresholds-out',
                    default=str(BASE / 'data/models' / THRESHOLDS_FILENAME))
    ap.add_argument('--min-coverage', type=float, default=None)
    ap.add_argument('--target-far', type=float, default=None)
    ap.add_argument('--no-write-thresholds', action='store_true')
    args = ap.parse_args()

    cfg = load_json(args.config)
    rej_cfg = cfg['rejection']
    min_coverage = float(args.min_coverage if args.min_coverage is not None
                         else rej_cfg['min_known_coverage'])
    target_far = float(args.target_far if args.target_far is not None
                       else rej_cfg['target_far'])

    model = V3Predictor.load(args.model)
    known = load_known(cfg)
    unknown = load_unknown()
    print(f"模型: {model.model_name}  温度={model.temperature:.3f}")
    print(f"已知集: val={len(known['val'])} test={len(known['test'])}；"
          f"未知集: calibration={len(unknown['calibration'])} test={len(unknown['test'])}")

    val_sc = score_all(model, known['val'])
    unk_cal_sc = score_all(model, unknown['calibration'])
    test_sc = score_all(model, known['test'])
    unk_test_sc = score_all(model, unknown['test'])

    thresholds = np.round(np.arange(0.05, 1.0, 0.01), 2)
    curve = threshold_curve(val_sc, unk_cal_sc, thresholds)
    # 验证集上的 no-threshold 基线
    val_acc = float(np.mean([s['pred_command'] == s['command'] for s in val_sc]))
    chosen = select_threshold(curve, min_coverage, target_far)
    thr = float(chosen['threshold'])

    def apply(scores):
        accepted = [s for s in scores if s['max_prob'] >= thr]
        rejected = [s for s in scores if s['max_prob'] < thr]
        return accepted, rejected

    acc_te, rej_te = apply(test_sc)
    acc_un, rej_un = apply(unk_test_sc)
    te_correct = float(np.mean([s['pred_command'] == s['command'] for s in accept_all(test_sc)])) \
        if test_sc else 0.0
    accepted_acc = float(np.mean([s['pred_command'] == s['command'] for s in acc_te])) if acc_te else 0.0

    by_negative = defaultdict(lambda: {'n': 0, 'accepted': 0, 'texts': set()})
    for s in unk_test_sc:
        key = s['label']
        by_negative[key]['n'] += 1
        by_negative[key]['accepted'] += int(s['max_prob'] >= thr)
        by_negative[key]['texts'].add(s['text'])
    by_type = defaultdict(lambda: {'n': 0, 'accepted': 0})
    for s in unk_test_sc:
        by_type[s['negative_type']]['n'] += 1
        by_type[s['negative_type']]['accepted'] += int(s['max_prob'] >= thr)

    # 每个未知文本的最大置信度（用于找最危险硬负例）
    text_stats = defaultdict(lambda: {'n': 0, 'n_accepted': 0, 'max_prob': 0.0, 'pred': Counter()})
    for s in unk_test_sc:
        rec = text_stats[s['text']]
        rec['n'] += 1
        rec['max_prob'] = max(rec['max_prob'], s['max_prob'])
        rec['n_accepted'] += int(s['max_prob'] >= thr)
        rec['pred'][s['pred_command']] += 1

    out = {
        'protocol': {
            'model': str(args.model),
            'model_sha256': sha256_file(Path(args.model)),
            'threshold_selected_on': 'val(known) + unknown_v3 calibration(36 clips/12 texts)',
            'test_used_for': 'final evaluation only（本脚本不据 test 改阈值）',
            'score': 'temperature-scaled max softmax probability',
            'selection_rule': {
                'min_known_coverage': min_coverage,
                'target_far': target_far,
                'rule': '满足 coverage>=min 且 FAR<=target 时最大化 coverage；否则最小化 FAR',
            },
            'config': str(args.config),
        },
        'sizes': {
            'known_val': len(val_sc), 'known_test': len(test_sc),
            'unknown_calibration': len(unk_cal_sc), 'unknown_test': len(unk_test_sc),
            'unknown_test_texts': len(text_stats),
        },
        'selected_threshold': thr,
        'selected_on_validation': {
            'known_coverage': chosen['known_coverage'],
            'known_false_reject_rate': chosen['known_false_reject_rate'],
            'accepted_accuracy': chosen['accepted_accuracy'],
            'unknown_false_accept_rate': chosen['unknown_false_accept_rate'],
            'n_known_accepted': chosen['n_known_accepted'],
        },
        'validation_curve': curve[::5],
        'test_results': {
            'no_threshold_accuracy': te_correct,
            'known_coverage': len(acc_te) / len(test_sc),
            'known_false_reject_rate': 1.0 - len(acc_te) / len(test_sc),
            'accepted_accuracy': accepted_acc,
            'n_known_accepted': len(acc_te),
            'unknown_false_accept_rate': len(acc_un) / len(unk_test_sc),
            'unknown_rejection_rate': 1.0 - len(acc_un) / len(unk_test_sc),
            'n_unknown_test': len(unk_test_sc),
            'unknown_far_wilson95': wilson_ci(len(acc_un), len(unk_test_sc)),
            'far_rule_of_three_upper95': (3.0 / len(unk_test_sc)) if unk_test_sc else None,
            'known_coverage_bootstrap': group_bootstrap_coverage(
                test_sc, [s['max_prob'] >= thr for s in test_sc]),
            'accepted_accuracy_bootstrap': accepted_accuracy_bootstrap(acc_te),
            'by_label': {k: {'n': v['n'], 'accepted': v['accepted'],
                             'false_accept_rate': v['accepted'] / v['n']}
                         for k, v in by_negative.items()},
            'hard_negative_far': (by_negative['hard_negative']['accepted']
                                  / by_negative['hard_negative']['n']
                                  if by_negative['hard_negative']['n'] else None),
            'by_negative_type': {k: {'n': v['n'], 'accepted': v['accepted']}
                                 for k, v in sorted(by_type.items())},
        },
        'unknown_test_per_text': {
            text: {'n': v['n'], 'n_accepted': v['n_accepted'],
                   'max_prob': round(v['max_prob'], 4),
                   'pred_distribution': dict(v['pred'])}
            for text, v in sorted(text_stats.items())
        },
        'limitations': [
            f'未知测试集仅 {len(unk_test_sc)} 条 / {len(text_stats)} 个文本，'
            'FAR 的 95% 置信上界约 8.3%（rule of three），观测 0% 不等于可靠安全。',
            '未知样本为合成 TTS（3 个 edge 音色），部分与已知命令共享粤语词汇，'
            '不能外推到真人语音或任意未登记命令。',
            '阈值选择规则（min_coverage/target_far）是预注册的工程折中，'
            '实际部署应按使用场景调整并重新评估。',
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'selected_threshold': thr,
                      'selected_on_validation': out['selected_on_validation'],
                      'test_results': out['test_results']},
                     ensure_ascii=False, indent=2))
    print('saved', out_path)

    if not args.no_write_thresholds:
        thresholds_payload = {
            'format': 'v3_thresholds/1.0',
            'selected_threshold': thr,
            'score': 'max_prob',
            'temperature': model.temperature,
            'model_sha256': out['protocol']['model_sha256'],
            'selected_on': 'validation + unknown calibration',
            'known_coverage_on_validation': chosen['known_coverage'],
            'unknown_false_accept_rate_on_validation': chosen['unknown_false_accept_rate'],
            'test_metrics_reference': {
                'known_coverage': len(acc_te) / len(test_sc),
                'accepted_accuracy': accepted_acc,
                'unknown_false_accept_rate': len(acc_un) / len(unk_test_sc),
                'n_known_test': len(test_sc), 'n_unknown_test': len(unk_test_sc),
            },
            'notes': ('阈值只在验证集+未知校准集选择；test 指标仅作参考记录，'
                      '不参与阈值选择。'),
            'generated_by': 'scripts/evaluate_v3_rejection.py',
        }
        tp = Path(args.thresholds_out)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(thresholds_payload, ensure_ascii=False, indent=2),
                      encoding='utf-8')
        print('thresholds saved', tp)


def accept_all(scores):
    return scores


if __name__ == '__main__':
    main()

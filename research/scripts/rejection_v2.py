# -*- coding: utf-8 -*-
"""第二阶段：拒识机制检查。

约束（来自任务要求）：
- 阈值只在**验证集**上选择；测试集只用于最终评估。
- 未知命令样本不足时，结论必须限定适用范围。

本脚本评估的四个量：
  1. 已知命令覆盖率 known_coverage   = #(已知样本置信度 >= 阈值) / #已知样本
  2. 已知命令误拒识率 known_false_reject_rate = 1 - known_coverage
  3. 被接受样本的准确率 accepted_accuracy
  4. 未知命令误接受率 unknown_false_accept_rate

阈值选择目标：在验证集上，先满足 known_coverage >= min_coverage，
再最小化 unknown_false_accept_rate（验证集上同时有已知与未知样本时）；
若无未知样本，则退化为在验证集上最大化 accepted_accuracy。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.config import load_config  # noqa: E402
from src.manifest import load_manifest  # noqa: E402
from src.dataset_v2 import sha256_file, INTENTS  # noqa: E402
from scripts.run_v2_experiments import (  # noqa: E402
    FeatureCache, build_templates, predict_one, confidence_from_dists,
    load_split, load_usable_records, subset, classification_report,
)

UNKNOWN_MANIFEST = BASE / 'data' / 'tts_unknown' / 'manifest.csv'
OUT_PATH = BASE / 'outputs' / 'v2_rejection.json'


def scores_for(records, templates, cache, config, max_frames=60):
    normalize = config.get('baselines', {}).get('dtw', {}).get('normalize', 'path')
    out = []
    for r in records:
        q = cache.get(r)
        pred, dists = predict_one(q, templates, normalize=normalize, max_frames=max_frames)
        conf = confidence_from_dists(dists)[pred]
        out.append({'utterance_id': r.utterance_id, 'command': r.command,
                    'voice': r.speaker_id, 'expression_group_id': r.expression_group_id,
                    'pred_command': pred, 'confidence': float(conf),
                    'correct': pred == r.command})
    return out


def rejection_table(known, unknown, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 1.0, 101), 3)
    rows = []
    for t in thresholds:
        ka = [s for s in known if s['confidence'] >= t]
        ua = [s for s in unknown if s['confidence'] >= t]
        rows.append({
            'threshold': float(t),
            'known_coverage': len(ka) / len(known) if known else 0.0,
            'known_false_reject_rate': 1.0 - (len(ka) / len(known) if known else 0.0),
            'accepted_accuracy': float(np.mean([s['correct'] for s in ka])) if ka else 0.0,
            'n_accepted': len(ka),
            'unknown_false_accept_rate': len(ua) / len(unknown) if unknown else None,
            'n_unknown': len(unknown),
        })
    return rows


def select_threshold(val_rows, min_coverage=0.5, target_far=0.10):
    """只在验证集上选阈值。"""
    has_unknown = any(r['unknown_false_accept_rate'] is not None for r in val_rows)
    if has_unknown:
        feasible = [r for r in val_rows
                    if r['known_coverage'] >= min_coverage
                    and r['unknown_false_accept_rate'] <= target_far]
        if feasible:
            # 满足 FAR 约束下优先覆盖率，其次接受准确率
            return max(feasible, key=lambda r: (r['known_coverage'],
                                                r['accepted_accuracy'],
                                                -r['unknown_false_accept_rate']))
        # 无可行解：在覆盖率允许范围内最小化 FAR
        by_cov = [r for r in val_rows if r['known_coverage'] >= min_coverage]
        pool = by_cov or val_rows
        return min(pool, key=lambda r: (r['unknown_false_accept_rate'], -r['known_coverage']))
    by_cov = [r for r in val_rows if r['known_coverage'] >= min_coverage] or val_rows
    return max(by_cov, key=lambda r: (r['accepted_accuracy'], r['known_coverage']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-coverage', type=float, default=0.5)
    ap.add_argument('--target-far', type=float, default=0.10)
    args = ap.parse_args()

    config = load_config(BASE / 'config.json')
    group2subset, split_meta = load_split()
    records = load_usable_records()
    train = subset(records, group2subset, 'train')
    val = subset(records, group2subset, 'val')
    test = subset(records, group2subset, 'test')

    cache = FeatureCache(config)
    templates = build_templates(train, cache, 'all', None, args.seed)

    unknown = load_manifest(UNKNOWN_MANIFEST, check_files=True) if UNKNOWN_MANIFEST.exists() else []
    # 未知样本按文本分成「校准/测试」两半：校准用于验证集阈值选择，
    # 测试用于最终报告，避免同一文本同时参与阈值选择和评估。
    texts = sorted({r.text for r in unknown})
    cal_texts = set(texts[: max(1, len(texts) // 2)])
    unk_cal = [r for r in unknown if r.text in cal_texts]
    unk_test = [r for r in unknown if r.text not in cal_texts]

    known_val = scores_for(val, templates, cache, config)
    unk_cal_sc = scores_for(unk_cal, templates, cache, config) if unk_cal else []
    known_test = scores_for(test, templates, cache, config)
    unk_test_sc = scores_for(unk_test, templates, cache, config) if unk_test else []

    # 验证集阈值曲线：已知来自验证集，未知来自未参与校准的文本组（统一在此选择一次）
    val_rows = rejection_table(known_val, unk_cal_sc)
    chosen = select_threshold(val_rows, args.min_coverage, args.target_far)
    thr = chosen['threshold']

    accepted_test = [s for s in known_test if s['confidence'] >= thr]
    accepted_unk = [s for s in unk_test_sc if s['confidence'] >= thr]

    base_metrics = classification_report([s['command'] for s in known_test],
                                         [s['pred_command'] for s in known_test])
    out = {
        'protocol': {
            'threshold_selected_on': 'validation split (track A) + 校准用未知文本',
            'test_used_for': 'final evaluation only, no parameter changes after this run',
            'confidence_definition': 'DTW 距离 softmax 置信度，与 src/baselines.py predict_dtw 一致',
            'split_sha256': split_meta['sha256'],
        },
        'sizes': {'train': len(train), 'val': len(val), 'test': len(test),
                  'unknown_cal': len(unk_cal), 'unknown_test': len(unk_test),
                  'unknown_texts_total': len(texts),
                  'unknown_cal_texts': len(cal_texts),
                  'unknown_test_texts': len(texts) - len(cal_texts)},
        'validation_curve': val_rows[::5],
        'selected_threshold': thr,
        'selected_on_validation': {
            'known_coverage': chosen['known_coverage'],
            'known_false_reject_rate': chosen['known_false_reject_rate'],
            'accepted_accuracy': chosen['accepted_accuracy'],
            'unknown_false_accept_rate': chosen['unknown_false_accept_rate'],
        },
        'test_results': {
            'no_threshold_accuracy': base_metrics['accuracy'],
            'no_threshold_macro_f1': base_metrics['macro_f1'],
            'known_coverage': len(accepted_test) / len(known_test) if known_test else 0.0,
            'known_false_reject_rate': 1.0 - (len(accepted_test) / len(known_test)
                                              if known_test else 0.0),
            'accepted_accuracy': float(np.mean([s['correct'] for s in accepted_test]))
            if accepted_test else 0.0,
            'n_known_accepted': len(accepted_test),
            'unknown_false_accept_rate': (len(accepted_unk) / len(unk_test_sc))
            if unk_test_sc else None,
            'unknown_rejection_rate': (1.0 - len(accepted_unk) / len(unk_test_sc))
            if unk_test_sc else None,
            'n_unknown_test': len(unk_test_sc),
        },
        'bootstrap_95ci_no_threshold': group_bootstrap(known_test),
        'per_sample_test': known_test,
        'limitations': [
            f'未知命令样本仅 {len(unknown)} 条合成 TTS（{len(texts)} 个不同文本），'
            '全部来自与已知命令相同的 3 个 TTS 音色，且部分文本与已知命令共享粤语词汇。'
            '未知误接受率不能外推到真人语音或任意未登记命令。',
            '已知命令测试集为合成 TTS 的表达留出集；拒识结论仅适用于该合成条件。',
            '若 unknown_test 为空，则未知误接受率不可估计。',
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('sizes', 'selected_threshold', 'test_results')},
                     ensure_ascii=False, indent=2))
    print('saved', OUT_PATH)


def group_bootstrap(rows, n_boot=2000, seed=42):
    from collections import defaultdict
    by_group = defaultdict(list)
    for r in rows:
        by_group[r['expression_group_id']].append(r)
    groups = sorted(by_group)
    if not groups:
        return None
    rng = np.random.RandomState(seed)
    accs = []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        sample = []
        for i in pick:
            sample.extend(by_group[groups[i]])
        accs.append(float(np.mean([s['correct'] for s in sample])))
    return {'accuracy_ci': [float(np.percentile(accs, 2.5)),
                            float(np.percentile(accs, 97.5))],
            'n_groups': len(groups)}


if __name__ == '__main__':
    main()

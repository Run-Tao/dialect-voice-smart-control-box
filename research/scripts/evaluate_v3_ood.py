# -*- coding: utf-8 -*-
"""第三阶段：OOD / 拒识分数对照（max-prob vs 嵌入距离）。

背景
----
``scripts/evaluate_v3_rejection.py`` 用温度标定后的 max-prob 作拒识分数。
在 v3.1 数据上发现：闭集 6 类 CNN 对「硬负例」（開冷氣/關電視/調暗啲燈 等）
仍会以高置信度判成已知类，max-prob 阈值在覆盖率约束下无法压低 FAR。
本脚本比较四类**无监督 OOD 分数**（只用训练集嵌入与验证集/校准集选择阈值）：

1. ``max_prob``：温度标定最大类概率（基线，与 rejection.json 一致）；
2. ``cos_centroid``：与各类训练嵌入 centroids 的最大余弦相似度；
3. ``mahalanobis``：到各类 centroids 的最小马氏距离（池化协方差 + 收缩）；
4. ``knn_distance``：到训练集 k 近邻的平均欧氏距离（越小越像已知）。

协议
----
* 阈值只在 val(known 440) + unknown_v3 calibration(36) 上选择；
* test known(483) + unknown test(36) 只评估一次；
* 主约束：known_coverage >= min_coverage (默认 0.85)，在此约束下最小化 FAR；
  若没有可行点，则报告 FAR 最小点（并显式标注不可行）。
* 同时报告每个分数的 AUC（known val vs unknown calibration），用于比较可分性。

产物：``outputs/v3_experiments/main_v3_1/ood.json``
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from scripts.evaluate_v3_rejection import (  # noqa: E402
    load_known, load_unknown, sha256_file, wilson_ci,
)
from src.v3_model import V3Predictor  # noqa: E402

INTENTS = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']


def embed_all(model, recs, log=print):
    embs = []
    t0 = __import__('time').time()
    for i, r in enumerate(recs):
        seq = model.feature_sequence(r['audio_path'])
        embs.append(model.embed_features(seq))
        if (i + 1) % 400 == 0:
            log(f'    嵌入 {i + 1}/{len(recs)} {__import__("time").time() - t0:.1f}s')
    return np.asarray(embs, dtype=np.float64)


def auc_score(pos, neg):
    """Score 越大越像已知；返回 AUC。"""
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def build_scores(model, known, unknown, log=print):
    tr = known['train']
    log('提取训练集嵌入（用于 centroids / kNN）...')
    Etr = embed_all(model, tr, log=log)
    ytr = np.asarray([INTENTS.index(r['command']) for r in tr])
    centroids = np.stack([Etr[ytr == c].mean(0) for c in range(6)])
    cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    # 池化类内协方差（收缩）
    resid = Etr - centroids[ytr]
    cov = np.cov(resid, rowvar=False)
    cov = (1 - 0.1) * cov + 0.1 * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    cov_inv = np.linalg.pinv(cov)

    def score_records(recs, tag):
        out = []
        log(f'提取 {tag} 嵌入...')
        E = embed_all(model, recs, log=log)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        cos = (En @ cn.T).max(1)
        maha = []
        for e in E:
            d = np.einsum('ij,jk,ik->i', centroids - e, cov_inv, centroids - e)
            maha.append(float(np.sqrt(np.min(d))))
        maha = np.asarray(maha)
        # kNN：到训练集最近 5 个的平均距离
        Etr_n = Etr / (np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-12)
        sims = En @ Etr_n.T
        k = min(5, Etr.shape[0])
        knn = []
        for i in range(En.shape[0]):
            idx = np.argpartition(-sims[i], k - 1)[:k]
            dist = np.linalg.norm(E[i] - Etr[idx], axis=1)
            knn.append(float(dist.mean()))
        knn = np.asarray(knn)
        for r, c, m, kn in zip(recs, cos, maha, knn):
            pred = model.predict(r['audio_path'])
            out.append({
                'utterance_id': r['utterance_id'],
                'command': r.get('command'),
                'pred_command': pred.command,
                'max_prob': float(pred.confidence),
                'cos_centroid': float(c),
                'mahalanobis': float(-m),  # 越大越像已知
                'knn_distance': float(-kn),
                'correct': (r.get('command') == pred.command) if r.get('command') else None,
            })
        return out

    val_scores = score_records(known['val'], 'val')
    test_scores = score_records(known['test'], 'test')
    cal_scores = score_records(unknown['calibration'], 'unknown_cal')
    unk_scores = score_records(unknown['test'], 'unknown_test')
    return val_scores, test_scores, cal_scores, unk_scores


def threshold_curve(known, unknown, score_key, thresholds):
    rows = []
    for t in thresholds:
        ka = [s for s in known if s[score_key] >= t]
        ua = [s for s in unknown if s[score_key] >= t]
        rows.append({
            'threshold': float(t),
            'known_coverage': len(ka) / len(known),
            'accepted_accuracy': float(np.mean([s['correct'] for s in ka])) if ka else 0.0,
            'unknown_false_accept_rate': len(ua) / len(unknown),
            'n_known_accepted': len(ka),
            'n_unknown_accepted': len(ua),
        })
    return rows


def select(curve, min_coverage):
    feasible = [r for r in curve if r['known_coverage'] >= min_coverage]
    if feasible:
        # 先最小化 FAR，再最大化覆盖率
        return min(feasible, key=lambda r: (r['unknown_false_accept_rate'],
                                            -r['known_coverage'],
                                            -r['accepted_accuracy']))
    return min(curve, key=lambda r: (r['unknown_false_accept_rate'], -r['known_coverage']))


def evaluate_score(name, direction, val_s, test_s, cal_s, unk_s, min_coverage, n_grid=200):
    """direction='high'：分数越大越像已知；'low'：越小越像已知（取负后统一）。"""
    def key(s):
        return s[name] if direction == 'high' else -s[name]
    vv = [{'score': key(s), **s} for s in val_s]
    tt = [{'score': key(s), **s} for s in test_s]
    cc = [{'score': key(s), **s} for s in cal_s]
    uu = [{'score': key(s), **s} for s in unk_s]
    # 阈值网格：在 val known 与 cal unknown 的分数分布范围内
    allv = np.asarray([s['score'] for s in vv] + [s['score'] for s in cc])
    thresholds = np.linspace(float(np.min(allv)), float(np.max(allv)), n_grid)
    curve = threshold_curve(vv, cc, 'score', thresholds)
    chosen = select(curve, min_coverage)
    thr = chosen['threshold']
    acc_te = [s for s in tt if s['score'] >= thr]
    acc_unk = [s for s in uu if s['score'] >= thr]
    return {
        'score': name,
        'direction': direction,
        'selected_threshold_raw': (thr if direction == 'high' else -thr),
        'selected_on_validation': chosen,
        'auc_known_val_vs_unknown_cal': auc_score([s['score'] for s in vv],
                                                  [s['score'] for s in cc]),
        'test': {
            'known_coverage': len(acc_te) / len(tt),
            'accepted_accuracy': float(np.mean([s['correct'] for s in acc_te])) if acc_te else 0.0,
            'unknown_false_accept_rate': len(acc_unk) / len(uu),
            'unknown_rejection_rate': 1 - len(acc_unk) / len(uu),
            'far_wilson95': wilson_ci(len(acc_unk), len(uu)),
            'n_known': len(tt), 'n_unknown': len(uu),
        },
        'curve_sample': curve[::max(1, len(curve) // 20)],
    }


def main():
    ap = argparse.ArgumentParser(description='v3 OOD/拒识分数对照')
    ap.add_argument('--model', default=str(BASE / 'data/models/v3_cnn_ensemble.npz'))
    ap.add_argument('--config', default=str(BASE / 'config_v3.json'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--min-coverage', type=float, default=0.85)
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    model = V3Predictor.load(args.model)
    known = load_known(cfg)
    unknown = load_unknown()
    val_s, test_s, cal_s, unk_s = build_scores(model, known, unknown, log=log)

    results = {
        'protocol': {
            'model': args.model,
            'model_sha256': sha256_file(Path(args.model)),
            'threshold_selected_on': 'val(known 440) + unknown calibration(36)',
            'test_used_for': 'final evaluation only',
            'constraint': f'known_coverage >= {args.min_coverage}，最小化 FAR；无可行点则报告 FAR 最小点',
            'scores': {
                'max_prob': '温度标定后最大类概率（越大越像已知）',
                'cos_centroid': '与训练集各类 centroid 的最大余弦相似度（越大越像已知）',
                'mahalanobis': '到训练集类 centroid 的最小马氏距离（取负后越大越像已知）',
                'knn_distance': '到训练集 k=5 近邻的平均欧氏距离（取负后越大越像已知）',
            },
        },
        'n': {'known_val': len(val_s), 'known_test': len(test_s),
              'unknown_cal': len(cal_s), 'unknown_test': len(unk_s)},
        'scores': {},
    }
    for name, direction in [('max_prob', 'high'), ('cos_centroid', 'high'),
                            ('mahalanobis', 'high'), ('knn_distance', 'high')]:
        log(f'评估分数: {name}')
        results['scores'][name] = evaluate_score(name, direction, val_s, test_s,
                                                 cal_s, unk_s, args.min_coverage)
    out = Path(args.out) if args.out else (
        BASE / 'outputs/v3_experiments/main_v3_1/ood.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {k: {'auc': v['auc_known_val_vs_unknown_cal'],
                   'val_coverage': v['selected_on_validation']['known_coverage'],
                   'val_far': v['selected_on_validation']['unknown_false_accept_rate'],
                   'test_coverage': v['test']['known_coverage'],
                   'test_accepted_acc': v['test']['accepted_accuracy'],
                   'test_far': v['test']['unknown_false_accept_rate']}
               for k, v in results['scores'].items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('saved', out)


if __name__ == '__main__':
    main()

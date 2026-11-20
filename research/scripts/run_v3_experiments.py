# -*- coding: utf-8 -*-
"""第三阶段：正式对照实验（E1–E6/E9/E10）。

实验协议
--------
* 数据：``data/tts_corpus/manifest_yue_v3_1.csv`` + 固定划分
  ``data/splits/track_a_expression_split_v3_1.csv``（test 31 组 / 483 条，
  与 v3 完全一致；v3.1 只新增 train 的 correction 语料）。
* 选择纪律：所有超参/融合权重/TTA 配置只在 **val** 上选择；
  test 只用于最终一次性评估（每个候选都报告 test，但结论以 val 选择为准）。
* 评估两个测试视图：
  - full：v3.1 全部 test（483 条，含 4 个语速/音高条件变体 + 基础条件）；
  - matched_v2：仅 v2 条件（text_morphology/rate_slow/legacy，111 条），
    与二期报告在相同音频上可比。
* 统计：配对 McNemar 精确检验 + 按表达组重采样的自助法 CI；
  条件/语域分层准确率；逐类 F1。

实验列表
--------
E1  v2 基线：MFCC13 + CMVN + DTW 全模板（v2 train 397 条），max_frames=60；
E2  经典判别式：MFCC+Δ+ΔΔ 统计嵌入 + 多项逻辑回归（v3.1 train，l2 在 val 选）；
E3  CNN 集成（3 种子）：v3.1 train，验证集早停 + 温度标定（训练脚本完成）；
E4  CNN + TTA（时间平移，配置在 val 选）；
E5  CNN 与经典模型概率融合（融合权重在 val 选）；
E6  统计检验：E1 vs E3、E2 vs E3（McNemar + 分组自助法）；
E9  延迟与模型体积；
E10 错误分析（混淆矩阵、主要错误对、correction 语域）。

运行示例::

    python scripts/run_v3_experiments.py --run-name main_v3_1
    python scripts/run_v3_experiments.py --experiments E1 E2 --run-name main_v3_1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.dataset_v2 import sha256_file  # noqa: E402
from src.features import extract_mfcc  # noqa: E402
from src.features_v3 import (  # noqa: E402
    FeatureSpec, extract_sequence, stats_embedding,
)
from src.v3_classifier import MultinomialLogisticRegression  # noqa: E402
from src.v3_model import V3Predictor  # noqa: E402

INTENTS = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']
V2_CONDITIONS = {'text_morphology', 'rate_slow', 'legacy', 'baseline_v1', ''}


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_audio(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for cand in (BASE / p, BASE / 'data' / 'tts_corpus' / p):
        if cand.exists():
            return cand
    return BASE / p


def load_v3_records(cfg):
    data = cfg['data']
    manifest = BASE / data['manifest']
    split = BASE / data['split_file']
    with open(split, encoding='utf-8-sig', newline='') as f:
        group2subset = {r['expression_group_id']: r['subset'] for r in csv.DictReader(f)}
    recs = []
    with open(manifest, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            gid = row.get('expression_group_id', '')
            if gid not in group2subset:
                continue
            if (row.get('quality_status') or 'ok') != 'ok':
                continue
            if row['command'] not in INTENTS:
                continue
            recs.append({
                'utterance_id': row['utterance_id'],
                'audio_path': str(resolve_audio(row['audio_path'])),
                'command': row['command'],
                'subset': group2subset[gid],
                'expression_group_id': gid,
                'register': row.get('register', ''),
                'condition': row.get('condition', ''),
                'voice': row.get('voice_identity', row.get('speaker_id', '')),
                'text': row.get('text', ''),
            })
    return recs


def load_v2_records():
    """v2 训练语料（用于复现 E1 基线）。"""
    manifest = BASE / 'data/tts_corpus/manifest_yue_v2.csv'
    split = BASE / 'data/splits/track_a_expression_split.csv'
    with open(split, encoding='utf-8-sig', newline='') as f:
        group2subset = {r['expression_group_id']: r['subset'] for r in csv.DictReader(f)}
    recs = []
    with open(manifest, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            gid = row.get('expression_group_id', '')
            if gid not in group2subset:
                continue
            recs.append({
                'utterance_id': row['utterance_id'],
                'audio_path': str(resolve_audio(row['audio_path'])),
                'command': row['command'],
                'subset': group2subset[gid],
                'expression_group_id': gid,
                'register': row.get('register', ''),
                'condition': row.get('condition', ''),
                'voice': row.get('voice_identity', row.get('speaker_id', '')),
                'text': row.get('text', ''),
            })
    return recs


def macro_f1(y, p, n_cls=6):
    out = []
    for c in range(n_cls):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(out))


def full_metrics(recs, y_true, y_pred, probabilities=None):
    """逐类指标 + 混淆矩阵 + 分层准确率。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = INTENTS
    per_class = {}
    for i, c in enumerate(labels):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {'precision': round(prec, 4), 'recall': round(rec, 4),
                        'f1': round(f1, 4), 'support': int(np.sum(y_true == i))}
    cm = np.zeros((6, 6), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    strat = {}
    for key in ('condition', 'register', 'voice'):
        groups = defaultdict(list)
        for r, ok in zip(recs, (y_true == y_pred)):
            groups[r[key]].append(bool(ok))
        strat[key] = {k: {'n': len(v), 'accuracy': round(float(np.mean(v)), 4)}
                      for k, v in sorted(groups.items())}
    return {
        'n': int(len(y_true)),
        'accuracy': round(float(np.mean(y_true == y_pred)), 4),
        'macro_f1': round(macro_f1(y_true, y_pred), 4),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'stratified_accuracy': strat,
        'n_expression_groups': len({r['expression_group_id'] for r in recs}),
    }


def mcnemar_exact(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def paired_compare(recs, pred_a, pred_b, name_a='A', name_b='B', seed=42, n_boot=5000):
    """同一测试集上两个模型的配对比较（McNemar + 按表达组自助法）。"""
    y = np.asarray([INTENTS.index(r['command']) for r in recs])
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    b01 = int(np.sum(a_ok & ~b_ok))
    b10 = int(np.sum(~a_ok & b_ok))
    p = mcnemar_exact(b01, b10)
    by_group = defaultdict(list)
    for r, x, z in zip(recs, a_ok, b_ok):
        by_group[r['expression_group_id']].append((int(x), int(z)))
    groups = sorted(by_group)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        sa = sz = n = 0
        for i in pick:
            for x, z in by_group[groups[i]]:
                sa += x; sz += z; n += 1
        diffs.append((sz - sa) / n)
    diffs = np.asarray(diffs)
    return {
        f'{name_a}_only_correct': b01,
        f'{name_b}_only_correct': b10,
        'both_correct': int(np.sum(a_ok & b_ok)),
        'neither_correct': int(np.sum(~a_ok & ~b_ok)),
        'accuracy_delta_b_minus_a': round(float(np.mean(b_ok) - np.mean(a_ok)), 4),
        'mcnemar_exact_p_two_sided': p,
        'group_bootstrap': {
            'acc_diff_mean': round(float(diffs.mean()), 4),
            'ci95': [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            'p_two_sided_bootstrap': float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))),
            'n_groups': len(groups),
        },
        'interpretation': ('差异显著（McNemar p<0.05）' if p < 0.05 else '差异不显著（McNemar p>=0.05）'),
    }


def split_recs(recs):
    return {s: [r for r in recs if r['subset'] == s] for s in ('train', 'val', 'test')}


def v2_view(recs):
    return [r for r in recs if r['condition'] in V2_CONDITIONS]


# ---------------------------------------------------------------------------
# E1：v2 基线（MFCC + DTW）
# ---------------------------------------------------------------------------

def make_dtw_funcs():
    from numba import njit

    @njit(cache=True)
    def _dtw_pair(x, y):
        n = x.shape[0]; m = y.shape[0]
        D = np.empty((n, m), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                s = 0.0
                for d in range(x.shape[1]):
                    t = x[i, d] - y[j, d]
                    s += t * t
                D[i, j] = np.sqrt(s)
        C = np.empty((n + 1, m + 1), dtype=np.float64)
        C[0, 0] = 0.0
        for j in range(1, m + 1):
            C[0, j] = np.inf
        for i in range(1, n + 1):
            C[i, 0] = np.inf
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                b = C[i - 1, j]
                if C[i, j - 1] < b:
                    b = C[i, j - 1]
                if C[i - 1, j - 1] < b:
                    b = C[i - 1, j - 1]
                C[i, j] = D[i - 1, j - 1] + b
        i = n; j = m; steps = 0
        while i > 0 or j > 0:
            steps += 1
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                a = C[i - 1, j - 1]; b = C[i - 1, j]; c = C[i, j - 1]
                if a <= b and a <= c:
                    i -= 1; j -= 1
                elif b <= c:
                    i -= 1
                else:
                    j -= 1
        return C[n, m] / max(steps, 1)

    @njit(cache=True)
    def _dist_matrix(Q, T):
        nq = Q.shape[0]; nt = T.shape[0]
        out = np.empty((nq, nt), dtype=np.float64)
        for a in range(nq):
            for b in range(nt):
                out[a, b] = _dtw_pair(Q[a], T[b])
        return out

    return _dist_matrix


def downsample(seq, max_frames):
    if max_frames is None or seq.shape[0] <= max_frames:
        return np.asarray(seq, dtype=np.float64)
    idx = np.linspace(0, seq.shape[0] - 1, int(max_frames)).astype(int)
    return np.asarray(seq[idx], dtype=np.float64)


def mfcc_sequences(recs, sample_rate=16000, max_frames=60):
    out = []
    t0 = time.time()
    for r in recs:
        from src.audio_io import load_wav
        audio, sr = load_wav(r['audio_path'])
        seq = extract_mfcc(audio, sr, n_fft=512, hop_length=160, n_mfcc=13, n_mels=26,
                           fmin=50.0, fmax=8000.0, preemphasis=0.97, use_cmvn=True)
        seq = downsample(seq, max_frames)
        if seq.shape[0] < max_frames:  # 过短音频：边缘复制到固定帧数，便于批量 DTW
            pad = np.repeat(seq[-1:], max_frames - seq.shape[0], axis=0)
            seq = np.concatenate([seq, pad], axis=0)
        out.append(seq)
    print(f'    MFCC 特征 {len(recs)} 条 {time.time() - t0:.1f}s', flush=True)
    return out


def run_e1(recs_by_split, v2_by_split, log=print):
    """v2 基线：只用 v2 train 的 397 条 MFCC+DTW 模板，评估 v3.1 test。"""
    log('E1: 提取 v2 train / v3.1 test 的 MFCC（max_frames=60）...')
    train_v2 = v2_by_split['train']
    test = recs_by_split['test']
    T = mfcc_sequences(train_v2)
    Q = mfcc_sequences(test)
    dist_matrix = make_dtw_funcs()
    Tm = np.stack(T)  # [Nt, 60, 13]
    Qm = np.stack(Q)
    log(f'E1: DTW 矩阵 {Qm.shape[0]}×{Tm.shape[0]} ...')
    t0 = time.time()
    D = dist_matrix(Qm, Tm)
    log(f'E1: DTW 完成 {time.time() - t0:.1f}s')
    y_train = np.asarray([INTENTS.index(r['command']) for r in train_v2])
    preds, confs = [], []
    for i in range(D.shape[0]):
        scores = np.array([D[i, y_train == c].min() for c in range(6)])
        preds.append(int(scores.argmin()))
        z = -(scores - scores.min())
        p = np.exp(z / 0.2); p /= p.sum()
        confs.append(float(p.max()))
    latency = (time.time() - t0) / max(len(test), 1)
    metrics = full_metrics(test, [INTENTS.index(r['command']) for r in test], preds)
    metrics['latency_sec_per_sample'] = round(latency, 4)
    metrics['n_templates'] = len(train_v2)
    return metrics, preds, confs


# ---------------------------------------------------------------------------
# E2：经典判别式（MFCC+Δ+ΔΔ 统计嵌入 + 逻辑回归）
# ---------------------------------------------------------------------------

CLASSICAL_SPEC = FeatureSpec(
    include_mfcc=True, include_delta=True, include_delta2=True, n_mels=26, n_mfcc=13,
    use_cmvn=True, include_logmel=False, include_pitch=False, include_energy=False)


def extract_classical(recs, log=print):
    X = []
    t0 = time.time()
    for r in recs:
        from src.audio_io import load_wav
        audio, sr = load_wav(r['audio_path'])
        seq = extract_sequence(audio, sr, CLASSICAL_SPEC)
        X.append(stats_embedding(seq))
    log(f'    经典特征 {len(recs)} 条 {time.time() - t0:.1f}s dim={len(X[0])}')
    return np.asarray(X, dtype=np.float64)


def run_e2(recs_by_split, log=print):
    tr, va, te = recs_by_split['train'], recs_by_split['val'], recs_by_split['test']
    log('E2: 提取经典统计特征...')
    Xtr = extract_classical(tr, log); Xva = extract_classical(va, log); Xte = extract_classical(te, log)
    ytr = np.asarray([INTENTS.index(r['command']) for r in tr])
    yva = np.asarray([INTENTS.index(r['command']) for r in va])
    yte = np.asarray([INTENTS.index(r['command']) for r in te])
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1]
    best = (-1, None, None)
    for l2 in grid:
        clf = MultinomialLogisticRegression(l2=l2, standardize=True, epochs=800, lr=0.5)
        t0 = time.time()
        clf.fit(Xtr, ytr)
        va_acc = float(np.mean(clf.predict(Xva) == yva))
        log(f'    l2={l2:g} val_acc={va_acc:.4f} ({time.time() - t0:.1f}s)')
        if va_acc > best[0]:
            best = (va_acc, l2, clf)
    val_acc, l2, clf = best
    t0 = time.time()
    proba = clf.predict_proba(Xte)
    latency = (time.time() - t0) / len(te)
    pred = proba.argmax(1)
    metrics = full_metrics(te, yte, pred)
    metrics['selected_l2'] = l2
    metrics['val_accuracy'] = val_acc
    metrics['latency_sec_per_sample'] = round(latency, 5)
    metrics['embedding_dim'] = int(Xtr.shape[1])
    val_proba = clf.predict_proba(Xva)
    return metrics, pred.tolist(), [float(x) for x in proba.max(1)], {
        'val_probs': val_proba.astype(np.float32), 'test_probs': proba.astype(np.float32),
        'val_labels': yva, 'test_labels': yte, 'l2': l2}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def cnn_feature_sequences(model, recs, log=print):
    seqs = []
    t0 = time.time()
    for r in recs:
        seqs.append(model.feature_sequence(r['audio_path']))
    log(f'    CNN 特征 {len(recs)} 条 {time.time() - t0:.1f}s')
    return seqs


def predict_cnn(model, recs, seqs=None, tta_shifts=None, log=print):
    if seqs is None:
        seqs = cnn_feature_sequences(model, recs, log=log)
    old_tta = list(model.tta_shifts)
    model.tta_shifts = list(tta_shifts or [])
    t0 = time.time()
    probs, preds = [], []
    for seq in seqs:
        p = model.predict_features(seq)
        preds.append(INTENTS.index(p.command))
        probs.append([p.scores[c] for c in INTENTS])
    model.tta_shifts = old_tta
    latency = (time.time() - t0) / max(len(recs), 1)
    return np.asarray(preds), np.asarray(probs, dtype=np.float64), latency


def main():
    ap = argparse.ArgumentParser(description='v3 正式对照实验')
    ap.add_argument('--config', default=str(BASE / 'config_v3.json'))
    ap.add_argument('--model', default=str(BASE / 'data/models/v3_cnn_ensemble.npz'))
    ap.add_argument('--run-name', default='main_v3_1')
    ap.add_argument('--experiments', nargs='+', default=['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E9', 'E10'])
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    cfg = load_config(args.config)
    recs = load_v3_records(cfg)
    recs_by_split = split_recs(recs)
    log(f'v3.1 数据: train={len(recs_by_split["train"])} val={len(recs_by_split["val"])} '
        f'test={len(recs_by_split["test"])}')

    out_dir = BASE / 'outputs' / 'v3_experiments' / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / 'results.json'
    results = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else {}
    results.setdefault('meta', {
        'run_name': args.run_name,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'config': args.config,
        'manifest': str(cfg['data']['manifest']),
        'split': str(cfg['data']['split_file']),
        'manifest_sha256': sha256_file(BASE / cfg['data']['manifest']),
        'split_sha256': sha256_file(BASE / cfg['data']['split_file']),
        'model': args.model,
        'n_test_full': len(recs_by_split['test']),
        'n_test_matched_v2': len(v2_view(recs_by_split['test'])),
        'protocol': '超参/TTA/融合权重仅在 val 选择；test 用于最终评估。',
    })

    preds_store = {}
    # ---------------- E1 ----------------
    if 'E1' in args.experiments:
        log('=== E1 v2 基线 (MFCC+DTW) ===')
        v2_by_split = split_recs(load_v2_records())
        metrics, pred_full, conf = run_e1(recs_by_split, v2_by_split, log=log)
        test = recs_by_split['test']
        matched_idx = [i for i, r in enumerate(test) if r['condition'] in V2_CONDITIONS]
        metrics_matched = full_metrics([test[i] for i in matched_idx],
                                       [INTENTS.index(test[i]['command']) for i in matched_idx],
                                       [pred_full[i] for i in matched_idx])
        results['E1_v2_dtw_baseline'] = {
            'full_test': metrics, 'matched_v2_test': metrics_matched,
            'model': 'MFCC13+CMVN, DTW all templates(path norm, max_frames=60), class-min',
            'train_size': len(v2_by_split['train']),
        }
        preds_store['E1'] = pred_full
        np.savez_compressed(out_dir / 'preds_E1.npz', preds=np.asarray(pred_full, dtype=np.int64))
        _save_predictions(out_dir, 'E1_v2_dtw_baseline', test, pred_full, conf)

    # ---------------- E2 ----------------
    if 'E2' in args.experiments:
        log('=== E2 经典判别式 (MFCC+Δ+ΔΔ stats + logreg) ===')
        metrics, pred, conf, payload = run_e2(recs_by_split, log=log)
        test = recs_by_split['test']
        matched_idx = [i for i, r in enumerate(test) if r['condition'] in V2_CONDITIONS]
        metrics_matched = full_metrics([test[i] for i in matched_idx],
                                       [INTENTS.index(test[i]['command']) for i in matched_idx],
                                       [pred[i] for i in matched_idx])
        results['E2_classical_logreg'] = {
            'full_test': metrics, 'matched_v2_test': metrics_matched,
            'model': 'MFCC13+Δ+ΔΔ 统计嵌入(mean/std/min/max/p10/p90) + 多项逻辑回归',
            'train_size': len(recs_by_split['train']),
        }
        preds_store['E2'] = pred
        np.savez_compressed(out_dir / 'preds_E2.npz', preds=np.asarray(pred, dtype=np.int64))
        _save_predictions(out_dir, 'E2_classical_logreg', test, pred, conf)
        np.savez_compressed(out_dir / 'E2_probs.npz', **payload)

    # ---------------- E3/E4/E5 ----------------
    need_cnn = any(e in args.experiments for e in ('E3', 'E4', 'E5', 'E6', 'E9', 'E10'))
    if need_cnn:
        model = V3Predictor.load(args.model)
        log(f'=== CNN 集成：{model.model_name} 温度={model.temperature:.3f} ===')
        val, test = recs_by_split['val'], recs_by_split['test']
        val_seqs = cnn_feature_sequences(model, val, log=log)
        test_seqs = cnn_feature_sequences(model, test, log=log)
        yva = np.asarray([INTENTS.index(r['command']) for r in val])
        yte = np.asarray([INTENTS.index(r['command']) for r in test])

        # E4 TTA 选择（仅 val）
        tta_candidates = {'none': [], 'pm4': [-4, 0, 4], 'pm8': [-8, 0, 8],
                          'pm12': [-12, -6, 0, 6, 12]}
        tta_val = {}
        for name, shifts in tta_candidates.items():
            pv, prob_v, _ = predict_cnn(model, val, seqs=val_seqs, tta_shifts=shifts, log=log)
            tta_val[name] = float(np.mean(pv == yva))
            log(f'  TTA {name:5s} val_acc={tta_val[name]:.4f}')
        best_tta = max(tta_val, key=tta_val.get)
        log(f'  TTA 选择（val）: {best_tta}')

        # E3 无 TTA 结果（协议基线）
        pred_va0, prob_va0, lat_va0 = predict_cnn(model, val, seqs=val_seqs, tta_shifts=[], log=log)
        pred_te0, prob_te0, lat_te0 = predict_cnn(model, test, seqs=test_seqs, tta_shifts=[], log=log)
        m_te0 = full_metrics(test, yte, pred_te0)
        matched_idx = [i for i, r in enumerate(test) if r['condition'] in V2_CONDITIONS]
        m_te0_matched = full_metrics([test[i] for i in matched_idx],
                                     [INTENTS.index(test[i]['command']) for i in matched_idx],
                                     [int(pred_te0[i]) for i in matched_idx])
        m_va0 = full_metrics(val, yva, pred_va0)
        results['E3_cnn_ensemble'] = {
            'full_test': m_te0, 'matched_v2_test': m_te0_matched, 'val': m_va0,
            'model': f'log-mel40+Δ, 1D-CNN×{len(model._seed_params)} seeds, 温度标定',
            'train_size': len(recs_by_split['train']),
            'temperature': model.temperature,
            'val_accuracy': float(np.mean(pred_va0 == yva)),
            'latency_sec_per_sample_inference': round(float(lat_te0), 5),
        }
        preds_store['E3'] = pred_te0.tolist()
        np.savez_compressed(out_dir / 'preds_E3.npz',
                            preds=np.asarray(pred_te0, dtype=np.int64))
        _save_predictions(out_dir, 'E3_cnn', test, pred_te0.tolist(),
                          [float(prob_te0[i].max()) for i in range(len(test))])

        # E4 最佳 TTA 的 test 结果
        if best_tta != 'none':
            pred_te_tta, prob_te_tta, _ = predict_cnn(model, test, seqs=test_seqs,
                                                      tta_shifts=tta_candidates[best_tta], log=log)
        else:
            pred_te_tta, prob_te_tta = pred_te0, prob_te0
        m_te_tta = full_metrics(test, yte, pred_te_tta)
        results['E4_cnn_tta'] = {
            'selected_on_val': best_tta,
            'val_candidates': tta_val,
            'full_test': m_te_tta,
            'selected_val_accuracy': tta_val[best_tta],
        }
        preds_store['E4'] = pred_te_tta.tolist()
        np.savez_compressed(out_dir / 'E3_E4_cnn_probs.npz',
                            val_probs=prob_va0.astype(np.float32),
                            test_probs=prob_te0.astype(np.float32),
                            tta_test_probs=prob_te_tta.astype(np.float32),
                            tta_shifts=np.asarray(tta_candidates[best_tta], dtype=np.int64),
                            val_labels=yva, test_labels=yte)

        # E5 融合（CNN 与经典 logreg；权重在 val 选）
        e2_npz = out_dir / 'E2_probs.npz'
        if e2_npz.exists():
            e2 = np.load(e2_npz)
            prob_cnn_va = prob_va0
            prob_cnn_te = prob_te0
            prob_cls_va = e2['val_probs'].astype(np.float64)
            prob_cls_te = e2['test_probs'].astype(np.float64)
            best = (-1.0, None)
            for w in np.linspace(0.0, 1.0, 11):
                pv = w * prob_cnn_va + (1 - w) * prob_cls_va
                acc = float(np.mean(pv.argmax(1) == yva))
                if acc > best[0]:
                    best = (acc, float(w))
            w = best[1]
            pt = w * prob_cnn_te + (1 - w) * prob_cls_te
            pred_f = pt.argmax(1)
            metrics_f = full_metrics(test, yte, pred_f)
            metrics_f_matched = full_metrics([test[i] for i in matched_idx],
                                             [INTENTS.index(test[i]['command']) for i in matched_idx],
                                             [int(pred_f[i]) for i in matched_idx])
            results['E5_fusion'] = {
                'selected_on_val': {'cnn_weight': w, 'val_accuracy': best[0]},
                'full_test': metrics_f, 'matched_v2_test': metrics_f_matched,
            }
            preds_store['E5'] = pred_f.tolist()
            np.savez_compressed(out_dir / 'preds_E5.npz',
                                preds=np.asarray(pred_f, dtype=np.int64))
            _save_predictions(out_dir, 'E5_fusion_final', test, pred_f.tolist(),
                              [float(pt[i].max()) for i in range(len(test))])
            final_pred, final_prob = pred_f, pt
        else:
            log('E5 跳过：缺少 E2_probs.npz（先跑 E2）')
            final_pred, final_prob = pred_te_tta, prob_te_tta

        # E6 统计检验
        e1_pred = preds_store.get('E1')
        if e1_pred is None and (out_dir / 'preds_E1.npz').exists():
            e1_pred = np.load(out_dir / 'preds_E1.npz')['preds'].tolist()
        if e1_pred is not None and 'E3_cnn_ensemble' in results:
            matched_idx = [i for i, r in enumerate(test) if r['condition'] in V2_CONDITIONS]
            results['E6_statistics'] = {
                'E1_vs_E3_full': paired_compare(test, e1_pred, pred_te0.tolist(),
                                                'E1_v2_dtw', 'E3_cnn'),
                'E1_vs_E3_matched_v2': paired_compare([test[i] for i in matched_idx],
                                                      [e1_pred[i] for i in matched_idx],
                                                      [int(pred_te0[i]) for i in matched_idx],
                                                      'E1_v2_dtw', 'E3_cnn'),
            }
            e2_pred = preds_store.get('E2')
            if e2_pred is None and (out_dir / 'preds_E2.npz').exists():
                e2_pred = np.load(out_dir / 'preds_E2.npz')['preds'].tolist()
            if e2_pred is not None:
                results['E6_statistics']['E2_vs_E3_full'] = paired_compare(
                    test, e2_pred, pred_te0.tolist(), 'E2_classical', 'E3_cnn')
                results['E6_statistics']['E2_vs_E3_matched_v2'] = paired_compare(
                    [test[i] for i in matched_idx], [e2_pred[i] for i in matched_idx],
                    [int(pred_te0[i]) for i in matched_idx], 'E2_classical', 'E3_cnn')
        else:
            log('E6 跳过：缺少 E1/E2 预测（先跑 E1/E2）')

        # E9 延迟与模型体积
        artifact = Path(args.model)
        results['E9_latency_size'] = {
            'model_size_mb': round(artifact.stat().st_size / 1e6, 3),
            'n_seeds': len(model._seed_params),
            'inference_sec_per_sample': round(float(lat_te0), 5),
            'feature_extraction_seconds_total': None,
            'note': 'inference_sec_per_sample 只含 CNN 前向（不含特征提取与文件IO）；'
                    '特征提取约 8–15ms/条（16k 2–4s 音频，numpy 单线程）。',
        }

        # E10 错误分析
        yt = yte
        final_name = 'E5_fusion' if 'E5' in preds_store else 'E4_cnn_tta'
        cm = np.zeros((6, 6), dtype=int)
        for a, b in zip(yt, final_pred):
            cm[a, b] += 1
        err = Counter()
        for r, a, b in zip(test, yt, final_pred):
            if a != b:
                err[(INTENTS[a], INTENTS[b])] += 1
        results['E10_error_analysis'] = {
            'model': final_name,
            'confusion_matrix': cm.tolist(),
            'labels': INTENTS,
            'top_error_pairs': [{'true': a, 'pred': b, 'count': n}
                                for (a, b), n in err.most_common(10)],
            'errors_by_register': dict(sorted(Counter(
                r['register'] for r, a, b in zip(test, yt, final_pred) if a != b).items())),
            'correction_register_accuracy': next(
                (v['accuracy'] for k, v in full_metrics(test, yt, final_pred)['stratified_accuracy']['register'].items()
                 if k == 'correction'), None),
        }
        # 保存每个 test 样本的最终预测，便于审计
        _save_predictions(out_dir, final_name + '_final', test, final_pred.tolist(),
                          [float(final_prob[i].max()) for i in range(len(test))])

    results['meta']['updated_at'] = datetime.now().isoformat(timespec='seconds')
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'run_manifest.json').write_text(json.dumps({
        'command': ' '.join(sys.argv),
        'config': args.config,
        'model': args.model,
        'manifest_sha256': results['meta']['manifest_sha256'],
        'split_sha256': results['meta']['split_sha256'],
        'created_at': results['meta']['created_at'],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'结果已写入 {results_path}')


def _save_predictions(out_dir, name, test, preds, confs):
    path = out_dir / f'predictions_{name}.csv'
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['utterance_id', 'expression_group_id', 'command', 'condition',
                    'register', 'voice', 'pred_command', 'correct', 'confidence'])
        for r, p, c in zip(test, preds, confs):
            w.writerow([r['utterance_id'], r['expression_group_id'], r['command'],
                        r['condition'], r['register'], r['voice'],
                        INTENTS[int(p)], int(INTENTS[int(p)] == r['command']), f'{c:.6f}'])


if __name__ == '__main__':
    main()

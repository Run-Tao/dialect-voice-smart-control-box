# -*- coding: utf-8 -*-
"""第二阶段对照实验运行器（MFCC + DTW 主方案）。

实验矩阵
--------
E1 基线：仅 v1 合格历史训练数据（固定测试集）
E2 数据规模：25% / 50% / 75% / 100% 训练表达组（验证集、测试集固定）
E3 模板数量：每类 1/2/3/5/10/N 个模板（从训练池中选取）
E4 模板选择：random / medoid(代表性) / all(min聚合) / 最新策略

协议约束
--------
- 测试集来自固定划分文件 data/splits/track_a_expression_split.csv（记录 sha256）。
- 所有超参（max_frames、模板数量、模板策略、拒识阈值）只在验证集选择；
  测试集在一次运行中被评估一次，运行后不允许基于测试结果回调参数。
- 结果写入 outputs/v2_experiments/<run_id>/，包含参数、划分、指标、混淆矩阵、
  每类 P/R/F1、推理耗时、运行命令。

用法::

    python scripts/run_v2_experiments.py --experiments E1 E2
    python scripts/run_v2_experiments.py --experiments all --seeds 42 7 2026
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.config import load_config  # noqa: E402
from src.manifest import load_manifest  # noqa: E402
from src.features import extract_mfcc  # noqa: E402
from src.audio_io import load_wav  # noqa: E402
from src.alignment import dtw_distance_nd  # noqa: E402
from src.metrics import confusion_matrix  # noqa: E402
from src.dataset_v2 import INTENTS, sha256_file, read_csv_rows, _hash_seed  # noqa: E402

OUT_ROOT = BASE / 'outputs' / 'v2_experiments'
SPLIT_PATH = BASE / 'data' / 'splits' / 'track_a_expression_split.csv'
MANIFEST_PATH = BASE / 'data' / 'tts_corpus' / 'manifest_yue_v2.csv'


# ---------------------------------------------------------------------------
# 特征缓存
# ---------------------------------------------------------------------------

class FeatureCache:
    """MFCC 序列缓存；特征参数完全来自 config.json，与既有实现一致。"""

    def __init__(self, config):
        self.cfg = config['features']
        self._cache = {}
        self.extract_seconds = 0.0

    def get(self, record):
        key = record.utterance_id
        if key not in self._cache:
            t0 = time.perf_counter()
            audio, sr = load_wav(record.audio_path)
            c = self.cfg
            self._cache[key] = extract_mfcc(
                audio, sr, n_fft=c['n_fft'], hop_length=c['hop_length'],
                n_mfcc=c['n_mfcc'], n_mels=c['n_mels'], fmin=c['fmin'],
                fmax=c['fmax'], preemphasis=c['preemphasis'], use_cmvn=c['use_cmvn'])
            self.extract_seconds += time.perf_counter() - t0
        return self._cache[key]

    def __len__(self):
        return len(self._cache)


# ---------------------------------------------------------------------------
# 模板构建与预测
# ---------------------------------------------------------------------------

def build_templates(records, cache, strategy, n_per_class, seed, medoid_order=None):
    """按策略从训练记录构造每类模板。

    strategy:
      - 'all'      : 全部训练样本作为模板（原项目默认，min 聚合）
      - 'random'   : 每类随机抽 n 个（唯一含随机性的策略）
      - 'medoid'   : 每类选到同类其他样本平均DTW距离最小的 n 个（代表性模板）
      - 'longest'  : 每类选音频时长最长的 n 个（覆盖更多韵律）

    medoid 需 O(n²) 次 DTW，调用方可传入预先算好的 `medoid_order`
    （{cmd: [按medoid排序的record列表]}）以复用，避免每个 k 重算。
    """
    by_class = defaultdict(list)
    for r in records:
        by_class[r.command].append(r)

    templates = {}
    for cmd, recs in by_class.items():
        recs = sorted(recs, key=lambda r: r.utterance_id)
        k = min(n_per_class, len(recs)) if n_per_class else len(recs)
        if strategy == 'all' or k >= len(recs):
            chosen = recs
        elif strategy == 'random':
            rng = np.random.RandomState(_hash_seed('tpl', seed, cmd) % (2 ** 32))
            idx = rng.permutation(len(recs))[:k]
            chosen = [recs[i] for i in sorted(idx)]
        elif strategy == 'medoid':
            if medoid_order is not None and cmd in medoid_order:
                ordered = medoid_order[cmd]
            else:
                ordered = _medoid_order(recs, cache)
            chosen = [r for r in ordered[:k]]
        elif strategy == 'longest':
            cand = sorted(recs, key=lambda r: -float(r.duration or 0))
            chosen = cand[:k]
        else:
            raise ValueError(strategy)
        templates[cmd] = [cache.get(r) for r in chosen]
    return templates


def _medoid_order(recs, cache):
    """返回按「到同类其他模板平均 DTW 距离」升序排列的记录。"""
    seqs = [cache.get(r) for r in recs]
    n = len(recs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance_nd(seqs[i], seqs[j], norm='path', max_frames=60)
            D[i, j] = D[j, i] = d
    order = np.argsort(D.mean(axis=1))
    return [recs[i] for i in order]


def compute_medoid_orders(records, cache):
    by_class = defaultdict(list)
    for r in records:
        by_class[r.command].append(r)
    return {cmd: _medoid_order(sorted(recs, key=lambda r: r.utterance_id), cache)
            for cmd, recs in by_class.items()}


def predict_one(query, templates, normalize='path', max_frames=60, aggregate='min'):
    """返回 (预测类别, 各类距离字典)。aggregate: min / mean_top2。"""
    dists = {}
    for cmd, tpls in templates.items():
        vals = sorted(dtw_distance_nd(query, t, norm=normalize, max_frames=max_frames)
                      for t in tpls)
        if aggregate == 'mean_top2' and len(vals) > 1:
            dists[cmd] = float(np.mean(vals[:2]))
        else:
            dists[cmd] = float(vals[0])
    best = min(dists, key=dists.get)
    return best, dists


def confidence_from_dists(dists, temperature=0.2):
    """与 src.baselines.predict_dtw 相同的 softmax 置信度定义，保证与既有链路一致。"""
    vals = np.array(list(dists.values()), dtype=np.float64)
    vals = vals - vals.min()
    prob = np.exp(-vals / max(temperature, 1e-6))
    prob = prob / prob.sum()
    return {c: float(p) for c, p in zip(dists.keys(), prob)}


def evaluate(records, templates, cache, config, max_frames, aggregate='min'):
    """在给定记录集上评估，返回指标、逐样本预测与时延。"""
    normalize = config.get('baselines', {}).get('dtw', {}).get('normalize', 'path')
    y_true, y_pred, rows = [], [], []
    t0 = time.perf_counter()
    for r in records:
        q = cache.get(r)
        pred, dists = predict_one(q, templates, normalize=normalize,
                                  max_frames=max_frames, aggregate=aggregate)
        conf = confidence_from_dists(dists)[pred]
        y_true.append(r.command)
        y_pred.append(pred)
        rows.append({'utterance_id': r.utterance_id, 'speaker_id': r.speaker_id,
                     'expression_group_id': r.expression_group_id,
                     'condition': r.condition, 'true_command': r.command,
                     'pred_command': pred, 'confidence': round(conf, 6),
                     'distance': round(dists[pred], 6),
                     'dists': {k: round(v, 6) for k, v in dists.items()}})
    elapsed = time.perf_counter() - t0
    metrics = classification_report(y_true, y_pred)
    metrics['inference_seconds_total'] = elapsed
    metrics['inference_seconds_per_sample'] = elapsed / max(len(records), 1)
    return metrics, rows


def classification_report(y_true, y_pred, labels=None):
    labels = list(labels or INTENTS)
    cm, labels = confusion_matrix(np.array(y_true), np.array(y_pred), labels)
    per_class = {}
    f1s = []
    for i, lab in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_class[lab] = {'precision': round(prec, 4), 'recall': round(rec, 4),
                          'f1': round(f1, 4), 'support': int(cm[i, :].sum())}
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if len(y_true) else 0.0
    return {'accuracy': acc, 'macro_f1': float(np.mean(f1s)),
            'labels': labels, 'confusion_matrix': cm.tolist(),
            'per_class': per_class}


# ---------------------------------------------------------------------------
# 划分读取
# ---------------------------------------------------------------------------

def load_split(path=SPLIT_PATH):
    rows = read_csv_rows(path)
    group2subset = {r['expression_group_id']: r['subset'] for r in rows}
    meta = {'split_file': str(path.relative_to(BASE)),
            'sha256': sha256_file(path)}
    return group2subset, meta


def load_usable_records():
    recs = load_manifest(MANIFEST_PATH, check_files=True)
    return [r for r in recs if (r.quality_status or 'ok') != 'excluded']


def subset(records, group2subset, name):
    return [r for r in records if group2subset.get(r.expression_group_id) == name]


# ---------------------------------------------------------------------------
# 实验定义
# ---------------------------------------------------------------------------

def experiment_E1(records, group2subset, cache, config, seed):
    """E1 基线：仅 v1 历史合格训练数据。"""
    train_all = subset(records, group2subset, 'train')
    test = subset(records, group2subset, 'test')
    val = subset(records, group2subset, 'val')
    train = [r for r in train_all if r.generation_batch == 'v1']
    out = {'train_origin_filter': 'generation_batch == v1',
           'train_size': len(train), 'val_size': len(val), 'test_size': len(test)}
    if not train:
        out['error'] = 'v1 训练样本为空，无法构造基线'
        return out
    templates = build_templates(train, cache, 'all', None, seed)
    mf = 60
    val_m, _ = evaluate(val, templates, cache, config, mf) if val else ({}, [])
    test_m, _ = evaluate(test, templates, cache, config, mf)
    out['templates_per_class'] = {k: len(v) for k, v in templates.items()}
    out['val_metrics'] = val_m
    out['test_metrics'] = test_m
    return out


def experiment_E2(records, group2subset, cache, config, seed, fractions=(0.25, 0.5, 0.75, 1.0)):
    """E2 数据规模：按训练表达组比例取子集；验证集与测试集固定不变。"""
    train_all = subset(records, group2subset, 'train')
    val = subset(records, group2subset, 'val')
    test = subset(records, group2subset, 'test')
    groups = sorted({r.expression_group_id for r in train_all})
    # 表达组 → 命令映射
    gid_cmd = {}
    for r in train_all:
        gid_cmd.setdefault(r.expression_group_id, r.command)
    results = []
    for frac in fractions:
        n = max(len(INTENTS), int(round(len(groups) * frac)))
        per_class = max(1, n // len(INTENTS))
        per_cmd = defaultdict(list)
        for g in groups:
            per_cmd[gid_cmd[g]].append(g)
        chosen = []
        for cmd in INTENTS:
            gs = sorted(per_cmd[cmd], key=lambda g: _hash_seed('size', seed, g))
            chosen.extend(gs[:per_class])
        train = [r for r in train_all if r.expression_group_id in set(chosen)]
        templates = build_templates(train, cache, 'all', None, seed)
        val_m, _ = evaluate(val, templates, cache, config, 60) if val else ({}, [])
        test_m, _ = evaluate(test, templates, cache, config, 60)
        results.append({
            'fraction': frac, 'train_groups': len(chosen), 'train_size': len(train),
            'train_by_command': dict(Counter(r.command for r in train)),
            'templates_per_class': {k: len(v) for k, v in templates.items()},
            'val_accuracy': val_m.get('accuracy'), 'val_macro_f1': val_m.get('macro_f1'),
            'test_accuracy': test_m['accuracy'], 'test_macro_f1': test_m['macro_f1'],
            'test_per_class_f1': {k: v['f1'] for k, v in test_m['per_class'].items()},
            'test_confusion_matrix': test_m['confusion_matrix'],
            'inference_seconds_per_sample': test_m['inference_seconds_per_sample'],
        })
    return {'fixed_val_size': len(val), 'fixed_test_size': len(test),
            'curve': results}


def experiment_E3(records, group2subset, cache, config, seed, k_list=(1, 2, 3, 5, 10, None)):
    """E3 模板数量：medoid 策略下每类模板数（medoid 排序只算一次，各 k 复用）。"""
    train = subset(records, group2subset, 'train')
    val = subset(records, group2subset, 'val')
    test = subset(records, group2subset, 'test')
    orders = compute_medoid_orders(train, cache)
    results = []
    for k in k_list:
        templates = build_templates(train, cache, 'medoid' if k else 'all', k, seed,
                                    medoid_order=orders if k else None)
        val_m, _ = evaluate(val, templates, cache, config, 60) if val else ({}, [])
        test_m, _ = evaluate(test, templates, cache, config, 60)
        results.append({
            'k_per_class': k if k else 'all',
            'templates_total': sum(len(v) for v in templates.values()),
            'val_accuracy': val_m.get('accuracy'), 'val_macro_f1': val_m.get('macro_f1'),
            'test_accuracy': test_m['accuracy'], 'test_macro_f1': test_m['macro_f1'],
            'inference_seconds_per_sample': test_m['inference_seconds_per_sample'],
            'test_per_class_f1': {k2: v['f1'] for k2, v in test_m['per_class'].items()},
        })
    return {'results': results}


def experiment_E4(records, group2subset, cache, config, seed, n_per_class=5, seeds=(42, 7, 2026)):
    """E4 模板选择策略 + 多种子 + 分组自助法置信区间。

    注意：只有 `random` 策略含随机性，需要多种子；
    `all` / `medoid` / `longest` 完全确定，只跑一次（多种子结果会完全相同，
    报告多种子会让读者误以为做过重复实验）。此点在结果中显式记录。
    """
    train = subset(records, group2subset, 'train')
    val = subset(records, group2subset, 'val')
    test = subset(records, group2subset, 'test')
    strategies = ['all', 'random', 'medoid', 'longest']
    orders = compute_medoid_orders(train, cache)
    results = {}
    for strat in strategies:
        deterministic = strat != 'random'
        seeds_used = [seeds[0]] if deterministic else list(seeds)
        per_seed = []
        for s in seeds_used:
            templates = build_templates(train, cache, strat, n_per_class, s,
                                        medoid_order=orders)
            test_m, rows = evaluate(test, templates, cache, config, 60)
            per_seed.append({'seed': s, 'test_accuracy': test_m['accuracy'],
                             'test_macro_f1': test_m['macro_f1'], 'rows': rows,
                             'inference_seconds_per_sample':
                                 test_m['inference_seconds_per_sample']})
        accs = [p['test_accuracy'] for p in per_seed]
        f1s = [p['test_macro_f1'] for p in per_seed]
        ci = group_bootstrap_ci(per_seed[0]['rows'], n_boot=2000, seed=seed)
        results[strat] = {
            'n_per_class': n_per_class if strat != 'all' else 'all',
            'deterministic': deterministic,
            'seeds_used': seeds_used,
            'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'macro_f1_mean': float(np.mean(f1s)), 'macro_f1_std': float(np.std(f1s)),
            'per_seed': [{k: v for k, v in p.items() if k != 'rows'} for p in per_seed],
            'group_bootstrap_95ci': ci,
            'test_per_class': per_seed[0]['rows'],
        }
    return {'strategies': results,
            'note': f'只有 random 使用多随机种子 {list(seeds)}；'
                    'all/medoid/longest 为确定性策略，单次运行即可。'}


def group_bootstrap_ci(rows, n_boot=2000, seed=42, alpha=0.05):
    """按表达组重采样的自助法（避免把同组高度相关样本当独立样本）。"""
    by_group = defaultdict(list)
    for r in rows:
        by_group[r['expression_group_id']].append(r)
    groups = sorted(by_group)
    rng = np.random.RandomState(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        sample = []
        for i in pick:
            sample.extend(by_group[groups[i]])
        y = [r['true_command'] for r in sample]
        p = [r['pred_command'] for r in sample]
        acc = float(np.mean(np.array(y) == np.array(p)))
        f1 = classification_report(y, p)['macro_f1']
        accs.append(acc)
        f1s.append(f1)
    return {
        'n_groups': len(groups),
        'accuracy_ci': [float(np.percentile(accs, 100 * alpha / 2)),
                        float(np.percentile(accs, 100 * (1 - alpha / 2)))],
        'macro_f1_ci': [float(np.percentile(f1s, 100 * alpha / 2)),
                        float(np.percentile(f1s, 100 * (1 - alpha / 2)))],
    }


def experiment_trackB(records, cache, config, seed, voices=None):
    """轨道B（辅助）：留一音色泛化。仅 3 个 edge 音色可做，样本少，属辅助证据。

    注意：不使用轨道A的表达组划分，因为轨道B关心的是音色外推；
    这里直接 hold out 一个 voice_identity 的全部样本作为测试集。
    """
    edge = [r for r in records if r.tts_engine == 'edge_tts']
    all_voices = sorted({r.voice_identity for r in edge})
    if voices is None:
        voices = all_voices
    results = {}
    for v in voices:
        test = [r for r in edge if r.voice_identity == v]
        train = [r for r in edge if r.voice_identity != v]
        if not test or not train:
            continue
        # 未见音色测试：训练只用其它音色。
        # 模板固定为 medoid 5/类：全模板方案在约 400 条训练样本下每次评估需数分钟，
        # 三个留出音色合计将明显超出可承受时间；该选择对所有留出音色一致，音色间可比。
        templates = _aux_templates(train, cache, seed)
        m, rows = evaluate(test, templates, cache, config, 60)
        # 同时报告「音色未见 + 文本表达也未见」的严格子集：
        # 测试样本的表达组若出现在训练集中，则该样本对本项不算严格未见表达
        train_groups = {r.expression_group_id for r in train}
        strict = [r for r in test if r.expression_group_id not in train_groups]
        if strict:
            m_strict, _ = evaluate(strict, templates, cache, config, 60)
        else:
            m_strict = None
        ci = group_bootstrap_ci(rows, n_boot=2000, seed=seed)
        results[v] = {
            'train_samples': len(train), 'test_samples': len(test),
            'train_voices': sorted({r.voice_identity for r in train}),
            'test_by_command': dict(Counter(r.command for r in test)),
            'test_accuracy': m['accuracy'], 'test_macro_f1': m['macro_f1'],
            'test_confusion_matrix': m['confusion_matrix'],
            'test_per_class': m['per_class'],
            'group_bootstrap_95ci': ci,
            'strict_unseen_expression': {
                'test_samples': len(strict),
                'test_accuracy': m_strict['accuracy'] if m_strict else None,
                'test_macro_f1': m_strict['macro_f1'] if m_strict else None,
            } if m_strict else None,
            'note': '合成TTS；单一留出音色；测试集规模小，仅作辅助证据',
        }
    return {
        'protocol': '留一音色（edge 三音色）；canto-tts 因训练/测试音色数不对称不参与',
        'independent_voice_count_in_corpus': 4,
        'voices': results,
        'limitation': '每个留出音色测试集仅约百条合成样本，且与训练共享同一批文本表达时'
                      '存在表达泄漏；strict_unseen_expression 子集进一步剔除了'
                      '训练集中出现过的表达组。',
    }


def experiment_feature_distance(records, group2subset, cache, config, seed):
    """max_frames 对照（模板固定 medoid 5/类以控制计算量）。"""
    train = subset(records, group2subset, 'train')
    val = subset(records, group2subset, 'val')
    test = subset(records, group2subset, 'test')
    templates = _aux_templates(train, cache, seed)

    def eval_with(max_frames):
        val_m, _ = evaluate(val, templates, cache, config, max_frames) if val else ({}, [])
        test_m, _ = evaluate(test, templates, cache, config, max_frames)
        return {'max_frames': max_frames,
                'val_accuracy': val_m.get('accuracy'),
                'val_macro_f1': val_m.get('macro_f1'),
                'test_accuracy': test_m['accuracy'],
                'test_macro_f1': test_m['macro_f1'],
                'inference_seconds_per_sample': test_m['inference_seconds_per_sample']}

    results = [eval_with(mf) for mf in (40, 60, 100)]
    # 说明：不扫描 None（不截断）与 150 帧以上。语料最长 4.64s≈462 帧，
    # DTW 复杂度随帧数平方增长（150²/60²≈6.25 倍，462²/60²≈59 倍），
    # 600 条语料下单次全量评估需数十小时，不可行。
    # 40/60/100 三档已足以判断「60 帧截断是否仍是瓶颈」。
    best = max(results, key=lambda r: (r['val_accuracy'] or 0, r['val_macro_f1'] or 0))
    return {'max_frames_sweep': results,
            'selected_on_validation': best['max_frames'],
            'template_strategy': 'medoid 5/类',
            'note': '选择只看验证集；不扫描 None/150+ 帧，因 DTW 代价随帧数平方增长'}


def experiment_normalize(records, group2subset, cache, config, seed):
    """距离归一化方式对照（path vs max_len）。"""
    train = subset(records, group2subset, 'train')
    val = subset(records, group2subset, 'val')
    test = subset(records, group2subset, 'test')
    templates = _aux_templates(train, cache, seed)
    out = {}
    for norm in ('path', 'max_len'):
        cfg = {**config, 'baselines': {**config.get('baselines', {}),
                                       'dtw': {**config.get('baselines', {}).get('dtw', {}),
                                               'normalize': norm}}}
        val_m, _ = evaluate(val, templates, cache, cfg, 60) if val else ({}, [])
        test_m, _ = evaluate(test, templates, cache, cfg, 60)
        out[norm] = {'val_accuracy': val_m.get('accuracy'),
                     'val_macro_f1': val_m.get('macro_f1'),
                     'test_accuracy': test_m['accuracy'],
                     'test_macro_f1': test_m['macro_f1']}
    out['template_strategy'] = 'medoid 5/类'
    return out


def _aux_templates(train, cache, seed, k=5):
    orders = compute_medoid_orders(train, cache)
    return build_templates(train, cache, 'medoid', k, seed, medoid_order=orders)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', nargs='+', default=['E1', 'E2', 'E3', 'E4'],
                    help='E1/E2/E3/E4/B/F/N 或 all')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--config', default='config.json')
    ap.add_argument('--run-name', default=None)
    args = ap.parse_args()
    if 'all' in args.experiments:
        args.experiments = ['E1', 'E2', 'E3', 'E4', 'B', 'F', 'N']

    config = load_config(BASE / args.config)
    group2subset, split_meta = load_split()
    records = load_usable_records()
    cache = FeatureCache(config)

    counts = {k: len(subset(records, group2subset, k)) for k in ('train', 'val', 'test')}
    group_counts = {k: len({r.expression_group_id for r in subset(records, group2subset, k)})
                    for k in ('train', 'val', 'test')}
    print('数据:', {k: counts[k] for k in counts}, '表达组:', group_counts)
    print('划分哈希:', split_meta['sha256'])

    run_id = args.run_name or datetime.now().strftime('run_%Y%m%d_%H%M%S')
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 增量合并：分段运行不同实验时，保留同一 run 目录下已有的实验结果，
    # 避免后跑的实验覆盖先跑的结果（例如先跑 E1/E2，再跑 E3/E4）。
    prior = {}
    prior_path = out_dir / 'results.json'
    if prior_path.exists():
        try:
            prior = json.loads(prior_path.read_text(encoding='utf-8'))
            print(f"合并已有结果: {list(prior.get('experiments', {}))}")
        except Exception:
            prior = {}

    common = {
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'git_commit': _git_commit(),
        'manifest': {'path': str(MANIFEST_PATH.relative_to(BASE)),
                     'sha256': sha256_file(MANIFEST_PATH)},
        'split': split_meta,
        'config_file': args.config,
        'features': config['features'],
        'dtw': config.get('baselines', {}).get('dtw', {}),
        'dataset_sizes': {'samples': counts, 'groups': group_counts},
        'command_used': ' '.join(sys.argv),
        'protocol': {
            'train': 'split=train 的记录',
            'val': 'split=val 的记录，仅用于超参选择',
            'test': 'split=test 的记录，固定，只评估一次',
            'split_unit': 'expression_group_id（同一表达的所有声学变体同属一个子集）',
        },
    }

    # 合并同一 run 目录下已有的实验结果（分段运行时保留先跑的实验）
    aggregated = dict(prior.get('experiments', {}) or {})
    t_start = time.perf_counter()

    def _flush():
        """每完成一个实验即落盘，避免长跑中断丢失全部结果。"""
        payload = {**common, 'feature_extraction_seconds': cache.extract_seconds,
                   'total_elapsed_sec': time.perf_counter() - t_start,
                   'experiments': aggregated,
                   'partial': True}
        p = out_dir / 'results.json'
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return p

    try:
        _run_all(args, records, group2subset, cache, config, aggregated, _flush)
    finally:
        result = {**common, 'feature_extraction_seconds': cache.extract_seconds,
                  'total_elapsed_sec': time.perf_counter() - t_start,
                  'experiments': aggregated, 'partial': False}
        path = out_dir / 'results.json'
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('结果:', path)


def _run_all(args, records, group2subset, cache, config, aggregated, flush):
    if 'E1' in args.experiments:
        print('[E1] 基线 ...', flush=True)
        t0 = time.perf_counter()
        r = experiment_E1(records, group2subset, cache, config, 42)
        r['elapsed_sec'] = time.perf_counter() - t0
        aggregated['E1_baseline'] = r
        print('   test acc=', r.get('test_metrics', {}).get('accuracy'), flush=True)
        flush()
    if 'E2' in args.experiments:
        print('[E2] 数据规模 ...', flush=True)
        t0 = time.perf_counter()
        r = experiment_E2(records, group2subset, cache, config, 42)
        r['elapsed_sec'] = time.perf_counter() - t0
        aggregated['E2_data_scale'] = r
        for c in r['curve']:
            print(f"   frac={c['fraction']} n={c['train_size']} "
                  f"test_acc={c['test_accuracy']:.4f}", flush=True)
        flush()
    if 'E3' in args.experiments:
        print('[E3] 模板数量 ...', flush=True)
        t0 = time.perf_counter()
        r = experiment_E3(records, group2subset, cache, config, 42)
        r['elapsed_sec'] = time.perf_counter() - t0
        aggregated['E3_template_count'] = r
        for c in r['results']:
            print(f"   k={c['k_per_class']} test_acc={c['test_accuracy']:.4f}", flush=True)
        flush()
    if 'E4' in args.experiments:
        print('[E4] 模板选择 + 多种子 + 自助法 ...', flush=True)
        t0 = time.perf_counter()
        r = experiment_E4(records, group2subset, cache, config, 42, seeds=args.seeds)
        r['elapsed_sec'] = time.perf_counter() - t0
        aggregated['E4_template_selection'] = r
        for s, v in r['strategies'].items():
            print(f"   {s}: acc={v['accuracy_mean']:.4f}±{v['accuracy_std']:.4f} "
                  f"f1={v['macro_f1_mean']:.4f}±{v['macro_f1_std']:.4f}", flush=True)
        flush()
    if 'B' in args.experiments:
        print('[B] 轨道B 留一音色 ...', flush=True)
        t0 = time.perf_counter()
        r = experiment_trackB(records, cache, config, 42)
        r['elapsed_sec'] = time.perf_counter() - t0
        aggregated['B_unseen_voice'] = r
        for v, s in r['voices'].items():
            print(f"   {v}: acc={s['test_accuracy']:.4f} n_test={s['test_samples']}", flush=True)
        flush()
    if 'N' in args.experiments:
        print('[N] 距离归一化对照 ...', flush=True)
        r = experiment_normalize(records, group2subset, cache, config, 42)
        aggregated['N_normalize'] = r
        print('  ', {k: round(v['test_accuracy'], 4) for k, v in r.items()
                     if isinstance(v, dict)}, flush=True)
        flush()
    if 'F' in args.experiments:
        print('[F] max_frames 扫描（仅验证集选择）...', flush=True)
        t0 = time.perf_counter()
        r = experiment_feature_distance(records, group2subset, cache, config, 42)
        r['elapsed_sec'] = time.perf_counter() - t0
        aggregated['F_max_frames'] = r
        for c in r['max_frames_sweep']:
            print(f"   mf={c['max_frames']} val={c['val_accuracy']:.4f} "
                  f"test={c['test_accuracy']:.4f}", flush=True)
        flush()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""第三阶段声学前端 / 嵌入 / 分类器消融实验。

严格协议
--------
1. 只使用固定划分 ``data/splits/track_a_expression_split.csv``（track A：未见表达）。
2. **所有超参数只按 val 选择**；test 对每个候选只预测一次（选好后一次性评估），
   不参与任何选择过程。
3. 结果写入 ``outputs/v3_acoustic/ablation.json``，包含每个候选的特征集/嵌入/
   分类器/超参/val 指标/test 指标（Acc/Macro-F1/每类 F1/混淆矩阵）/参数规模/延迟，
   以及 LIGHT、CURTAIN 两组方向对二分类准确率。
4. 最优候选按 val Acc > val Macro-F1 > 更少参数（更简单）排序；同时输出推荐配置
   并保存最优模型 ``outputs/v3_acoustic/best_model.npz``（供最终系统加载）。

用法
----
    python scripts/run_v3_acoustic_ablation.py            # 完整消融
    python scripts/run_v3_acoustic_ablation.py --quick    # 轻量子集（快速筛）
    python scripts/run_v3_acoustic_ablation.py --no-cache # 忽略特征缓存

依赖：numpy（主环境 D:\\anaconda3\\python.exe 已装）；不依赖 sklearn。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_io import load_wav  # noqa: E402
from src.features_v3 import (  # noqa: E402
    FeatureSpec, extract_sequence, fixed_length_embedding, stats_embedding,
)
from src.metrics import confusion_matrix, macro_f1  # noqa: E402
from src.v3_classifier import (  # noqa: E402
    MODEL_FORMAT_VERSION, KNNClassifier, LinearSVM,
    MultinomialLogisticRegression,
)

INTENTS = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE',
           'ALL_OFF', 'STATUS_ALL']
PAIRS = [('LIGHT_ON', 'LIGHT_OFF'), ('CURTAIN_OPEN', 'CURTAIN_CLOSE')]

MANIFEST = ROOT / 'data' / 'tts_corpus' / 'manifest_yue_v2.csv'
SPLIT = ROOT / 'data' / 'splits' / 'track_a_expression_split.csv'
OUT_DIR = ROOT / 'outputs' / 'v3_acoustic'
CACHE_DIR = OUT_DIR / 'cache'
ABLATION_JSON = OUT_DIR / 'ablation.json'
BEST_MODEL = OUT_DIR / 'best_model.npz'

V2_BASELINE_TEST_ACC = 0.5495   # 二期 MFCC13+CMVN+DTW 全模板，path 归一化

# ---------------------------------------------------------------------------
# 特征集 / 嵌入 / 分类器定义
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    'mfcc': FeatureSpec(),
    'mfcc_delta': FeatureSpec(include_delta=True, include_delta2=True),
    'mfcc_delta_pitch': FeatureSpec(include_delta=True, include_delta2=True,
                                    include_pitch=True),
    'logmel_delta': FeatureSpec(n_mels=40, include_mfcc=False, include_logmel=True,
                                include_delta=True, include_delta2=True),
    'mfcc_delta_pitch_energy': FeatureSpec(include_delta=True, include_delta2=True,
                                           include_pitch=True, include_energy=True),
    'logmel_delta_pitch': FeatureSpec(n_mels=40, include_mfcc=False, include_logmel=True,
                                      include_delta=True, include_delta2=True,
                                      include_pitch=True),
}
EMBEDDINGS = ('stats', 'fixed40', 'stats+fixed40')
FIXED_FRAMES = 40
CLASSIFIERS = ('logreg', 'linear_svm', 'knn')

# 超参数网格：全部只在 val 上选择
LOGREG_GRID = [{'l2': l2, 'standardize': std}
               for std in (True, False)
               for l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
SVM_GRID = [{'c': c, 'standardize': True} for c in (0.1, 1.0, 10.0)]
KNN_GRID = [{'k': k} for k in (1, 3, 5, 7, 9, 11)]

LOGREG_MAX_ITER = 300
SVM_EPOCHS = 80


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve_audio_path(audio_path) -> Path:
    """兼容两种清单写法：相对 data/tts_corpus 的路径，或相对项目根的 data/... 路径。"""
    p = Path(str(audio_path).replace('\\', '/'))
    if p.is_absolute():
        return p
    s = p.as_posix()
    if s.startswith('data/') or s.startswith('./data/'):
        return (ROOT / p).resolve()
    return ROOT / 'data' / 'tts_corpus' / p


def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_rows(manifest_path=MANIFEST, split_path=SPLIT):
    manifest_path = Path(manifest_path)
    split_path = Path(split_path)
    with open(split_path, encoding='utf-8-sig', newline='') as f:
        split = {r['expression_group_id']: r['subset'] for r in csv.DictReader(f)}
    with open(manifest_path, encoding='utf-8-sig', newline='') as f:
        manifest = list(csv.DictReader(f))
    rows = []
    for r in manifest:
        gid = r['expression_group_id']
        if gid not in split:
            raise KeyError(f'清单中的表达组不在固定划分里: {gid}')
        rows.append({
            'id': r['utterance_id'],
            'path': resolve_audio_path(r['audio_path']),
            'command': r['command'],
            'group': gid,
            'split': split[gid],
            'voice': r['voice_identity'],
            'condition': r['condition'],
        })
    return rows


def groups(rows):
    out = {'train': [], 'val': [], 'test': []}
    for r in rows:
        out[r['split']].append(r)
    return out


def command_labels(rows):
    """命令标签用字符串本身，保证与 predict()/混淆矩阵 labels 一致。"""
    return np.asarray([r['command'] for r in rows], dtype=object)


# ---------------------------------------------------------------------------
# 特征提取（带缓存）
# ---------------------------------------------------------------------------

def cache_path(feature_set: str, spec: FeatureSpec, manifest_sha: str) -> Path:
    return CACHE_DIR / f'{feature_set}_{spec.fingerprint()}_{manifest_sha[:10]}.npz'


def load_or_extract(rows, feature_set, spec, manifest_sha, use_cache=True):
    """返回 {utterance_id: float32 [T,D]} 与提取耗时（ms/条）。

    缓存文件名包含清单 sha 前缀（不同数据集互不覆盖）；同时兼容早期不带清单
    哈希的缓存文件名，命中前会校验文件内 manifest_sha。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(feature_set, spec, manifest_sha)
    legacy = CACHE_DIR / f'{feature_set}_{spec.fingerprint()}.npz'
    for cand in ([path, legacy] if legacy != path else [path]):
        if not (use_cache and cand.exists()):
            continue
        with np.load(cand, allow_pickle=True) as z:
            if str(z['manifest_sha']) != manifest_sha:
                continue
            ids = [str(i) for i in z['ids']]
            seqs = z['seqs']
            ms = float(z['extract_ms_per_sample'])
            if len(ids) == len(rows):
                return {i: seqs[k] for k, i in enumerate(ids)}, ms
    t0 = time.time()
    seqs = []
    for r in rows:
        audio, sr = load_wav(r['path'])
        seqs.append(extract_sequence(audio, sr, spec).copy())
    elapsed = time.time() - t0
    ms = 1000.0 * elapsed / max(len(rows), 1)
    ids = [r['id'] for r in rows]
    arr = np.empty(len(ids), dtype=object)
    for k, s in enumerate(seqs):
        arr[k] = s
    np.savez_compressed(path, ids=np.asarray(ids), seqs=arr,
                        manifest_sha=np.asarray(manifest_sha),
                        extract_ms_per_sample=np.asarray(ms),
                        feature_set=np.asarray(feature_set),
                        spec_json=np.asarray(json.dumps(spec.to_dict(), ensure_ascii=False)))
    return {i: seqs[k] for k, i in enumerate(ids)}, ms


# ---------------------------------------------------------------------------
# 嵌入
# ---------------------------------------------------------------------------

def embed_one(seq, embedding: str) -> np.ndarray:
    if embedding == 'stats':
        return stats_embedding(seq)
    if embedding == 'fixed40':
        return fixed_length_embedding(seq, n_frames=FIXED_FRAMES)
    if embedding == 'stats+fixed40':
        return np.concatenate([stats_embedding(seq),
                               fixed_length_embedding(seq, n_frames=FIXED_FRAMES)])
    raise KeyError(embedding)


def build_matrix(rows, feats, embedding):
    return np.asarray([embed_one(feats[r['id']], embedding) for r in rows],
                      dtype=np.float64)


# ---------------------------------------------------------------------------
# 指标与模型
# ---------------------------------------------------------------------------

def full_metrics(y_true, y_pred, labels):
    cm, _ = confusion_matrix(y_true, y_pred, labels=labels)
    cm = np.asarray(cm, dtype=int)
    per_class_f1 = {}
    for i, name in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[name] = round(float(f1), 4)
    return {
        'accuracy': round(float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), 4),
        'macro_f1': round(float(macro_f1(y_true, y_pred, labels=labels)), 4),
        'per_class_f1': per_class_f1,
        'n': int(len(y_true)),
        'confusion_matrix': cm.tolist(),
        'labels': list(labels),
    }


def make_estimator(classifier: str, hp: dict, seed: int = 0):
    if classifier == 'logreg':
        return MultinomialLogisticRegression(
            l2=hp['l2'], standardize=hp['standardize'],
            epochs=LOGREG_MAX_ITER, tol=1e-8, seed=seed)
    if classifier == 'linear_svm':
        return LinearSVM(c=hp['c'], standardize=hp['standardize'],
                         epochs=SVM_EPOCHS, seed=seed)
    if classifier == 'knn':
        return KNNClassifier(k=hp['k'], weights='uniform', seed=seed)
    raise KeyError(classifier)


def grid_for(classifier: str):
    return {'logreg': LOGREG_GRID, 'linear_svm': SVM_GRID,
            'knn': KNN_GRID}[classifier]


def selection_key(metrics):
    """val 选择键：Acc > Macro-F1 > n_params（越小越简单）。"""
    return (metrics['accuracy'], metrics['macro_f1'], -metrics.get('_n_params', 0))


def select_on_val(classifier, Xtr, ytr, Xva, yva, labels, feature_config):
    """在 val 上选超参；返回 (best_est, best_hp, best_val_metrics, grid_log)。"""
    grid_log = []
    best = None
    for hp in grid_for(classifier):
        est = make_estimator(classifier, hp)
        est.fit(Xtr, ytr, feature_config=feature_config)
        val_metrics = full_metrics(yva, est.predict(Xva), labels)
        val_metrics['_n_params'] = int(est.n_params)
        grid_log.append({'hyperparams': hp, 'val_accuracy': val_metrics['accuracy'],
                         'val_macro_f1': val_metrics['macro_f1'],
                         'n_params': val_metrics['_n_params']})
        key = selection_key(val_metrics)
        if best is None or key > best[0]:
            best = (key, est, hp, val_metrics)
    _, est, hp, val_metrics = best
    val_metrics = dict(val_metrics)
    val_metrics.pop('_n_params', None)
    return est, hp, val_metrics, grid_log


def measure_latency(est, Xva, repeats=5):
    t0 = time.perf_counter()
    for _ in range(repeats):
        est.predict(Xva)
    ms = 1000.0 * (time.perf_counter() - t0) / max(repeats * len(Xva), 1)
    return round(ms, 5)


def evaluate_binary_pair(rows_sub, feats, embedding, classifier, pair):
    """方向对二分类：同一特征/嵌入/分类器，val 选超参、test 只评一次。"""
    a, b = pair
    def pick(rs):
        rs = [r for r in rs if r['command'] in (a, b)]
        return rs
    tr, va, te = pick(rows_sub['train']), pick(rows_sub['val']), pick(rows_sub['test'])
    if min(len(tr), len(va), len(te)) == 0:
        return {'pair': f'{a}/{b}', 'error': 'empty split'}
    Xtr = build_matrix(tr, feats, embedding)
    Xva = build_matrix(va, feats, embedding)
    Xte = build_matrix(te, feats, embedding)
    ytr = np.asarray([a if r['command'] == a else b for r in tr], dtype=object)
    yva = np.asarray([a if r['command'] == a else b for r in va], dtype=object)
    yte = np.asarray([a if r['command'] == a else b for r in te], dtype=object)
    labels = [a, b]
    est, hp, val_metrics, _ = select_on_val(
        classifier, Xtr, ytr, Xva, yva, labels, feature_config=None)
    test_metrics = full_metrics(yte, est.predict(Xte), labels)
    n_a = int(np.sum(yte == a))
    majority_acc = round(max(n_a, len(yte) - n_a) / float(len(yte)), 4)
    return {
        'pair': f'{a}/{b}',
        'selected_hyperparams': hp,
        'val': val_metrics,
        'test': test_metrics,
        'n_val': int(len(va)),
        'n_test': int(len(te)),
        'majority_baseline_test_acc': majority_acc,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='v3 声学前端/分类器消融')
    ap.add_argument('--quick', action='store_true',
                    help='轻量子集快速筛（特征集/嵌入/分类器各取前若干项）')
    ap.add_argument('--no-cache', action='store_true', help='忽略特征缓存')
    ap.add_argument('--out', default=str(ABLATION_JSON), help='输出 JSON 路径')
    ap.add_argument('--manifest', default=str(MANIFEST), help='清单 CSV 路径')
    ap.add_argument('--split', default=str(SPLIT), help='固定划分 CSV 路径')
    ap.add_argument('--feature-sets', default=None,
                    help='逗号分隔的特征集子集，如 mfcc_delta,mfcc_delta_pitch')
    ap.add_argument('--embeddings', default=None,
                    help='逗号分隔的嵌入子集，如 stats,stats+fixed40')
    ap.add_argument('--classifiers', default=None,
                    help='逗号分隔的分类器子集，如 logreg')
    ap.add_argument('--no-pairs', action='store_true', help='跳过方向对二分类')
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    split_path = Path(args.split)
    out_path = Path(args.out)

    t_start = time.time()
    rows = load_rows(manifest_path, split_path)
    manifest_sha = sha256_file(manifest_path)
    sub = groups(rows)
    counts = {k: len(v) for k, v in sub.items()}
    print(f'[data] rows={len(rows)} {counts}', flush=True)
    if counts['train'] == 0 or counts['val'] == 0 or counts['test'] == 0:
        raise RuntimeError('固定划分缺少子集')

    feature_sets = list(FEATURE_SETS)
    if args.quick:
        feature_sets = ['mfcc_delta', 'logmel_delta']
    if args.feature_sets:
        feature_sets = [x for x in args.feature_sets.split(',') if x]
        unknown = [x for x in feature_sets if x not in FEATURE_SETS]
        if unknown:
            raise KeyError(f'未知特征集: {unknown}; 可选 {list(FEATURE_SETS)}')
    embeddings = (tuple(x for x in args.embeddings.split(',') if x)
                  if args.embeddings else EMBEDDINGS)
    classifiers = (tuple(x for x in args.classifiers.split(',') if x)
                   if args.classifiers else CLASSIFIERS)
    for e in embeddings:
        if e not in EMBEDDINGS:
            raise KeyError(f'未知嵌入: {e}; 可选 {EMBEDDINGS}')
    for c in classifiers:
        if c not in CLASSIFIERS:
            raise KeyError(f'未知分类器: {c}; 可选 {CLASSIFIERS}')
    print(f'[plan] manifest={manifest_path.name} split={split_path.name} '
          f'feature_sets={feature_sets} embeddings={embeddings} '
          f'classifiers={classifiers} pairs={not args.no_pairs}', flush=True)

    # 1) 提取全部候选特征集（共同复用同一批音频）
    feats, extract_ms = {}, {}
    for name in feature_sets:
        spec = FEATURE_SETS[name]
        t0 = time.time()
        feats[name], extract_ms[name] = load_or_extract(
            rows, name, spec, manifest_sha, use_cache=not args.no_cache)
        print(f'[features] {name}: D={output_dim_cached(feats[name])} '
              f'extract={extract_ms[name]:.2f}ms/条 (elapsed {time.time()-t0:.1f}s)',
              flush=True)

    # 2) 逐候选：val 选超参 -> test 只评一次
    candidates = []
    y_sub = {k: command_labels(v) for k, v in sub.items()}
    labels = INTENTS

    def dump_partial(status='partial'):
        """每完成若干候选即落盘，避免长任务被中断时丢掉已完成结果。"""
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                'meta': {'script': 'scripts/run_v3_acoustic_ablation.py',
                         'status': status,
                         'manifest': _rel(manifest_path),
                         'manifest_sha256': manifest_sha,
                         'split': _rel(split_path),
                         'split_sha256': sha256_file(split_path),
                         'data_counts': counts,
                         'selection': 'hyperparameters selected on val only; '
                                      'test predicted once per candidate',
                         'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')},
                'candidates': candidates,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as exc:  # pragma: no cover
            print(f'[warn] 部分结果落盘失败: {exc}', flush=True)
    for fs_name in feature_sets:
        for emb in embeddings:
            t0 = time.time()
            Xtr = build_matrix(sub['train'], feats[fs_name], emb)
            Xva = build_matrix(sub['val'], feats[fs_name], emb)
            Xte = build_matrix(sub['test'], feats[fs_name], emb)
            embed_ms = 1000.0 * (time.time() - t0) / (
                len(sub['train']) + len(sub['val']) + len(sub['test']))
            for clf in classifiers:
                feature_config = {
                    'feature_set': fs_name, 'spec': FEATURE_SETS[fs_name].to_dict(),
                    'embedding': emb, 'fixed_frames': FIXED_FRAMES,
                }
                t_sel = time.time()
                est, hp, val_metrics, grid_log = select_on_val(
                    clf, Xtr, y_sub['train'], Xva, y_sub['val'], labels, feature_config)
                sel_seconds = time.time() - t_sel
                # 关键纪律：test 仅在此处预测一次
                test_metrics = full_metrics(y_sub['test'], est.predict(Xte), labels)
                latency = measure_latency(est, Xva)
                rec = {
                    'feature_set': fs_name,
                    'embedding': emb,
                    'classifier': clf,
                    'selected_hyperparams': hp,
                    'val': val_metrics,
                    'test': test_metrics,
                    'n_features': int(Xtr.shape[1]),
                    'n_params': int(est.n_params),
                    'n_train': int(len(sub['train'])),
                    'n_val': int(len(sub['val'])),
                    'n_test': int(len(sub['test'])),
                    'feature_extract_ms_per_sample': round(extract_ms[fs_name], 3),
                    'embedding_ms_per_sample': round(embed_ms, 4),
                    'inference_ms_per_sample': latency,
                    'val_selection_seconds': round(sel_seconds, 2),
                    'val_grid': grid_log,
                }
                candidates.append(rec)
                if len(candidates) % 3 == 0:
                    dump_partial()
                print(f"[cand] {fs_name:24s} {emb:14s} {clf:10s} "
                      f"val={val_metrics['accuracy']:.4f}/f1={val_metrics['macro_f1']:.4f} "
                      f"test={test_metrics['accuracy']:.4f}/f1={test_metrics['macro_f1']:.4f} "
                      f"hp={hp} params={rec['n_params']} ({time.time()-t0:.1f}s)",
                      flush=True)

    # 3) 方向对二分类（对全部 logreg 候选：val 选超参，test 一次；
    #    logreg 是当前最强族；若最优候选不是 logreg，最后再对其补算）
    print('[pairs] 方向对二分类（logreg 候选）...', flush=True)
    for rec in candidates:
        if args.no_pairs or rec['classifier'] != 'logreg':
            rec['directional'] = {}
            continue
        fs_name, emb = rec['feature_set'], rec['embedding']
        rec['directional'] = {}
        for pair in PAIRS:
            res = evaluate_binary_pair(sub, feats[fs_name], emb, 'logreg', pair)
            rec['directional'][res['pair']] = res
        print(f"[pair] {fs_name:24s} {emb:14s} logreg     "
              + ' | '.join(
                  f"{p}: val={r['val']['accuracy']:.3f} test={r['test']['accuracy']:.3f}"
                  for p, r in rec['directional'].items() if 'test' in r),
              flush=True)

    # 4) 汇总：按 val 选择最优（不使用 test）
    def sort_key(rec):
        return (rec['val']['accuracy'], rec['val']['macro_f1'],
                -rec['n_params'], rec['feature_set'], rec['embedding'], rec['classifier'])

    ranked = sorted(candidates, key=sort_key, reverse=True)
    best = ranked[0]
    if not args.no_pairs and not best.get('directional'):
        best['directional'] = {}
        for pair in PAIRS:
            res = evaluate_binary_pair(sub, feats[best['feature_set']],
                                       best['embedding'], best['classifier'], pair)
            best['directional'][res['pair']] = res
    print(f"[best] {best['feature_set']} | {best['embedding']} | {best['classifier']} "
          f"hp={best['selected_hyperparams']} val={best['val']['accuracy']:.4f} "
          f"test={best['test']['accuracy']:.4f}", flush=True)

    # 5) 保存最优模型（train 训练，val 选超参；test 未参与训练）
    best_model_path = None
    try:
        fs_name, emb = best['feature_set'], best['embedding']
        Xtr = build_matrix(sub['train'], feats[fs_name], emb)
        feature_config = {
            'feature_set': fs_name, 'spec': FEATURE_SETS[fs_name].to_dict(),
            'embedding': emb, 'fixed_frames': FIXED_FRAMES,
        }
        est = make_estimator(best['classifier'], best['selected_hyperparams'])
        est.fit(Xtr, y_sub['train'], feature_config=feature_config)
        best_model_path = est.save(OUT_DIR / 'best_model')
    except Exception as exc:  # pragma: no cover - 保存失败不应毁掉实验记录
        print(f'[warn] 最优模型保存失败: {exc}', flush=True)

    # 6) 输出 JSON
    d = sorted((r for r in candidates if r['feature_set'] == best['feature_set']
                and r['embedding'] == best['embedding']),
               key=lambda r: r['val']['accuracy'], reverse=True)
    by_classifier = {r['classifier']: r['val']['accuracy'] for r in d}
    payload = {
        'meta': {
            'script': 'scripts/run_v3_acoustic_ablation.py',
            'model_format_version': MODEL_FORMAT_VERSION,
            'quick': bool(args.quick),
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'platform': platform.platform(),
            'elapsed_seconds': round(time.time() - t_start, 1),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'protocol': {
                'split': _rel(split_path),
                'split_sha256': sha256_file(split_path),
                'manifest': _rel(manifest_path),
                'manifest_sha256': manifest_sha,
                'selection': 'hyperparameters selected on val only; test predicted once per candidate',
                'direction_pairs': 'LIGHT_ON/OFF 与 CURTAIN_OPEN/CLOSE 二分类：'
                                   '对全部 logreg 候选分别用 val 选超参、test 只评一次；'
                                   '若最优候选不是 logreg 再对其补算',
                'ranking': 'val accuracy, then val macro-F1, then fewer parameters',
                'intents': INTENTS,
            },
            'data_counts': counts,
            'v2_baseline_test_accuracy': (
                V2_BASELINE_TEST_ACC if manifest_path.name == MANIFEST.name else None),
            'notes': 'v3.1 数据上的结果用于与 CNN 主模型同划分对比；'
                     'test 未参与任何超参选择。',
        },
        'feature_sets': {
            name: {'spec': spec.to_dict(),
                   'sequence_dim': int(next(iter(feats[name].values())).shape[1]),
                   'extract_ms_per_sample': round(float(extract_ms[name]), 3)}
            for name, spec in FEATURE_SETS.items() if name in feats
        },
        'embeddings': {
            'used': list(embeddings),
            'stats': ['mean', 'std', 'min', 'max', 'p10', 'p90'],
            'fixed40': FIXED_FRAMES,
        },
        'search_space': {
            'classifiers': list(classifiers),
            'logreg': LOGREG_GRID,
            'linear_svm': SVM_GRID,
            'knn': KNN_GRID,
            'logreg_max_iter': LOGREG_MAX_ITER,
            'svm_epochs': SVM_EPOCHS,
        },
        'candidates': candidates,
        'ranking': [
            {'rank': i + 1, 'feature_set': r['feature_set'],
             'embedding': r['embedding'], 'classifier': r['classifier'],
             'selected_hyperparams': r['selected_hyperparams'],
             'val_accuracy': r['val']['accuracy'],
             'val_macro_f1': r['val']['macro_f1'],
             'test_accuracy': r['test']['accuracy'],
             'test_macro_f1': r['test']['macro_f1'],
             'n_params': r['n_params']}
            for i, r in enumerate(ranked)
        ],
        'best_overall': {k: best[k] for k in (
            'feature_set', 'embedding', 'classifier', 'selected_hyperparams',
            'val', 'test', 'n_features', 'n_params', 'inference_ms_per_sample',
            'embedding_ms_per_sample', 'feature_extract_ms_per_sample', 'directional')},
        'best_model_file': best_model_path,
        'same_feature_val_by_classifier': by_classifier,
        'limitations': [
            '语料为 600 条合成 TTS（3 个 edge 粤语音色 + 1 个 canto-tts 遗留音色），'
            '不是真人录音；结论只适用于合成语音。',
            '固定划分 track A 为「未见表达、已见音色」；不能外推到未见音色/真实麦克风。',
            'test 仅 111 条（31 个表达组），准确率标准误约 ±4.5 个百分点，候选间小差异不显著。',
            'test 未参与任何超参选择，但多次查看 test 结果用于报告会让读者产生选择偏差；'
            '本实验每个候选只评估一次。',
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] 写出 {out_path}，候选 {len(candidates)} 个，'
          f'总耗时 {time.time()-t_start:.1f}s', flush=True)


def output_dim_cached(feats) -> int:
    return int(next(iter(feats.values())).shape[1])


if __name__ == '__main__':
    main()

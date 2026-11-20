# -*- coding: utf-8 -*-
"""TTS语义配对上的句子级OLS/Ridge映射对照。不是逐帧映射，也不是真人实验。"""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.representations import FeatureExtractor, sentence_vector
from src.classifiers import StandardScalerNumpy, SoftmaxRegression
from src.mapping import OLSMapping, RidgeMapping, mapping_diagnostics
from src.metrics import classification_metrics

base = Path(r'D:\Run_Tao\DialectVoice_Research')
cfg = load_config(base / 'config.json')
ex = FeatureExtractor(cfg['features'])
records = load_manifest(base / 'data' / 'tts_corpus' / 'manifest.csv')
canto = {r.semantic_pair_id: r for r in records if r.source == 'canto_tts'}
mandarin = {r.semantic_pair_id: r for r in records if r.source == 'edge_tts_mandarin' and r.speaker_id == 'zh-CN-XiaoxiaoNeural'}
pairs = []
for pid, cr in canto.items():
    if pid in mandarin:
        mr = mandarin[pid]
        pairs.append((cr, mr))
train_exprs = {'direct', 'colloquial', 'polite'}
val_exprs = {'question'}
test_exprs = {'correction'}
train_pairs = [p for p in pairs if p[0].expression_type in train_exprs]
val_pairs = [p for p in pairs if p[0].expression_type in val_exprs]
test_pairs = [p for p in pairs if p[0].expression_type in test_exprs]
print('pairs:', len(pairs), len(train_pairs), len(val_pairs), len(test_pairs))

def feat(r):
    return sentence_vector(ex.sequence_from_record(r)).astype(np.float64)

Xtr = np.stack([feat(c) for c, m in train_pairs]); Ytr = np.stack([feat(m) for c, m in train_pairs])
Xte = np.stack([feat(c) for c, m in test_pairs]); yte = np.array([c.command for c, m in test_pairs])
Xval = np.stack([feat(c) for c, m in val_pairs]); yval = np.array([c.command for c, m in val_pairs])
Ytgt = Ytr
classes = sorted(set(yte))
train_labels = np.array([c.command for c, m in train_pairs])
centroids = np.stack([Ytgt[train_labels == cls].mean(axis=0) for cls in classes])

# 基线：canto特征直接分类
scaler = StandardScalerNumpy().fit(Xtr)
clf = SoftmaxRegression(l2=0.001, learning_rate=0.1, epochs=1000).fit(scaler.transform(Xtr), np.array([c.command for c, m in train_pairs]))
base_pred = clf.predict(scaler.transform(Xte))
base_metrics = classification_metrics(yte, base_pred)

def map_predict(model):
    Z = np.stack([model.transform(Xte[k].reshape(1, -1)).reshape(-1) for k in range(len(Xte))])
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    C = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    pred_idx = np.argmax(Z @ C.T, axis=1)
    return np.array([classes[i] for i in pred_idx])

res = {
    'note': 'synthetic TTS semantic pairs; sentence-level mapping; not frame-aligned; not real dialect performance',
    'num_pairs': len(pairs),
    'split': {'train_expressions': sorted(train_exprs), 'val_expressions': sorted(val_exprs), 'test_expressions': sorted(test_exprs),
              'train': len(train_pairs), 'val': len(val_pairs), 'test': len(test_pairs)},
    'baseline_direct_classifier': base_metrics,
    'mapping': {}
}
for name, model in [('ols', OLSMapping(fit_intercept=True)), ('ridge', RidgeMapping(alpha=0.001, fit_intercept=True))]:
    try:
        model.fit(Xtr.reshape(len(Xtr), 1, -1), Ytr.reshape(len(Ytr), 1, -1))
        diag = mapping_diagnostics(model, Xtr.reshape(len(Xtr), 1, -1), Ytr.reshape(len(Ytr), 1, -1))
        pred = map_predict(model)
        metrics = classification_metrics(yte, pred)
        metrics['mapping_train_diagnostics'] = diag
        res['mapping'][name] = metrics
    except Exception as exc:
        res['mapping'][name] = {'error': repr(exc)}

out = base / 'outputs' / 'tts_mapping_experiment.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=2))
print('saved:', out)

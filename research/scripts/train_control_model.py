# -*- coding: utf-8 -*-
"""在TTS粤语数据上训练六类控制模型，并校准拒识阈值。"""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.representations import FeatureExtractor, sentence_vector
from src.classifiers import StandardScalerNumpy, SoftmaxRegression, LinearSVM
from src.rejection import calibrate_threshold, rejection_metrics
from src.metrics import classification_metrics
from src.model_store import save_bundle

base = Path(r'D:\Run_Tao\DialectVoice_Research')
cfg = load_config(base / 'config.json')
records = load_manifest(base / 'data' / 'tts_corpus' / 'manifest.csv')
records = [r for r in records if r.language == 'yue']
train_exprs = {'direct', 'colloquial', 'polite'}
val_exprs = {'question'}
test_exprs = {'correction'}
train = [r for r in records if r.expression_type in train_exprs]
val = [r for r in records if r.expression_type in val_exprs]
test = [r for r in records if r.expression_type in test_exprs]
print('sizes:', len(train), len(val), len(test))

ex = FeatureExtractor(cfg['features'])
def xy(rs):
    X = np.stack([sentence_vector(ex.sequence_from_record(r)) for r in rs])
    y = np.array([r.command for r in rs])
    return X, y
Xtr, ytr = xy(train)
Xval, yval = xy(val)
Xte, yte = xy(test)
scaler = StandardScalerNumpy().fit(Xtr)
Xtr_s, Xval_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xte)

candidates = {
    'softmax': SoftmaxRegression(l2=0.001, learning_rate=0.1, epochs=1000),
    'linear_svm': LinearSVM(l2=0.001, learning_rate=0.05, epochs=800),
}
best = None
for name, model in candidates.items():
    model.fit(Xtr_s, ytr)
    val_pred = model.predict(Xval_s)
    val_acc = float(np.mean(val_pred == yval))
    print(name, 'val_acc=', val_acc)
    if best is None or val_acc > best[0]:
        best = (val_acc, name, model)
val_acc, best_name, model = best
val_proba = model.predict_proba(Xval_s)
val_conf = np.max(val_proba, axis=1)
val_pred = model.predict(Xval_s)
cal = calibrate_threshold(val_conf, val_pred == yval, min_coverage=0.8)
threshold = cal['threshold']

test_proba = model.predict_proba(Xte_s)
test_conf = np.max(test_proba, axis=1)
test_pred = model.predict(Xte_s)
test_acc = float(np.mean(test_pred == yte))
accepted = test_conf >= threshold
accepted_acc = float(np.mean((test_pred == yte)[accepted])) if accepted.any() else 0.0
metrics = classification_metrics(yte, test_pred)
metrics.update({
    'accepted_coverage': float(np.mean(accepted)),
    'accepted_accuracy': accepted_acc,
    'threshold': float(threshold),
    'val_threshold_calibration': cal,
    'chosen_model': best_name,
    'val_accuracy': val_acc,
    'train_size': len(train), 'val_size': len(val), 'test_size': len(test),
    'train_expressions': sorted(train_exprs),
    'val_expressions': sorted(val_exprs),
    'test_expressions': sorted(test_exprs),
})

bundle_path = base / 'data' / 'models' / 'tts_control_model.pkl'
save_bundle(bundle_path, model, scaler, threshold, metadata={
    'chosen_model': best_name, 'metrics': metrics,
    'note': 'synthetic TTS only; no real dialect performance'
})
out = base / 'outputs' / 'control_model_metrics.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print('model:', bundle_path)

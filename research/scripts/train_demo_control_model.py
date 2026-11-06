# -*- coding: utf-8 -*-
"""训练仅用于端到端演示的控制模型。注意：训练和演示音频同分布，不能当评估结果。"""
from pathlib import Path
import json
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.representations import FeatureExtractor, sentence_vector
from src.classifiers import StandardScalerNumpy, SoftmaxRegression
from src.rejection import calibrate_threshold
from src.model_store import save_bundle

base = Path(r'D:\Run_Tao\DialectVoice_Research')
cfg = load_config(base / 'config.json')
records = [r for r in load_manifest(base / 'data' / 'tts_corpus' / 'manifest.csv') if r.language == 'yue']
ex = FeatureExtractor(cfg['features'])
X = np.stack([sentence_vector(ex.sequence_from_record(r)) for r in records])
y = np.array([r.command for r in records])
scaler = StandardScalerNumpy().fit(X)
model = SoftmaxRegression(l2=0.001, learning_rate=0.1, epochs=1500)
model.fit(scaler.transform(X), y)
proba = model.predict_proba(scaler.transform(X))
conf = np.max(proba, axis=1)
pred = model.predict(scaler.transform(X))
cal = calibrate_threshold(conf, pred == y, min_coverage=0.9)
path = base / 'data' / 'models' / 'tts_control_model_demo.pkl'
conservative_threshold = max(cal['threshold'], 0.9)
save_bundle(path, model, scaler, conservative_threshold, metadata={
    'demo_only': True,
    'warning': 'trained on all synthetic TTS yue samples; not an evaluation model',
    'train_accuracy': float(np.mean(pred == y)),
    'threshold_calibration': cal,
    'conservative_threshold': conservative_threshold,
})
out = base / 'outputs' / 'demo_control_model_info.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    'demo_only': True,
    'train_accuracy': float(np.mean(pred == y)),
    'threshold': conservative_threshold,
    'calibration': cal,
    'num_samples': len(records),
}, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved', path)
print('train acc', np.mean(pred == y), 'threshold', cal['threshold'])

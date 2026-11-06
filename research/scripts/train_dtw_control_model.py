# -*- coding: utf-8 -*-
"""训练DTW控制模型，校准拒识阈值，并保存评估/演示模型。"""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.representations import FeatureExtractor
from src.dtw_model import DTWControlModel
from src.rejection import calibrate_threshold_with_negatives, rejection_metrics
from src.metrics import classification_metrics

base = Path(r'D:\Run_Tao\DialectVoice_Research')
cfg = load_config(base / 'config.json')
yue = [r for r in load_manifest(base / 'data' / 'tts_corpus' / 'manifest.csv') if r.language == 'yue']
train = [r for r in yue if r.expression_type in {'direct', 'colloquial', 'polite'}]
val = [r for r in yue if r.expression_type == 'question']
test = [r for r in yue if r.expression_type == 'correction']
unknown = load_manifest(base / 'data' / 'tts_unknown' / 'manifest.csv')
ex = FeatureExtractor(cfg['features'])
max_frames = int(cfg['baselines']['dtw'].get('max_frames', 60))

model = DTWControlModel.fit(train, ex, threshold=0.7, max_frames=max_frames)
val_pred = [model.predict_sequence(ex.sequence_from_record(r)) for r in val]
val_y = np.array([r.command for r in val])
val_p = np.array([p.command for p in val_pred])
val_conf = np.array([p.confidence for p in val_pred])
# 未知校准/测试按文本分组，避免同一文本同时进校准和测试
texts = sorted({r.text for r in unknown})
cal_texts = set(texts[:len(texts)//2])
unknown_cal = [r for r in unknown if r.text in cal_texts]
unknown_test = [r for r in unknown if r.text not in cal_texts]
cal_conf = np.array([model.predict_sequence(ex.sequence_from_record(r)).confidence for r in unknown_cal])
cal = calibrate_threshold_with_negatives(val_conf, val_p == val_y, cal_conf, target_far=0.1, min_coverage=0.2)
model.threshold = cal['threshold']

test_pred = [model.predict_sequence(ex.sequence_from_record(r)) for r in test]
test_y = np.array([r.command for r in test])
test_p = np.array([p.command for p in test_pred])
test_conf = np.array([p.confidence for p in test_pred])
test_metrics = classification_metrics(test_y, test_p)
unknown_conf = np.array([model.predict_sequence(ex.sequence_from_record(r)).confidence for r in unknown_test])
rej = rejection_metrics(test_conf, test_p == test_y, unknown_conf, model.threshold)
metrics = {
    'model': 'dtw',
    'max_frames': max_frames,
    'threshold': model.threshold,
    'calibration': cal,
    'val_accuracy': float(np.mean(val_p == val_y)),
    'test_accuracy': test_metrics['accuracy'],
    'test_macro_f1': test_metrics['macro_f1'],
    'test_confusion_matrix': test_metrics['confusion_matrix'],
    'rejection': rej,
    'train_size': len(train), 'val_size': len(val), 'test_size': len(test),
    'unknown_cal_size': len(unknown_cal), 'unknown_test_size': len(unknown_test),
    'note': 'synthetic TTS only; expression holdout; not real dialect performance'
}
out = base / 'outputs' / 'dtw_control_metrics.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
model.save(base / 'data' / 'models' / 'tts_control_model_dtw.pkl')
# 演示模型：全量yue数据，高保守阈值0.9；仅用于端到端演示
demo = DTWControlModel.fit(yue, ex, threshold=0.9, max_frames=max_frames)
demo.save(base / 'data' / 'models' / 'tts_control_model_dtw_demo.pkl')
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print('saved models')

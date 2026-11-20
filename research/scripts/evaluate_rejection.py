# -*- coding: utf-8 -*-
"""评估已知命令覆盖率/接受准确率与未知命令拒识率。"""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.representations import FeatureExtractor, sentence_vector
from src.model_store import load_bundle
from src.rejection import rejection_metrics

base = Path(r'D:\Run_Tao\DialectVoice_Research')
cfg = load_config(base / 'config.json')
bundle = load_bundle(base / 'data' / 'models' / 'tts_control_model.pkl')
model, scaler, threshold = bundle['model'], bundle['scaler'], bundle['threshold']
ex = FeatureExtractor(cfg['features'])

def scores(records):
    conf, correct = [], []
    for r in records:
        x = scaler.transform(sentence_vector(ex.sequence_from_record(r)).reshape(1, -1))
        p = model.predict_proba(x)[0]
        conf.append(float(np.max(p)))
        correct.append(str(model.predict(x)[0]) == r.command)
    return np.array(conf), np.array(correct, dtype=bool)

yue = [r for r in load_manifest(base / 'data' / 'tts_corpus' / 'manifest.csv') if r.language == 'yue']
known = [r for r in yue if r.expression_type == 'correction']
unknown = load_manifest(base / 'data' / 'tts_unknown' / 'manifest.csv')
known_conf, known_correct = scores(known)
unknown_conf, _ = scores(unknown)
metrics = rejection_metrics(known_conf, known_correct, unknown_conf, threshold)
metrics['conservative_threshold_0_9'] = rejection_metrics(known_conf, known_correct, unknown_conf, 0.9)
metrics.update({
    'known_count': len(known), 'unknown_count': len(unknown),
    'known_accuracy_all': float(np.mean(known_correct)),
    'note': 'synthetic TTS only; not real dialect performance'
})
out = base / 'outputs' / 'rejection_metrics.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print('saved:', out)

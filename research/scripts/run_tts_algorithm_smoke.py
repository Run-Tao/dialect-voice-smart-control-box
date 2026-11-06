# -*- coding: utf-8 -*-
"""用canto-tts的12条音频做表达模板留出冒烟，不是真实粤语性能。"""
import csv
import json
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import ManifestRecord
from src.representations import FeatureExtractor, sentence_vector
from src.classifiers import StandardScalerNumpy, SoftmaxRegression
from src.baselines import predict_dtw, dtw_distance_normalized
from src.metrics import classification_metrics

base = Path(r'D:\Run_Tao\DialectVoice_Research')
cfg = load_config(base / 'config.json')
# 使用canto-tts的16k WAV；表达模板：direct训练，polite测试
log_path = base / 'outputs' / 'tts_canto_log.csv'
records = []
with open(log_path, 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        p = Path(row['audio_path'])
        p16 = base / 'data' / 'tts_validation' / 'canto_16k' / (p.stem + '.wav')
        records.append(ManifestRecord(
            utterance_id=row['utt_id'], audio_path=str(p16),
            speaker_id='canto_tts_default', dialect='yue-HK',
            command=row['intent_id'], language='yue',
            notes=row['expression_type']
        ))

def expr_of(r):
    return r.notes

def eval_split(train_expr, test_expr):
    train = [r for r in records if expr_of(r) == train_expr]
    test = [r for r in records if expr_of(r) == test_expr]
    ex = FeatureExtractor(cfg['features'])
    Xtr = np.stack([sentence_vector(ex.sequence_from_record(r)) for r in train])
    ytr = np.array([r.command for r in train])
    Xte = np.stack([sentence_vector(ex.sequence_from_record(r)) for r in test])
    yte = np.array([r.command for r in test])
    scaler = StandardScalerNumpy().fit(Xtr)
    model = SoftmaxRegression(l2=0.001, learning_rate=0.1, epochs=800)
    model.fit(scaler.transform(Xtr), ytr)
    pred = model.predict(scaler.transform(Xte))
    clf_metrics = classification_metrics(yte, pred)

    # DTW: 同一训练集构造每命令模板
    templates = {}
    for r in train:
        templates.setdefault(r.command, []).append(ex.sequence_from_record(r))
    dtw_pred = []
    for r in test:
        q = ex.sequence_from_record(r)
        p = predict_dtw(q, templates, normalize='path')
        dtw_pred.append(p.command)
    dtw_metrics = classification_metrics(yte, dtw_pred)
    return {
        'train_expression': train_expr,
        'test_expression': test_expr,
        'classifier_accuracy': clf_metrics['accuracy'],
        'classifier_macro_f1': clf_metrics['macro_f1'],
        'dtw_accuracy': dtw_metrics['accuracy'],
        'dtw_macro_f1': dtw_metrics['macro_f1'],
        'classifier_predictions': [{'true': str(t), 'pred': str(p)} for t, p in zip(yte, pred)],
    }

result = {
    'note': 'synthetic canto-tts only; expression-template holdout; not real dialect performance',
    'num_records': len(records),
    'experiments': [
        eval_split('direct', 'polite'),
        eval_split('polite', 'direct'),
    ]
}
out = base / 'outputs' / 'tts_algorithm_smoke.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
print('saved:', out)

# -*- coding: utf-8 -*-
"""对已训练 v3 CNN 工件做温度标定（只改置信度，不改 argmax）。

训练脚本内已有温度标定，但初版网格下界为 0.5，若最优点落在边界说明
需要更低温度。本脚本加载工件、用**验证集**重新计算原始 logits，
在更宽网格上最小化 NLL，并把新温度写回工件 meta（数组不变）。

用法::

    python scripts/calibrate_v3_temperature.py --model data/models/v3_cnn_ensemble.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.features_v3 import extract_sequence  # noqa: E402
from src.v3_model import V3Predictor  # noqa: E402


def resolve_audio(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for cand in (BASE / p, BASE / 'data' / 'tts_corpus' / p):
        if cand.exists():
            return cand
    return BASE / p


def load_val(cfg, model):
    """配置中的 val 划分（与训练脚本一致：split 文件为准）。"""
    with open(BASE / cfg['data']['split_file'], encoding='utf-8-sig', newline='') as f:
        group2subset = {r['expression_group_id']: r['subset'] for r in csv.DictReader(f)}
    ids = set(model.meta.get('val_utterance_ids', []))
    recs = []
    with open(BASE / cfg['data']['manifest'], encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['utterance_id'] not in ids:
                continue
            recs.append({'utterance_id': row['utterance_id'],
                         'audio_path': str(resolve_audio(row['audio_path'])),
                         'command': row['command']})
    return recs


def main():
    ap = argparse.ArgumentParser(description='v3 CNN 温度标定（仅验证集）')
    ap.add_argument('--model', default=str(BASE / 'data/models/v3_cnn_ensemble.npz'))
    ap.add_argument('--config', default=str(BASE / 'config_v3.json'))
    ap.add_argument('--grid-min', type=float, default=0.05)
    ap.add_argument('--grid-max', type=float, default=5.0)
    ap.add_argument('--grid-steps', type=int, default=100)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    model = V3Predictor.load(args.model)
    recs = load_val(cfg, model)
    if not recs:
        raise SystemExit('未找到验证集样本')
    classes = model.classes_
    y = np.asarray([classes.index(r['command']) for r in recs])
    logits = []
    for r in recs:
        seq = model.feature_sequence(r['audio_path'])
        from src.v3_cnn_numpy import prepare_input
        x = prepare_input(seq, model.n_frames)
        logits.append(model._raw_logits(x)[0])
    logits = np.stack(logits, 0)

    grid = np.linspace(args.grid_min, args.grid_max, args.grid_steps)
    best = (np.inf, None)
    curve = []
    for t in grid:
        z = logits / t
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))
        acc = float(np.mean(p.argmax(1) == y))
        curve.append({'temperature': float(t), 'val_nll': float(nll), 'val_accuracy': acc})
        if nll < best[0]:
            best = (float(nll), float(t))
    nll, temp = best
    old_temp = model.temperature
    model.temperature = temp
    # 重新写出工件：数组原样，只更新 meta
    path = Path(args.model)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in z.files if k != '__meta__'}
    meta = dict(model.meta)
    meta['temperature'] = float(temp)
    meta['temperature_calibration'] = {
        'grid': [args.grid_min, args.grid_max, args.grid_steps],
        'val_nll': nll,
        'previous_temperature': old_temp,
        'note': '仅验证集 NLL 最小化；argmax 不变；temperature=grid 边界时说明置信度仍偏保守/偏激进',
    }
    arrays['__meta__'] = np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    np.savez_compressed(path, **arrays)
    print(json.dumps({'temperature': temp, 'val_nll': nll,
                      'val_accuracy': curve[int(np.argmin([c['val_nll'] for c in curve]))]['val_accuracy'],
                      'previous_temperature': old_temp, 'model': str(path)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

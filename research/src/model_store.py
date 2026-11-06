# -*- coding: utf-8 -*-
"""模型bundle的保存和加载。"""
import pickle
from pathlib import Path


def save_bundle(path, model, scaler, threshold, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'threshold': float(threshold),
        'metadata': metadata or {},
    }
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    return path


def load_bundle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'模型文件不存在: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)

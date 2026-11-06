# -*- coding: utf-8 -*-
"""配置读取与校验。"""
import json
from pathlib import Path


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    required = ['features', 'split', 'alignment', 'outputs']
    for key in required:
        if key not in cfg:
            raise ValueError(f"配置缺少字段: {key}")

    feat = cfg['features']
    for key in ['sample_rate', 'n_fft', 'hop_length', 'n_mfcc', 'n_mels']:
        if key not in feat:
            raise ValueError(f"features缺少字段: {key}")
    if feat['n_fft'] <= 0 or feat['hop_length'] <= 0:
        raise ValueError("n_fft和hop_length必须为正数")
    if feat['n_mfcc'] > feat['n_mels']:
        raise ValueError("n_mfcc不能大于n_mels")

    sp = cfg['split']
    ratios = [float(sp['train_ratio']), float(sp['val_ratio']), float(sp['test_ratio'])]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split比例之和必须为1")
    return True

# -*- coding: utf-8 -*-
"""第三阶段 1D-CNN 的纯 numpy 推理实现（部署用，不依赖 torch）。

训练在 ``scripts/train_v3_cnn.py`` 中用 PyTorch 完成，权重导出为 numpy；
推理阶段只用 numpy 前向，避免部署环境安装 torch 及 OpenMP 冲突。

网络结构（与训练脚本严格一致）::

    Input [B, 80, 256]  (40 log-mel + 40 Δ log-mel, 256 帧)
    Conv1d(80->64, k=5, pad=2) -> BatchNorm -> ReLU -> MaxPool(2)
    Conv1d(64->128, k=5, pad=2) -> BatchNorm -> ReLU -> MaxPool(2)
    Conv1d(128->128, k=3, pad=1) -> BatchNorm -> ReLU
    Global mean + max pooling -> 256
    Dropout（仅训练）-> Linear(256 -> 6)

批量归一化使用训练期累计的 running_mean/running_var（eval 模式）。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

ARCH_NAME = 'cmd_cnn_v1'
CONV_LAYERS = [
    # (weight_key, in_ch, out_ch, kernel)
    ('conv0', 80, 64, 5),
    ('conv1', 64, 128, 5),
    ('conv2', 128, 128, 3),
]
N_FRAMES = 256
IN_DIM = 80
EMBED_DIM = 256
NUM_CLASSES = 6


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int) -> np.ndarray:
    """x [B,C,T], w [O,C,K] -> [B,O,T]。"""
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode='constant')
    B, C, T = x.shape
    K = w.shape[2]
    win = np.lib.stride_tricks.sliding_window_view(x, K, axis=2)  # [B,C,T,K]
    out = np.einsum('bctk,ock->bot', win, w, optimize=True)
    return out + b[None, :, None]


def _batch_norm(x: np.ndarray, gamma, beta, mean, var, eps: float = 1e-5) -> np.ndarray:
    return (x - mean[None, :, None]) / np.sqrt(var[None, :, None] + eps) * \
        gamma[None, :, None] + beta[None, :, None]


def _relu(x):
    return np.maximum(x, 0.0)


def _maxpool2(x):
    B, C, T = x.shape
    T2 = T // 2
    return x[:, :, :T2 * 2].reshape(B, C, T2, 2).max(axis=3)


def embed(params: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """输入 [B,80,256] -> 256 维嵌入（mean+max 池化后，未过 dropout/FC）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != IN_DIM:
        raise ValueError(f'输入必须为 [B,{IN_DIM},T]，当前 {x.shape}')
    for name, cin, cout, k in CONV_LAYERS:
        x = _conv1d(x, params[name + '.weight'], params[name + '.bias'], pad=k // 2)
        x = _batch_norm(x, params[name + '.bn.weight'], params[name + '.bn.bias'],
                        params[name + '.bn.running_mean'], params[name + '.bn.running_var'])
        x = _relu(x)
        if name in ('conv0', 'conv1'):
            x = _maxpool2(x)
    pooled = np.concatenate([x.mean(axis=2), x.max(axis=2)], axis=1)
    return pooled


def logits(params: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    e = embed(params, x)
    return e @ params['fc.weight'].T + params['fc.bias']


def softmax_np(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def resample_to_frames(seq: np.ndarray, n_frames: int) -> np.ndarray:
    """沿时间轴线性插值到 n_frames 帧（与二期 ``fixed_length_sequence`` 同定义）。"""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[0] < 1:
        raise ValueError('seq 必须是 [T,D] 且 T>=1')
    T = seq.shape[0]
    if T == n_frames:
        return seq
    if T == 1:
        return np.repeat(seq, n_frames, axis=0)
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, n_frames)
    out = np.empty((n_frames, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(dst, src, seq[:, d]).astype(np.float32)
    return out


def prepare_input(seq: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    """帧级特征 [T,D] -> 网络输入 [1,D,n_frames]。"""
    return resample_to_frames(seq, n_frames).T[None, :, :].astype(np.float32)


def export_manifest() -> Dict[str, object]:
    return {
        'arch': ARCH_NAME,
        'in_dim': IN_DIM,
        'n_frames': N_FRAMES,
        'embed_dim': EMBED_DIM,
        'num_classes': NUM_CLASSES,
        'conv_layers': [{'name': n, 'in': i, 'out': o, 'kernel': k} for n, i, o, k in CONV_LAYERS],
        'pooling': 'global_mean+global_max',
    }


def collect_arrays(state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """把训练脚本保存的参数整理成推理参数名。

    兼容两种输入：
    1. 已经是 ``conv0.weight`` 这类规范命名；
    2. PyTorch ``CmdCNN.state_dict()`` 的 ``net.0.weight / net.1.weight / fc.weight``
       这类顺序命名。
    """
    if 'conv0.weight' in state_dict:
        source = dict(state_dict)
    else:
        # net.Sequential 下标：0=conv,1=bn,2=relu,3=pool, 4=conv,5=bn,6=relu,7=pool,
        # 8=conv,9=bn,10=relu
        mapping = {}
        for idx, name in [(0, 'conv0'), (4, 'conv1'), (8, 'conv2')]:
            mapping[f'net.{idx}.weight'] = name + '.weight'
            mapping[f'net.{idx}.bias'] = name + '.bias'
        for idx, name in [(1, 'conv0'), (5, 'conv1'), (9, 'conv2')]:
            for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
                mapping[f'net.{idx}.{suffix}'] = name + '.bn.' + suffix
        mapping['fc.weight'] = 'fc.weight'
        mapping['fc.bias'] = 'fc.bias'
        source = {}
        for src, dst in mapping.items():
            if src in state_dict:
                source[dst] = state_dict[src]
    wanted: List[str] = []
    for name, _, _, _ in CONV_LAYERS:
        for suffix in ('.weight', '.bias', '.bn.weight', '.bn.bias',
                       '.bn.running_mean', '.bn.running_var'):
            wanted.append(name + suffix)
    wanted += ['fc.weight', 'fc.bias']
    out = {}
    for key in wanted:
        if key in source:
            out[key] = np.asarray(source[key], dtype=np.float64)
    missing = [k for k in wanted if k not in out]
    if missing:
        raise KeyError(f'缺少参数: {missing}')
    return out

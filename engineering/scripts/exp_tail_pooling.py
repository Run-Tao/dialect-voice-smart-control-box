# -*- coding: utf-8 -*-
"""验证"时序位置缺失"假设：把全局 mean+max 池化换成尾部加权池化，
看 test/val 准确率能否提升。卷积只前向一次，缓存 conv2 输出，
各种池化仅重算 FC，成本低。只读模型，不改工件。

假设：粤语家居命令的意图动词短语恒在句尾（"帮我打开窗簾"），
纠错句更是"唔係A，係B"结构，B 在句尾。全局池化丢失该位置信息。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.v3_model import V3Predictor
from runtime.v3_cnn_numpy import (_batch_norm, _conv1d, _maxpool2, _relu,
                                  CONV_LAYERS, prepare_input)
from runtime.v3_cnn_numpy import softmax_np

MANIFEST = ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'manifest_yue_v3_2.csv'


def conv_feat(params, x):
    """[1,80,256] -> conv2 relu 后 [1,128,64]（保持时序）。"""
    for name, cin, cout, k in CONV_LAYERS:
        x = _conv1d(x, params[name + '.weight'], params[name + '.bias'], k // 2)
        x = _batch_norm(x, params[name + '.bn.weight'], params[name + '.bn.bias'],
                        params[name + '.bn.running_mean'], params[name + '.bn.running_var'])
        x = _relu(x)
        if name in ('conv0', 'conv1'):
            x = _maxpool2(x)
    return x


def pooled(x, mode, alpha=1.0, frac=0.5):
    """x [1,C,T] -> [1,2C]"""
    C, T = x.shape[1], x.shape[2]
    g_mean, g_max = x.mean(2), x.max(2)
    if mode == 'global':
        return np.concatenate([g_mean, g_max], 1)
    t = np.arange(T, dtype=np.float64)
    if mode == 'tail_w':            # 指数尾部加权 mean + 全局 max
        w = np.exp(alpha * (t / (T - 1)))
        w = w / w.sum()
        wm = (x * w[None, None, :]).sum(2)
        return np.concatenate([wm, g_max], 1)
    if mode == 'tail_w_both':       # 加权 mean + 加权 max（用权重掩蔽后 max）
        w = np.exp(alpha * (t / (T - 1)))
        w = w / w.sum()
        wm = (x * w[None, None, :]).sum(2)
        wmax = (x * (w / w.max())[None, None, :]).max(2)
        return np.concatenate([wm, wmax], 1)
    if mode == 'tail_only':         # 只用尾部 frac 比例帧
        start = int(T * (1 - frac))
        xt = x[:, :, start:]
        return np.concatenate([xt.mean(2), xt.max(2)], 1)
    if mode == 'gmean_tailmax':     # 全局 mean + 尾部 max
        start = int(T * (1 - frac))
        return np.concatenate([g_mean, x[:, :, start:].max(2)], 1)
    raise ValueError(mode)


def main():
    m = V3Predictor.load(str(ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_v32b.npz'))
    C = list(m.classes_)

    for split in ('test', 'val'):
        rows = [r for r in csv.DictReader(open(MANIFEST, encoding='utf-8-sig'))
                if r['split'] == split]
        feats, y = [], []
        for r in rows:
            seq = m.feature_sequence(r['audio_path'])
            x = prepare_input(seq, m.n_frames)
            fs = [conv_feat(p, x)[0] for p in m._seed_params]   # 各种子 conv 输出
            feats.append(fs)
            y.append(r['command'])

        print(f'\n===== {split} ({len(y)} 条) =====')

        def evaluate(mode, **kw):
            preds = []
            for fs in feats:
                zs = []
                for p, f in zip(m._seed_params, fs):
                    e = pooled(f[None], mode, **kw)
                    zs.append(e @ p['fc.weight'].T + p['fc.bias'])
                # 注意：np.mean(keepdims=True) 会得到 [1,1,C]，必须去掉多余轴
                prob = softmax_np(np.mean(np.stack(zs, 0), axis=0), m.temperature)[0]
                preds.append(C[int(np.argmax(prob))])
            return sum(a == b for a, b in zip(preds, y)) / len(y)

        print(f'  global(基线)             {evaluate("global"):.4f}')
        for a in (0.5, 1.0, 2.0, 3.0):
            print(f'  tail_w      alpha={a:.1f}      {evaluate("tail_w", alpha=a):.4f}')
        for frac in (0.5, 0.35, 0.25):
            print(f'  tail_only   frac={frac:.2f}     {evaluate("tail_only", frac=frac):.4f}')
        for frac in (0.5, 0.35):
            print(f'  gmean_tailmax frac={frac:.2f}   {evaluate("gmean_tailmax", frac=frac):.4f}')


if __name__ == '__main__':
    main()

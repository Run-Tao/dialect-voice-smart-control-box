# -*- coding: utf-8 -*-
"""判定尾部加权池化是真机制改善还是噪声：
逐条对比 global vs tail_w(alpha) 的预测翻转，看救回/伤害的样本构成。
若救回集中在小置信度样本、且不伤害高置信度正确样本，则为真改善。
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
                                  CONV_LAYERS, prepare_input, softmax_np)

MANIFEST = ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'manifest_yue_v3_2.csv'


def conv_feat(params, x):
    x = np.asarray(x, dtype=np.float64)
    for name, cin, cout, k in CONV_LAYERS:
        x = _conv1d(x, params[name + '.weight'], params[name + '.bias'], k // 2)
        x = _batch_norm(x, params[name + '.bn.weight'], params[name + '.bn.bias'],
                        params[name + '.bn.running_mean'], params[name + '.bn.running_var'])
        x = _relu(x)
        if name in ('conv0', 'conv1'):
            x = _maxpool2(x)
    return x


def pooled(x, alpha):
    C, T = x.shape[1], x.shape[2]
    if alpha <= 0:
        return np.concatenate([x.mean(2), x.max(2)], 1)
    t = np.arange(T, dtype=np.float64)
    w = np.exp(alpha * (t / (T - 1)))
    w = w / w.sum()
    wm = (x * w[None, None, :]).sum(2)
    return np.concatenate([wm, x.max(2)], 1)


def main():
    m = V3Predictor.load(str(ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_v32b.npz'))
    C = list(m.classes_)
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding='utf-8-sig'))
            if r['split'] == 'test']
    data = []
    for r in rows:
        seq = m.feature_sequence(r['audio_path'])
        x = prepare_input(seq, m.n_frames)
        fs = [conv_feat(p, x)[0] for p in m._seed_params]
        data.append((fs, r['command'], r.get('text', ''), r.get('expression_type', '')))

    def predict(fs, alpha):
        zs = []
        for p, f in zip(m._seed_params, fs):
            e = pooled(f[None], alpha)
            zs.append(e @ p['fc.weight'].T + p['fc.bias'])
        prob = softmax_np(np.mean(np.stack(zs, 0), axis=0), m.temperature)[0]
        return C[int(np.argmax(prob))], float(prob.max())

    for alpha in (1.0, 2.0):
        rescued, hurt = [], []
        for fs, lab, text, typ in data:
            pg, cg = predict(fs, 0.0)
            pt, ct = predict(fs, alpha)
            if pg != lab and pt == lab:
                rescued.append((lab, pg, ct, cg, text, typ))
            elif pg == lab and pt != lab:
                hurt.append((lab, pt, cg, ct, text, typ))
        n = len(data)
        print(f'\n===== alpha={alpha} =====')
        print(f'救回 {len(rescued)} 条，伤害 {len(hurt)} 条 → 净 {len(rescued)-len(hurt):+d} 条'
              f'（{(len(rescued)-len(hurt))/n*100:+.2f}pp）')
        print('救回明细:')
        for lab, pg, ct, cg, text, typ in rescued:
            print(f'  [{cg:.3f}->{ct:.3f}] {lab:14s} (原判 {pg:14s}) {typ:14s} {text[:30]}')
        print('伤害明细:')
        for lab, pt, cg, ct, text, typ in hurt:
            print(f'  [{cg:.3f}->{ct:.3f}] {lab:14s} (改判 {pt:14s}) {typ:14s} {text[:30]}')


if __name__ == '__main__':
    main()

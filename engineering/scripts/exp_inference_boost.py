# -*- coding: utf-8 -*-
"""不改模型权重，纯粹在推理侧尝试提升准确率的实验：
  A) TTA 时间平移增广（循环平移 ±k 帧后平均概率）
  B) 关键词偏置：给某类别的 logit 加 bias（模拟 U2-KWS 简化版）
  C) margin/置信度感知的集成再加权
在冻结 test(483) 与 val(440) 上 A/B，找出无副作用的可部署增益。只读模型。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.v3_model import V3Predictor
from runtime.v3_cnn_numpy import prepare_input, softmax_np, embed as cnn_embed

MANIFEST = ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'manifest_yue_v3_2.csv'


def load_data(m, split):
    rows = [r for r in csv.DictReader(open(MANIFEST, encoding='utf-8-sig'))
            if r['split'] == split]
    seqs = [m.feature_sequence(r['audio_path']) for r in rows]
    y = [r['command'] for r in rows]
    return rows, seqs, y


def probs_tta(m, seq, shifts):
    """与 V3Predictor._forward 一致，但可指定 shifts。"""
    base = prepare_input(seq, m.n_frames)
    deployed = []
    for s in shifts:
        x = base if s == 0 else np.roll(base, int(s), axis=2)
        zs = []
        for si, p in enumerate(m._seed_params):
            emb = cnn_embed(p, x)[0]
            zs.append(emb @ p['fc.weight'].T + p['fc.bias'])
        deployed.append(softmax_np(np.mean(zs, axis=0, keepdims=True), m.temperature)[0])
    return np.mean(np.stack(deployed, 0), axis=0)


def acc(probs, y, classes):
    pred = [classes[int(np.argmax(p))] for p in probs]
    return sum(a == b for a, b in zip(pred, y)) / len(y), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_v32b.npz'))
    args = ap.parse_args()
    m = V3Predictor.load(args.model)
    C = list(m.classes_)
    print(f'模型 {m.model_name} classes={C}')

    data = {}
    for split in ('test', 'val'):
        rows, seqs, y = load_data(m, split)
        data[split] = (rows, seqs, y)
        print(f'{split}: {len(y)} 条')

    # 预计算各 shift 的 logits（一次前向，多次复用）
    shift_set = [0, -4, 4, -8, 8]
    cache = {}
    for split, (rows, seqs, y) in data.items():
        cache[split] = {}
        for s in shift_set:
            ps = []
            for seq in seqs:
                base = prepare_input(seq, m.n_frames)
                x = base if s == 0 else np.roll(base, int(s), axis=2)
                zs = [cnn_embed(p, x)[0] @ p['fc.weight'].T + p['fc.bias']
                      for p in m._seed_params]
                ps.append(softmax_np(np.mean(zs, axis=0, keepdims=True), m.temperature)[0])
            cache[split][s] = np.stack(ps)

    print('\n===== A) TTA 平移增广（平均概率）=====')
    combos = [[0], [0, -4, 4], [0, -8, 8], [-8, -4, 0, 4, 8]]
    for combo in combos:
        line = f'shifts={str(combo):28s}'
        for split in ('test', 'val'):
            _, y = data[split][0], data[split][2]
            p = np.mean(np.stack([cache[split][s] for s in combo], 0), axis=0)
            a, _ = acc(p, y, C)
            line += f'  {split}={a:.4f}'
        print(line)

    print('\n===== B) 关键词偏置（对 ALL_OFF / 窗帘类加 bias）=====')
    # 先取无 TTA 基线概率
    for split in ('test', 'val'):
        _, y = data[split][0], data[split][2]
        base = cache[split][0]
        a0, _ = acc(base, y, C)
        print(f'  {split} 基线={a0:.4f}')
    # bias 通过 logits 加，不是 prob；这里用 prob 近似（对数域加权）
    for cls in ['CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF']:
        ci = C.index(cls)
        for w in [1.05, 1.1, 1.2]:
            line = f'  bias {cls:15s} x{w:.2f}'
            for split in ('test', 'val'):
                _, y = data[split][0], data[split][2]
                p = cache[split][0].copy()
                p[:, ci] *= w
                p = p / p.sum(axis=1, keepdims=True)
                a, _ = acc(p, y, C)
                line += f'  {split}={a:.4f}'
            print(line)


if __name__ == '__main__':
    main()

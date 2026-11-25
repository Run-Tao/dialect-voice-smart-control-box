# -*- coding: utf-8 -*-
"""多模型在同一冻结集上的对比：准确率、混淆、以及逐条"救回/伤害"归因。
用法::
  python scripts/compare_models.py --models A.npz B.npz --split test
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.v3_model import V3Predictor

MANIFEST = ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'manifest_yue_v3_2.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    ap.add_argument('--show-flips', action='store_true', help='显示首尾模型间的翻转样本')
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(MANIFEST, encoding='utf-8-sig'))
            if r['split'] == args.split]
    print(f'{args.split}: {len(rows)} 条\n')

    preds_all = []
    for mp in args.models:
        m = V3Predictor.load(mp)
        ps, cs = [], []
        for r in rows:
            seq = m.feature_sequence(r['audio_path'])
            p = m._prediction_from_prob(m._forward(seq)['prob'])
            ps.append(p.command)
            cs.append(p.confidence)
        y = [r['command'] for r in rows]
        acc = sum(a == b for a, b in zip(ps, y)) / len(y)
        err = len(y) - sum(a == b for a, b in zip(ps, y))
        print(f'{Path(mp).name:42s} acc={acc:.4f} err={err:3d} '
              f'(T={m.temperature}, alpha={m.pool_alpha})')
        preds_all.append((mp, ps, cs))

    if len(preds_all) >= 2:
        _, first_p, first_c = preds_all[0]
        _, last_p, last_c = preds_all[-1]
        y = [r['command'] for r in rows]
        rescued, hurt = [], []
        for i, r in enumerate(rows):
            if first_p[i] != y[i] and last_p[i] == y[i]:
                rescued.append((i, r, first_p[i], first_c[i], last_c[i]))
            elif first_p[i] == y[i] and last_p[i] != y[i]:
                hurt.append((i, r, last_p[i], first_c[i], last_c[i]))
        print(f'\n首→尾：救回 {len(rescued)}，伤害 {len(hurt)}，净 {len(rescued)-len(hurt):+d}')
        if args.show_flips:
            print('救回:')
            for i, r, fp, fc, lc in rescued:
                print(f'  [{fc:.3f}->{lc:.3f}] {r["command"]:14s} (原判 {fp:14s}) '
                      f'{r.get("expression_type",""):14s} {r.get("text","")[:32]}')
            print('伤害:')
            for i, r, lp, fc, lc in hurt:
                print(f'  [{fc:.3f}->{lc:.3f}] {r["command"]:14s} (改判 {lp:14s}) '
                      f'{r.get("expression_type",""):14s} {r.get("text","")[:32]}')

        # 按命令统计尾模型错误
        print('\n尾模型按类别错误:')
        c = Counter()
        tot = Counter()
        for i, r in enumerate(rows):
            tot[r['command']] += 1
            if last_p[i] != r['command']:
                c[r['command']] += 1
        for k in tot:
            print(f'  {k:16s} 错 {c[k]:2d}/{tot[k]:3d} = {c[k]/tot[k]:5.1%}')


if __name__ == '__main__':
    main()

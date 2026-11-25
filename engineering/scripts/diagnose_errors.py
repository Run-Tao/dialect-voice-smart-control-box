# -*- coding: utf-8 -*-
"""深度错误诊断：导出 test/val 上每条错样本的全文、概率、表达类型，
并按多个维度聚合，找出可修复的真实模式。只读，不改任何工件。

用法::
  PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE /d/anaconda3/python.exe \
    scripts/diagnose_errors.py --model runtime/models/v3_cnn_ensemble_v32b.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.v3_model import V3Predictor

MANIFEST = ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'manifest_yue_v3_2.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_v32b.npz'))
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    ap.add_argument('--out', help='错误明细 JSON 输出')
    args = ap.parse_args()

    m = V3Predictor.load(args.model)
    print(f'模型: {m.model_name} | temp={m.temperature} | TTA shifts={m.tta_shifts}')
    rows = [r for r in csv.DictReader(open(args.manifest, encoding='utf-8-sig'))
            if r['split'] == args.split]
    print(f'{args.split} 集: {len(rows)} 条')

    errors = []
    cm = defaultdict(Counter)
    by_type = defaultdict(lambda: [0, 0])
    conf_correct, conf_wrong = [], []
    for r in rows:
        seq = m.feature_sequence(r['audio_path'])
        out = m._forward(seq)
        prob = out['prob']
        p = m._prediction_from_prob(prob)
        lab, pred = r['command'], p.command
        cm[lab][pred] += 1
        typ = r.get('expression_type', '?')
        by_type[typ][1] += 1
        (conf_wrong if pred != lab else conf_correct).append(p.confidence)
        if pred != lab:
            by_type[typ][0] += 1
            order = np.argsort(-prob)
            top3 = [(m.classes_[i], round(float(prob[i]), 4)) for i in order[:3]]
            errors.append({
                'uid': r.get('utterance_id', r.get('id', '?')),
                'lab': lab, 'pred': pred,
                'conf': round(p.confidence, 4),
                'margin': round(p.margin, 4),
                'text': r.get('text', ''),
                'type': typ,
                'voice': r.get('voice', r.get('speaker', '?')),
                'rate': r.get('rate', '?'),
                'top3': top3,
                'audio': r['audio_path'],
            })

    n = len(rows)
    nerr = len(errors)
    print(f'\nargmax 准确率: {(n-nerr)/n:.4f}  错误 {nerr}/{n}')
    print(f'正确样本平均置信度: {np.mean(conf_correct):.4f} | 错误样本: {np.mean(conf_wrong):.4f}')
    print('\n混淆矩阵（行=真实, 列=预测）:')
    classes = list(m.classes_)
    print(f'{"":16s}' + ''.join(f'{c[:7]:>9s}' for c in classes))
    for t in classes:
        print(f'{t:16s}' + ''.join(f'{cm[t][p]:9d}' for p in classes))
    print('\n按表达类型:')
    for t, (e, tot) in sorted(by_type.items(), key=lambda x: -x[1][0]):
        print(f'  {t:24s} 错 {e:2d}/{tot:3d} = {e/tot:5.1%}')
    print('\n错误明细（按置信度降序，越靠前越"自信地错"）:')
    for e in sorted(errors, key=lambda x: -x['conf']):
        t3 = ' '.join(f'{c}:{p:.3f}' for c, p in e['top3'])
        print(f"  [{e['conf']:.3f}] {e['lab']:14s}->{e['pred']:14s} "
              f"{e['text'][:34]:36s} | {t3}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            'model': args.model, 'split': args.split, 'n': n, 'n_err': nerr,
            'acc': (n - nerr) / n, 'errors': errors,
            'cm': {k: dict(v) for k, v in cm.items()},
            'by_type': {k: v for k, v in by_type.items()},
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n明细已写: {args.out}')


if __name__ == '__main__':
    main()

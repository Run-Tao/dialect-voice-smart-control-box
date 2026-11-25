# -*- coding: utf-8 -*-
"""准确率 A/B 对比：基线模型 vs v3.2 纠错覆盖修复模型。

两模型在同一份冻结测试集（manifest_yue_v3_1_vad.csv 的 split=test，483 条）上评估，
输出总体准确率、混淆矩阵、按纠错句式/类别分组的对比。只读，不改任何模型或工件。

用法::
  PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE /d/anaconda3/python.exe \
    scripts/eval_accuracy_ab.py --new ../DialectVoice_Research/data/models/v3_cnn_ensemble_v32.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.v3_model import V3Predictor

MANIFEST = ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'manifest_yue_v3_1_vad.csv'
BASE_MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'


def load_rows(split='test'):
    return [r for r in csv.DictReader(open(MANIFEST, encoding='utf-8-sig'))
            if r['split'] == split]


def evaluate(model: V3Predictor, rows):
    cm = defaultdict(Counter)
    err_by_text = defaultdict(lambda: [0, 0])
    err_by_type = defaultdict(lambda: [0, 0])
    err_by_cmd = defaultdict(lambda: [0, 0])
    n = 0
    for r in rows:
        try:
            p = model.predict_features(model.feature_sequence(r['audio_path']))
        except Exception:
            continue
        n += 1
        lab, pred = r['command'], p.command
        cm[lab][pred] += 1
        for key, table in (
            (r['text'], err_by_text),
            (r['expression_type'], err_by_type),
            (lab, err_by_cmd),
        ):
            table[key][1] += 1
            if pred != lab:
                table[key][0] += 1
    acc = sum(cm[l][l] for l in model.classes_) / n if n else 0.0
    return {'n': n, 'acc': acc, 'cm': cm, 'text': err_by_text,
            'type': err_by_type, 'cmd': err_by_cmd}


def show(name, res, classes):
    print(f'\n===== {name} =====')
    print(f'样本 {res["n"]}  argmax 准确率 {res["acc"]:.4f}  '
          f'（错误 {res["n"] - sum(res["cm"][l][l] for l in classes)}）')
    print('混淆矩阵（行=真实，列=预测）:')
    print(f'{"":16s}' + ''.join(f'{c[:7]:>9s}' for c in classes))
    for t in classes:
        print(f'{t:16s}' + ''.join(f'{res["cm"][t][p]:9d}' for p in classes))
    print('按类别:')
    for c in classes:
        e, tot = res['cmd'][c]
        print(f'  {c:16s} 错 {e:2d}/{tot:3d} = {e/tot:5.1%}' if tot else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True, help='v3.2 修复模型 npz 路径')
    ap.add_argument('--split', default='test', choices=['test', 'val'],
                    help='评估子集（默认 test；val 用作过拟合交叉检查）')
    ap.add_argument('--out', help='对比报告 JSON 输出路径（可选）')
    args = ap.parse_args()

    rows = load_rows(args.split)
    base = V3Predictor.load(str(BASE_MODEL))
    new = V3Predictor.load(args.new)
    print(f'基线: {base.model_name}  temperature={base.temperature}')
    print(f'新模型: {new.model_name}  temperature={new.temperature}')
    print(f'冻结 {args.split} 集: {len(rows)} 条')

    rb = evaluate(base, rows)
    rn = evaluate(new, rows)
    classes = list(base.classes_)
    show('基线 v3_cnn_ensemble_vad', rb, classes)
    show('v3.2 纠错覆盖修复', rn, classes)

    print('\n===== 关键差异（修复的错误）=====')
    print(f'总体准确率: {rb["acc"]:.4f} -> {rn["acc"]:.4f}  '
          f'({(rn["acc"] - rb["acc"]) * 100:+.2f} pp)')
    print('纠错句式文本对比:')
    texts = set(rb['text']) | set(rn['text'])
    for t in sorted(texts):
        eb, tb = rb['text'][t]
        en, tn = rn['text'][t]
        if eb or en:
            print(f'  {t[:28]:30s} 基线 {eb:2d}/{tb:2d} -> 新 {en:2d}/{tn:2d}')

    if args.out:
        def ser(res):
            return {
                'n': res['n'], 'acc': res['acc'],
                'cm': {k: dict(v) for k, v in res['cm'].items()},
                'cmd': {k: v for k, v in res['cmd'].items()},
                'text': {k: v for k, v in res['text'].items()},
                'type': {k: v for k, v in res['type'].items()},
            }
        data = {'base': ser(rb), 'new': ser(rn),
                'base_model': str(BASE_MODEL), 'new_model': args.new,
                'n_test': len(rows)}
        Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'\n报告已写: {args.out}')


if __name__ == '__main__':
    main()

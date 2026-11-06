# -*- coding: utf-8 -*-
"""从实验预测 CSV 做逐表达组误差分析（供报告 §5.4 使用）。

预测 CSV 由 scripts/run_v3_experiments.py 的 _save_predictions 生成，字段：
utterance_id, expression_group_id, command, condition, register, voice,
pred_command, correct, confidence

用法::

    D:\\anaconda3\\python.exe docs/report_assets/analyze_v3_errors.py \
        --pred outputs/v3_experiments/main_v3_1/predictions_E4_cnn_tta_final.csv

输出：
  * 每个表达组的准确率（升序，最差在前）
  * correction 语域逐组准确率
  * 按 condition / register / voice 汇总
只打印，不写文件（报告正文人工核对后引用）。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]


def load(path: Path):
    rows = []
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            r['correct'] = int(r['correct'])
            r['confidence'] = float(r['confidence'])
            rows.append(r)
    return rows


def group_stats(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['expression_group_id']].append(r)
    out = []
    for g, items in by.items():
        n = len(items)
        acc = sum(x['correct'] for x in items) / n
        out.append({
            'group': g,
            'command': items[0]['command'],
            'register': items[0]['register'],
            'n': n,
            'acc': acc,
            'errors': n - sum(x['correct'] for x in items),
        })
    return sorted(out, key=lambda x: (x['acc'], -x['n']))


def summarize(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r['correct'])
    return {k: {'n': len(v), 'acc': sum(v) / len(v)} for k, v in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', nargs='+', required=True)
    args = ap.parse_args()
    for p in args.pred:
        path = Path(p) if Path(p).is_absolute() else BASE / p
        rows = load(path)
        print(f'=== {path.name}  n={len(rows)}  acc={sum(r["correct"] for r in rows)/len(rows):.4f}')
        for key in ('condition', 'register', 'voice'):
            print(f'-- by {key}')
            for k, v in summarize(rows, key).items():
                print(f'   {k:28s} n={v["n"]:3d} acc={v["acc"]:.4f}')
        print('-- 最差 12 个表达组')
        for g in group_stats(rows)[:12]:
            print(f'   {g["group"]:22s} {g["command"]:14s} {g["register"]:12s} '
                  f'n={g["n"]:3d} acc={g["acc"]:.4f} err={g["errors"]}')
        print('-- correction 语域逐组')
        for g in group_stats(rows):
            if g['register'] == 'correction':
                print(f'   {g["group"]:22s} n={g["n"]:3d} acc={g["acc"]:.4f} err={g["errors"]}')
        print('-- 错误对（真实→预测, 前 10）')
        err = Counter()
        for r in rows:
            if not r['correct']:
                err[(r['command'], r['pred_command'])] += 1
        for (a, b), c in err.most_common(10):
            print(f'   {a:14s} -> {b:14s} {c}')
        print()


if __name__ == '__main__':
    main()

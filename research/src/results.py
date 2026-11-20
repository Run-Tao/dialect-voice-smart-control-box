# -*- coding: utf-8 -*-
"""实验结果保存。"""
import json
from pathlib import Path


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def save_summary_csv(path, rows, fieldnames):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    return path


def load_json(path):
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

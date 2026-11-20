# -*- coding: utf-8 -*-
"""核心对照：v1 历史数据 vs 扩充后数据，在同一固定测试集上的配对比较。

为什么要配对：
  E1 与 E2-100% 是在同一 test 集上评估的，可以直接逐样本比较（配对设计），
  比只看两个 Accuracy 数字更能判断「提升是否真实」。本脚本输出：

  1. 逐样本 2x2 列联表（两模型都对 / v1对扩充错 / v1错扩充对 / 都错）
  2. McNemar 精确二项检验 p 值（双侧）
  3. 按表达组重采样的配对自助法：Accuracy 差值的 95% 置信区间
     （按组重采样，避免同组高度相关样本被当成独立样本）
  4. 每类 F1 差值

注意：本脚本只读取固定划分与固定测试集，不调任何参数。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.config import load_config  # noqa: E402
from src.dataset_v2 import INTENTS, sha256_file  # noqa: E402
from scripts.run_v2_experiments import (  # noqa: E402
    FeatureCache, build_templates, evaluate, load_split, load_usable_records,
    subset, classification_report,
)

OUT = BASE / 'outputs' / 'v2_v1_vs_v2_comparison.json'


def mcnemar_exact(b, c):
    """McNemar 精确检验：H0 下 b、c 服从 Binomial(b+c, 0.5)。双尾 p。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def paired_group_bootstrap(rows_a, rows_b, n_boot=5000, seed=42):
    """按表达组重采样的 Accuracy 差值置信区间。rows_* 为 evaluate 返回的逐样本行。"""
    by_group = defaultdict(list)
    for ra, rb in zip(rows_a, rows_b):
        assert ra['utterance_id'] == rb['utterance_id']
        by_group[ra['expression_group_id']].append(
            (1.0 if ra['pred_command'] == ra['true_command'] else 0.0,
             1.0 if rb['pred_command'] == rb['true_command'] else 0.0))
    groups = sorted(by_group)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        a_ok = b_ok = n = 0
        for i in pick:
            for va, vb in by_group[groups[i]]:
                a_ok += va
                b_ok += vb
                n += 1
        diffs.append((b_ok - a_ok) / n)
    diffs = np.array(diffs)
    return {
        'acc_diff_mean': float(np.mean(diffs)),
        'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        'p_two_sided_bootstrap': float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))),
        'n_groups': len(groups),
    }


def main():
    config = load_config(BASE / 'config.json')
    group2subset, split_meta = load_split()
    records = load_usable_records()
    test = subset(records, group2subset, 'test')
    train_full = subset(records, group2subset, 'train')
    train_v1 = [r for r in train_full if r.generation_batch == 'v1']

    cache = FeatureCache(config)
    print(f"test={len(test)}  train_v1={len(train_v1)}  train_full={len(train_full)}", flush=True)

    tpl_v1 = build_templates(train_v1, cache, 'all', None, 42)
    print(f"v1 模板: {sum(len(v) for v in tpl_v1.values())}", flush=True)
    m_v1, rows_v1 = evaluate(test, tpl_v1, cache, config, 60)
    print(f"v1   acc={m_v1['accuracy']:.4f} f1={m_v1['macro_f1']:.4f}", flush=True)

    tpl_full = build_templates(train_full, cache, 'all', None, 42)
    print(f"full 模板: {sum(len(v) for v in tpl_full.values())}", flush=True)
    m_full, rows_full = evaluate(test, tpl_full, cache, config, 60)
    print(f"full acc={m_full['accuracy']:.4f} f1={m_full['macro_f1']:.4f}", flush=True)

    # 2x2 列联表
    b = sum(1 for a, c in zip(rows_v1, rows_full)
            if a['pred_command'] == a['true_command'] and c['pred_command'] != c['true_command'])
    cc = sum(1 for a, c in zip(rows_v1, rows_full)
             if a['pred_command'] != a['true_command'] and c['pred_command'] == c['true_command'])
    both = sum(1 for a, c in zip(rows_v1, rows_full)
               if a['pred_command'] == a['true_command'] and c['pred_command'] == c['true_command'])
    neither = len(test) - both - b - cc
    p = mcnemar_exact(b, cc)
    boot = paired_group_bootstrap(rows_v1, rows_full)

    # 每类 F1 差值
    f1_delta = {}
    for cmd in INTENTS:
        f1_delta[cmd] = {
            'v1_f1': m_v1['per_class'][cmd]['f1'],
            'full_f1': m_full['per_class'][cmd]['f1'],
            'delta': round(m_full['per_class'][cmd]['f1'] - m_v1['per_class'][cmd]['f1'], 4),
            'v1_support': m_v1['per_class'][cmd]['support'],
        }

    # 主要错误对：full 模型的真实→预测分布（排除正确）
    err_pairs = defaultdict(int)
    for r in rows_full:
        if r['pred_command'] != r['true_command']:
            err_pairs[(r['true_command'], r['pred_command'])] += 1
    err_top = sorted(err_pairs.items(), key=lambda kv: -kv[1])[:12]

    out = {
        'protocol': {
            'test_set': 'data/splits/track_a_expression_split.csv (subset=test)',
            'split_sha256': split_meta['sha256'],
            'test_size': len(test),
            'test_groups': len({r.expression_group_id for r in test}),
            'train_v1_size': len(train_v1),
            'train_full_size': len(train_full),
            'template_strategy': 'all（原项目默认，min 聚合）',
            'max_frames': 60,
            'note': '两模型在同一固定测试集上评估，因此可做配对比较；未调任何参数。',
        },
        'v1': {'accuracy': m_v1['accuracy'], 'macro_f1': m_v1['macro_f1'],
               'per_class': m_v1['per_class'], 'confusion_matrix': m_v1['confusion_matrix'],
               'labels': m_v1['labels']},
        'full': {'accuracy': m_full['accuracy'], 'macro_f1': m_full['macro_f1'],
                 'per_class': m_full['per_class'],
                 'confusion_matrix': m_full['confusion_matrix'],
                 'labels': m_full['labels']},
        'paired_comparison': {
            'both_correct': both,
            'v1_only_correct': b,
            'full_only_correct': cc,
            'neither': neither,
            'mcnemar_exact_p_two_sided': p,
            'accuracy_delta_full_minus_v1': float(m_full['accuracy'] - m_v1['accuracy']),
            'macro_f1_delta_full_minus_v1': float(m_full['macro_f1'] - m_v1['macro_f1']),
            'group_bootstrap_acc_diff': boot,
            'interpretation': (
                'McNemar p>0.05 且自助法差值 CI 跨 0 ⇒ 扩充数据前后差异不显著。'
                if (p > 0.05 and boot['ci95'][0] <= 0 <= boot['ci95'][1])
                else '差异在统计上可区分，需结合效应量与置信区间解释。'),
        },
        'per_class_f1_delta': f1_delta,
        'full_model_top_error_pairs': [
            {'true': a, 'pred': b_, 'count': n} for (a, b_), n in err_top],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out['paired_comparison'], ensure_ascii=False, indent=2))
    print('saved', OUT)


if __name__ == '__main__':
    main()

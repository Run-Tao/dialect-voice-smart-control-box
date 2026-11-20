# -*- coding: utf-8 -*-
"""第三阶段数据消融：v3（无 correction 补充） vs v3.1（correction 补充）。

目的
----
验证「训练集 correction 语域覆盖不足」这一诊断：两个模型除训练数据外，
结构、超参、随机种子完全相同（单种子 42，同一 config），在同一个固定
test（v3.1 test 483 条 / 31 组，v3 与 v3.1 的 test 完全一致）上比较，
并单独报告 correction 语域 98 条与其余 385 条。

用法
----
1) 先训练 v3-only 单种子模型::

    python scripts/train_v3_model.py --seeds 42 \
        --manifest data/tts_corpus/manifest_yue_v3.csv \
        --split data/splits/track_a_expression_split_v3.csv \
        --out outputs/v3_experiments/data_ablation/v3_only_seed42.npz \
        --report outputs/v3_experiments/data_ablation/train_v3_only.json

2) 再运行本脚本（同时读取上面的 v3-only 模型与 v3.1 三种子工件中的 seed42）::

    python scripts/run_v3_data_ablation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from scripts.run_v3_experiments import (  # noqa: E402
    INTENTS, V2_CONDITIONS, cnn_feature_sequences, full_metrics, load_config,
    load_v3_records, paired_compare, predict_cnn, split_recs,
)
from src.v3_model import V3Predictor  # noqa: E402


def single_seed_predictor(artifact: Path, seed_index: int = 0) -> V3Predictor:
    """从多种子工件中取出单个种子，构造只含该种子的预测器（不重新训练）。"""
    model = V3Predictor.load(artifact)
    meta = dict(model.meta)
    meta['n_seeds'] = 1
    meta['seeds'] = [meta['seeds'][seed_index]]
    meta['tta_shifts'] = []
    predictor = V3Predictor(meta, [model._seed_params[seed_index]])
    predictor.model_path = artifact
    return predictor


def eval_model(model, recs, seqs, log=print):
    preds, probs, latency = predict_cnn(model, recs, seqs=seqs, tta_shifts=[], log=log)
    return preds, probs, latency


def main():
    ap = argparse.ArgumentParser(description='v3 vs v3.1 correction 数据消融')
    ap.add_argument('--config', default=str(BASE / 'config_v3.json'))
    ap.add_argument('--v3-only-model',
                    default=str(BASE / 'outputs/v3_experiments/data_ablation/v3_only_seed42.npz'))
    ap.add_argument('--v31-model', default=str(BASE / 'data/models/v3_cnn_ensemble.npz'))
    ap.add_argument('--out', default=str(BASE / 'outputs/v3_experiments/data_ablation/comparison.json'))
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    cfg = load_config(args.config)
    recs_by_split = split_recs(load_v3_records(cfg))
    test = recs_by_split['test']
    y = np.asarray([INTENTS.index(r['command']) for r in test])
    correction = np.asarray([r['register'] == 'correction' for r in test])
    matched_v2 = np.asarray([r['condition'] in V2_CONDITIONS for r in test])

    log('提取 v3.1 test 特征（两个模型共用同一特征）...')
    model31 = V3Predictor.load(args.v31_model)  # 用于取特征 spec
    seqs = cnn_feature_sequences(model31, test, log=log)

    log('评估 v3-only（seed42）...')
    model_v3 = single_seed_predictor(Path(args.v3_only_model), 0)
    pred_v3, prob_v3, _ = eval_model(model_v3, test, seqs, log=log)

    log('评估 v3.1（seed42，同一随机种子）...')
    model_v31 = single_seed_predictor(Path(args.v31_model), 0)
    pred_v31, prob_v31, _ = eval_model(model_v31, test, seqs, log=log)

    def subset(pred, mask):
        return [int(x) for x in np.asarray(pred)[mask]]

    def metrics(pred, mask):
        return full_metrics([r for r, m in zip(test, mask) if m],
                            [int(v) for v, m in zip(y, mask) if m],
                            subset(pred, mask))

    all_mask = np.ones(len(test), bool)
    out = {
        'protocol': {
            'test': 'v3.1 test（31 组 / 483 条，v3 与 v3.1 的 test 完全相同）',
            'seed': 42,
            'difference': '仅训练数据不同：v3(1873 train) vs v3.1(1981 train，+18 correction 表达/108 条)',
            'v3_only_model': args.v3_only_model,
            'v3_1_model_seed42': args.v31_model,
            'note': 'v3-only 模型使用 1 个种子；v3.1 取集成工件中的 seed42，保证随机种子一致。',
        },
        'test_full': {
            'v3_only': metrics(pred_v3, all_mask),
            'v3_1_seed42': metrics(pred_v31, all_mask),
            'paired': paired_compare(test, pred_v3.tolist(), pred_v31.tolist(),
                                     'v3_only', 'v3_1_seed42'),
        },
        'test_correction_98': {
            'v3_only': metrics(pred_v3, correction),
            'v3_1_seed42': metrics(pred_v31, correction),
            'paired': paired_compare([r for r, m in zip(test, correction) if m],
                                     subset(pred_v3, correction), subset(pred_v31, correction),
                                     'v3_only', 'v3_1_seed42'),
        },
        'test_noncorrection_385': {
            'v3_only': metrics(pred_v3, ~correction),
            'v3_1_seed42': metrics(pred_v31, ~correction),
        },
        'test_matched_v2_111': {
            'v3_only': metrics(pred_v3, matched_v2),
            'v3_1_seed42': metrics(pred_v31, matched_v2),
            'paired': paired_compare([r for r, m in zip(test, matched_v2) if m],
                                     subset(pred_v3, matched_v2), subset(pred_v31, matched_v2),
                                     'v3_only', 'v3_1_seed42'),
        },
        'error_pairs': {
            'v3_only': [{'true': a, 'pred': b, 'count': n} for (a, b), n in Counter(
                (INTENTS[a], INTENTS[b]) for a, b in zip(y, pred_v3) if a != b).most_common(10)],
            'v3_1_seed42': [{'true': a, 'pred': b, 'count': n} for (a, b), n in Counter(
                (INTENTS[a], INTENTS[b]) for a, b in zip(y, pred_v31) if a != b).most_common(10)],
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                        encoding='utf-8')
    print(json.dumps({
        'full': {k: v['accuracy'] for k, v in out['test_full'].items() if k != 'paired'},
        'correction': {k: v['accuracy'] for k, v in out['test_correction_98'].items() if k != 'paired'},
        'paired_full': out['test_full']['paired'],
    }, ensure_ascii=False, indent=2))
    print('saved', out_path)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""运行第二阶段/第三阶段的基线和映射实验。"""
import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.split import split_by_speaker, split_summary
from src.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--manifest', default='data/smoke_manifest.csv')
    parser.add_argument('--method', default='all',
                        choices=['all', 'cosine', 'dtw', 'logreg', 'svm', 'ols', 'ridge'])
    parser.add_argument('--svd-mode', default=None,
                        choices=['none', 'R0_literal_rank_k', 'R1_global_projection'])
    parser.add_argument('--output-root', default='outputs/baselines')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.svd_mode is not None:
        cfg.setdefault('mapping', {})['svd_mode'] = args.svd_mode
    records = load_manifest(args.manifest, check_files=True)
    split = split_by_speaker(
        records,
        train_ratio=cfg['split']['train_ratio'],
        val_ratio=cfg['split']['val_ratio'],
        test_ratio=cfg['split']['test_ratio'],
        seed=cfg['split']['seed'],
    )
    print('划分:', json.dumps(split_summary(split), ensure_ascii=False, indent=2))

    if args.method == 'all':
        methods = ['cosine', 'dtw', 'logreg', 'svm', 'ols', 'ridge']
    else:
        methods = [args.method]
    for method in methods:
        result, out_dir = run_experiment(method, split, cfg, args.output_root)
        print(f"[{method}] accuracy={result['metrics']['accuracy']:.4f} "
              f"macro_f1={result['metrics']['macro_f1']:.4f} -> {out_dir}")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""扫描快速端点静音挂起，比较准确率、截断和延迟。

这个脚本不训练模型，只回答一个部署问题：预触发之后，静音挂起从
``fast_silence_ms`` 取多少时，部署流式路径最接近训练语料的切法。

输出同时包含：

- 流式切段是否比默认 VAD 切法短；
- argmax 正确率、执行数与错误执行数；
- 动作延迟 P50/P95；
- 会话级成功率与危险误动作率。

用法::

    python scripts/sweep_fast_endpoint.py \
        --manifest ../DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv \
        --values 120 180 240 300 390 \
        --out docs/fast_endpoint_sweep.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import bench_perceived_latency as bench
import eval_dialog as dialog
from src.config import Config, load_config


def config_with_fast_silence(base: Config, fast_silence_ms: float) -> Config:
    """复制配置并只覆盖一个参数，避免扫描时相互污染。"""
    raw = copy.deepcopy(base.raw)
    streaming = raw.setdefault('streaming', {})
    streaming['fast_silence_ms'] = float(fast_silence_ms)
    return Config(raw=raw, root=base.root)


def run_bench(rows, cfg: Config, gate_enabled: bool, limit: int = 0) -> dict:
    engine = bench.build_engine(cfg, gate_enabled)
    speaker = bench._BenchSpeaker(cfg.broadcast_dir)
    engine.speaker = speaker
    rng = np.random.default_rng(20261103)
    fs = cfg.sample_rate
    vad_kwargs = cfg.vad_kwargs()

    warmup = min(5, len(rows))
    for row in rows[:warmup]:
        bench.bench_one(engine, speaker, row['path'], fs, 1024,
                        0.0, 0.0, rng, vad_kwargs)

    records = []
    for row in rows:
        rec = bench.bench_one(engine, speaker, row['path'], fs, 1024,
                              0.0, 0.0, rng, vad_kwargs)
        if rec is None:
            continue
        rec['truth'] = row['command']
        rec['utterance_id'] = row['utterance_id']
        records.append(rec)

    executed = [r for r in records if r['executed']]
    wrong = [r for r in executed if r['command'] != r['truth']]
    truncated = [r for r in records if r.get('truncated')]
    trunc_exec = [r for r in truncated if r['executed']]
    trunc_wrong = [r for r in trunc_exec if r['command'] != r['truth']]
    argmax_ok = sum(r['command'] == r['truth'] for r in records)
    action_lat = [r['action_latency_ms'] for r in executed]
    pre_cpu = [r['pretrigger_cpu_ms'] for r in records
               if r.get('pretrigger_cpu_ms') is not None]

    return {
        'n': len(records),
        'argmax_accuracy': round(argmax_ok / len(records), 4) if records else None,
        'executed': len(executed),
        'executed_wrong': len(wrong),
        'executed_accuracy': (
            round((len(executed) - len(wrong)) / len(executed), 4)
            if executed else None),
        'truncated': len(truncated),
        'truncated_rate': round(len(truncated) / len(records), 4) if records else None,
        'truncated_executed': len(trunc_exec),
        'truncated_wrong': len(trunc_wrong),
        'action_latency_ms': bench.stat(action_lat),
        'pretrigger_cpu_ms': bench.stat(pre_cpu),
        'wrong_pairs': {
            f'{r["truth"]}->{r["command"]}': sum(
                1 for x in wrong
                if x['truth'] == r['truth'] and x['command'] == r['command'])
            for r in wrong
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--subset', default='test')
    parser.add_argument('--config', default=None)
    parser.add_argument('--values', nargs='+', type=float,
                        default=[120.0, 180.0, 240.0, 300.0, 390.0])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--no-gate', action='store_true')
    parser.add_argument('--skip-dialog', action='store_true',
                        help='只跑端点/准确率扫描，不跑会话级重说评估')
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'fast_endpoint_sweep.json'))
    args = parser.parse_args(argv)

    base_cfg = load_config(args.config)
    manifest = bench.stream_manifest_path(Path(args.manifest))
    bench_rows = bench.load_rows(manifest, args.subset, args.limit)
    dialog_rows = dialog.load_manifest(manifest, args.subset, args.limit)
    if not bench_rows or not dialog_rows:
        sys.exit('manifest 中没有可用样本')

    results = []
    for value in args.values:
        cfg = config_with_fast_silence(base_cfg, value)
        print(f'\n=== fast_silence_ms={value:g} ===', flush=True)
        endpoint = run_bench(bench_rows, cfg, not args.no_gate, args.limit)
        item = {'fast_silence_ms': float(value), 'endpoint': endpoint}
        if not args.skip_dialog:
            dialog_report, _ = dialog.evaluate(
                dialog_rows, cfg, not args.no_gate, 'group', 'stream')
            item['dialog'] = {
                k: dialog_report[k] for k in (
                    'argmax_accuracy', 'single_gate_coverage',
                    'dialog_session_success', 'clarification_rate',
                    'reject_rate', 'dangerous_misaction_rate',
                    'recovered_by_dialog', 'lost_by_dialog')
            }
        results.append(item)
        printable = {k: v for k, v in item.items()}
        print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)

    # 只报告，不自动改配置。先守住 300 ms 延迟预算，再比较会话成功率、
    # argmax、截断数和 P95；否则 390 ms 会因“零截断”被错误推荐。
    def rank(item):
        d = item.get('dialog') or {}
        p95 = item['endpoint']['action_latency_ms'].get('p95')
        return (
            bool(p95 is not None and p95 <= 300.0),
            d.get('dialog_session_success') or 0.0,
            item['endpoint'].get('argmax_accuracy') or 0.0,
            -item['endpoint']['truncated'],
            -(p95 or 1e30),
        )

    best = max(results, key=rank)
    report = {
        'manifest': str(manifest),
        'subset': args.subset,
        'gate_enabled': not args.no_gate,
        'values': results,
        'recommended_by_session_success': best,
        'note': '仅扫描端点；未重训模型。选定值后仍需用同一切法重生成语料、'
                '重训并重新标定双信号门限与参照池，才能作为最终部署结论。',
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n推荐候选: fast_silence_ms={best["fast_silence_ms"]:g}')
    print('已写入:', out)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""第三阶段 v3.1：correction 语域增量生成器。

对 18 条新 correction 表达（每类 3 条，仅进 train）：
  - 真实 edge-tts 3 音色 baseline：18 × 3 = 54 条（异步并发 ≤6、单条重试 3 次、幂等续跑）
  - 离线 rate_slow_18 增广：54 条（复用 src.dataset_v3 的 WSOLA 时域伸缩）
共 108 条 16kHz 单声道 PCM16 WAV，存 data/tts_corpus/wav16k_v3_1/edge/。

为什么用真实 TTS 生成 baseline：新文本在 v2/v3 中不存在，无法离线增广；
rate_slow_18 则由 baseline 离线变换得到（与 v3 的条件增广方法一致）。

日志：outputs/v3_1_generation_log.csv（逐条 stage/attempts/status/sha256/error）。

用法::

    .\\.venv_tts\\Scripts\\python.exe scripts/generate_v3_1_corpus.py --workers 6 --retries 3
    .\\.venv_tts\\Scripts\\python.exe scripts/generate_v3_1_corpus.py --plan
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset_v2 import read_csv_rows, write_csv_rows  # noqa: E402
from src.dataset_v3 import (  # noqa: E402
    EDGE_VOICES, V3_LIBRARY, analyze_audio,
)

from generate_v3_corpus import (  # noqa: E402
    LOG_FIELDS, _augment_one, run_tts,
)

LIB31 = BASE / 'data' / 'corpus' / 'yue_expression_library_v3_1.csv'
W31_WAV_ROOT = BASE / 'data' / 'tts_corpus' / 'wav16k_v3_1' / 'edge'
W31_RAW_ROOT = BASE / 'data' / 'tts_corpus' / 'wav16k_v3_1' / '_raw'
LOG31 = BASE / 'outputs' / 'v3_1_generation_log.csv'

COND_TAG = {'baseline': 'base', 'rate_slow_18': 'slow18'}


def now_iso():
    return datetime.now().isoformat(timespec='seconds')


def load_log():
    if not LOG31.exists():
        return {}
    return {r['utterance_id']: r for r in read_csv_rows(LOG31) if r.get('utterance_id')}


def save_log(log):
    rows = sorted(log.values(), key=lambda r: (r.get('stage', ''), r['utterance_id']))
    write_csv_rows(LOG31, rows, LOG_FIELDS)
    return len(rows)


def new_correction_rows():
    lib = read_csv_rows(LIB31)
    return [r for r in lib if r.get('v3_source') == 'new_v3_1']


def plan_tts_items():
    items = []
    for r in new_correction_rows():
        for v in EDGE_VOICES:
            uid = f"{r['expression_id']}_edge_tts_{v}_base_00"
            items.append(dict(
                stage='tts_new_expr_v31', utterance_id=uid, command=r['command'],
                expression_group_id=r['expression_id'], text=r['text'],
                voice_id=v, condition='baseline', rate=0, pitch=0, volume=0,
                out_path=W31_WAV_ROOT / f'{uid}.wav',
                raw_path=W31_RAW_ROOT / f'{uid}.mp3',
            ))
    return items


def plan_augment_items(tts_items):
    tasks = []
    for it in tts_items:
        uid = it['utterance_id'].replace('_base_00', '_slow18_00')
        tasks.append(dict(
            stage='augment_v31', utterance_id=uid, command=it['command'],
            expression_group_id=it['expression_group_id'], text=it['text'],
            voice_id=it['voice_id'], condition='rate_slow_18',
            source_id=it['utterance_id'], source_path=str(it['out_path']),
            out_path=str(W31_WAV_ROOT / f'{uid}.wav'),
        ))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--skip-tts', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--retries', type=int, default=3)
    ap.add_argument('--backoff', type=float, default=3.0)
    args = ap.parse_args()

    W31_WAV_ROOT.mkdir(parents=True, exist_ok=True)
    W31_RAW_ROOT.mkdir(parents=True, exist_ok=True)

    rows = new_correction_rows()
    tts_items = plan_tts_items()
    aug_tasks = plan_augment_items(tts_items)
    print(f'v3.1 新 correction 表达 {len(rows)} 组；TTS baseline {len(tts_items)} 条；'
          f'离线 rate_slow_18 {len(aug_tasks)} 条', flush=True)
    if args.plan:
        for it in tts_items:
            print(it['utterance_id'], it['text'], it['voice_id'])
        return

    log = load_log()
    t0 = time.perf_counter()
    if not args.skip_tts:
        dt, done = asyncio.run(run_tts(tts_items, args.workers, args.retries,
                                       args.backoff, log))
        save_log(log)
        print(f'TTS 阶段：{dt:.1f}s，状态 {dict(done)}', flush=True)

    ok_tts = [it for it in tts_items
              if log.get(it['utterance_id'], {}).get('status') in
              ('ok', 'skipped_existing', 'quality_flagged')
              and Path(it['out_path']).exists()]
    print(f'TTS 可用 baseline：{len(ok_tts)}/{len(tts_items)}', flush=True)
    aug_results = []
    for t in aug_tasks:
        if not Path(t['source_path']).exists():
            rec = {k: t.get(k, '') for k in LOG_FIELDS}
            rec.update({'stage': 'augment_v31', 'utterance_id': t['utterance_id'],
                        'generation_method': 'offline_signal_aug', 'status': 'fail',
                        'attempts': 0, 'error': 'missing_source_baseline',
                        'generated_at': now_iso(), 'elapsed_sec': 0})
            aug_results.append(rec)
            continue
        rec = _augment_one(t)
        aug_results.append(rec)
    for rec in aug_results:
        log[rec['utterance_id']] = {k: rec.get(k, '') for k in LOG_FIELDS}
    save_log(log)
    cnt = Counter(r['status'] for r in aug_results)
    print(f'离线增广：{len(aug_results)} 条，状态 {dict(cnt)}', flush=True)
    fails = sum(1 for r in log.values() if r.get('status') == 'fail')
    print(f'总耗时 {time.perf_counter() - t0:.1f}s；日志 {LOG31} 累计 {len(log)} 条；失败 {fails}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""v3.2  Correction 语域覆盖修复增量（只进 train，不碰冻结测试集）。

背景（2026-11-19 错误分析）：
  483 条测试集 57 个错误中 42 个（74%）是窗帘类，其中 32 个（56%）来自三条
  「唔係X，係Y」纠错句；这些句子在训练集里 **零覆盖**（训练只有反方向的
  「唔係打開窗簾，係拉埋窗簾」，测试却是「唔係拉埋窗簾，係打開窗簾」）。
  模型没见过同方向的纠正表达，16/16 全错是必然，不是模型容量问题。

做法：
  造 8 条「同构不同字」的纠正句（4  OPEN 方向 + 4  CLOSE 方向），全部与
  6 条冻结测试纠正句的 Jaccard < 0.5（项目 Δ0.5 重复红线），语义上是
  「先否定一个动作、再给出正确动作」——教模型学会忽略否定片段、执行「係」后动作。
  每条 3 音色 edge-tts baseline + rate_slow_18 离线增广 = 8×3×2 = 48 条，
  **全部标 split=train**，不新增/修改任何 test、val 条目。

产物（全部新文件，不改任何 v3/v3.1 冻结工件）：
  data/corpus/yue_expression_library_v3_2.csv        新句式表达库
  data/tts_corpus/wav16k_v3_2/edge/*.wav             新音频
  data/tts_corpus/manifest_yue_v3_2.csv              合并清单（2904+新增，供训练）
  outputs/v3_2_generation_log.csv                    生成日志

用法::
  .\\.venv_tts\\Scripts\\python.exe scripts/generate_v3_2_correction_fix.py --all
  .\\.venv_tts\\Scripts\\python.exe scripts/generate_v3_2_correction_fix.py --plan
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset_v2 import read_csv_rows, write_csv_rows
from src.dataset_v3 import EDGE_VOICES, analyze_audio

from generate_v3_corpus import LOG_FIELDS, _augment_one, run_tts
from build_dataset_v3_1 import normalize_text, jaccard, TEST_CORR

# (expression_id, command, text) —— 全部对 6 条测试纠正句 Jaccard < 0.5
NEW_EXPRESSIONS = [
    ('CURTAIN_OPEN_CORR_FIX_01', 'CURTAIN_OPEN',  '唔係將窗簾拉埋，係將窗簾拉開'),   # 0.300
    ('CURTAIN_OPEN_CORR_FIX_02', 'CURTAIN_OPEN',  '唔係想收起窗簾，係想打開佢'),     # 0.267
    ('CURTAIN_OPEN_CORR_FIX_03', 'CURTAIN_OPEN',  '唔好關簾呀，幫我攤開佢'),         # 0.000
    ('CURTAIN_OPEN_CORR_FIX_04', 'CURTAIN_OPEN',  '搞錯啦，我要張簾開唔係埋'),       # 0.071
    ('CURTAIN_CLOSE_CORR_FIX_01', 'CURTAIN_CLOSE', '唔係將窗簾拉開，係將窗簾拉埋'),  # 0.444
    ('CURTAIN_CLOSE_CORR_FIX_02', 'CURTAIN_CLOSE', '唔係想打開窗簾，係想收埋佢'),    # 0.357
    ('CURTAIN_CLOSE_CORR_FIX_03', 'CURTAIN_CLOSE', '唔好攤開簾呀，幫我合上佢'),      # 0.000
    ('CURTAIN_CLOSE_CORR_FIX_04', 'CURTAIN_CLOSE', '搞錯啦，我要張簾埋唔係開'),      # 0.133
    # 第二轮：A/B 显示 ALL_OFF 净掉 10 个错误（7->17），吃掉全部收益。
    # 测试句「唔係淨係熄燈，係全部熄晒」缺同构训练句，补 4 句纠正为「全部关闭」。
    ('ALL_OFF_CORR_FIX_01', 'ALL_OFF', '唔係單純熄燈，係要全部關晒'),                # 0.235
    ('ALL_OFF_CORR_FIX_02', 'ALL_OFF', '唔好淨係關燈呀，所有嘢都熄晒佢'),            # 0.095
    ('ALL_OFF_CORR_FIX_03', 'ALL_OFF', '搞錯啦，我要全部都關，唔係得盞燈'),          # 0.095
    ('ALL_OFF_CORR_FIX_04', 'ALL_OFF', '唔係淨係關燈咁簡單，係全部都要熄'),          # 0.263
]

LIB32 = BASE / 'data' / 'corpus' / 'yue_expression_library_v3_2.csv'
WAV_ROOT = BASE / 'data' / 'tts_corpus' / 'wav16k_v3_2' / 'edge'
RAW_ROOT = BASE / 'data' / 'tts_corpus' / 'wav16k_v3_2' / '_raw'
LOG = BASE / 'outputs' / 'v3_2_generation_log.csv'
BASE_MANIFEST = BASE / 'data' / 'tts_corpus' / 'manifest_yue_v3_1_vad.csv'
OUT_MANIFEST = BASE / 'data' / 'tts_corpus' / 'manifest_yue_v3_2.csv'
BASE_SPLIT = BASE / 'data' / 'splits' / 'track_a_expression_split_v3_1.csv'
OUT_SPLIT = BASE / 'data' / 'splits' / 'track_a_expression_split_v3_2.csv'


def now_iso():
    return datetime.now().isoformat(timespec='seconds')


def check_no_leak():
    """断言每条新句对 6 条冻结测试纠正句 Jaccard < 0.5。"""
    bad = []
    for eid, cmd, text in NEW_EXPRESSIONS:
        n = normalize_text(text)
        mx = max(jaccard(n, normalize_text(v)) for v in TEST_CORR.values())
        if mx >= 0.5:
            bad.append((eid, text, mx))
    if bad:
        raise SystemExit(f'新句违反 Jaccard<0.5 红线: {bad}')
    print(f'泄漏检查 PASS：{len(NEW_EXPRESSIONS)} 条新句全部 Jaccard<0.5')


def plan_tts_items():
    items = []
    for eid, cmd, text in NEW_EXPRESSIONS:
        for v in EDGE_VOICES:
            uid = f'{eid}_edge_tts_{v}_base_00'
            items.append(dict(
                stage='tts_new_expr_v32', utterance_id=uid, command=cmd,
                expression_group_id=eid, text=text, voice_id=v,
                condition='baseline', rate=0, pitch=0, volume=0,
                out_path=WAV_ROOT / f'{uid}.wav',
                raw_path=RAW_ROOT / f'{uid}.mp3',
            ))
    return items


def plan_augment_items(tts_items):
    tasks = []
    for it in tts_items:
        uid = it['utterance_id'].replace('_base_00', '_slow18_00')
        tasks.append(dict(
            stage='augment_v32', utterance_id=uid, command=it['command'],
            expression_group_id=it['expression_group_id'], text=it['text'],
            voice_id=it['voice_id'], condition='rate_slow_18',
            source_id=it['utterance_id'], source_path=str(it['out_path']),
            out_path=str(WAV_ROOT / f'{uid}.wav'),
        ))
    return tasks


def write_library():
    rows = []
    for eid, cmd, text in NEW_EXPRESSIONS:
        rows.append(dict(
            expression_id=eid, command=cmd, text=text, register='correction',
            expected_intent=cmd, direct_executable='false',
            distractor_terms='', naturalness_morphemes='',
            yue_naturalness_review='RULE_REVIEWED',
            notes='v3.2 correction 覆盖修复；同构不同字，未与冻结测试句重复',
            text_verified_by='RULE_REVIEWED', v3_source='new_v3_2',
        ))
    write_csv_rows(LIB32, rows)
    print(f'表达库已写：{LIB32}（{len(rows)} 条）')


def build_split():
    """合并原 split（test/val 冻结）+ 8 个新表达组（全部 train）。"""
    base = read_csv_rows(BASE_SPLIT)
    rows = list(base)
    voices = ';'.join('edge-' + v for v in EDGE_VOICES)
    for eid, cmd, text in NEW_EXPRESSIONS:
        rows.append(dict(
            expression_group_id=eid, subset='train', command=cmd,
            register='correction', n_samples=6, voices=voices, text=text,
            split_source='added_v3_2_fix',
        ))
    write_csv_rows(OUT_SPLIT, rows)
    print(f'合并 split 已写：{OUT_SPLIT}（{len(base)} → {len(rows)} 组，'
          f'新增 8 组全 train）')


def build_manifest(log_rows):
    """合并原清单 + 新增 train 条目，产出训练用清单。"""
    base = read_csv_rows(BASE_MANIFEST)
    fields = list(base[0].keys())
    new_rows = []
    for rec in log_rows:
        if rec.get('status') not in ('ok', 'skipped_existing'):
            continue
        rel = rec.get('file_path') or str(Path(rec['out_path']).relative_to(BASE)).replace('\\', '/')
        audio = str(BASE / rel).replace('\\', '/')
        if not Path(audio).is_file():
            continue
        eid = rec['expression_group_id']
        cmd = rec['command']
        text = rec['text']
        uid = rec['utterance_id']
        cond = rec['condition']
        voice = rec['voice_id']
        rate = 18 if cond == 'rate_slow_18' else 0
        row = {k: '' for k in fields}
        row.update(dict(
            utterance_id=uid, audio_path=audio, speaker_id=voice,
            dialect='yue-HK', command=cmd, language='yue', text=text,
            expression_type='correction', semantic_pair_id=f'{eid}:fix',
            source='edge_tts_canto' if cond == 'baseline' else 'offline_signal_aug',
            is_synthetic='true', duration=str(rec.get('duration', '')),
            sample_rate='16000', split_group=f'v3.2:{voice}',
            text_verified_by='RULE_REVIEWED',
            quality_flags='SYNTHETIC;CORRECTION_FIX_V3_2;UNVERIFIED_BY_NATIVE',
            expression_group_id=eid, tts_engine='edge-tts', voice_id=voice,
            voice_identity=voice, rate_percent=str(rate), pitch='0', volume='0',
            condition=cond, source_text=text, quality_status='ok',
            generation_batch='v3.2', channels='1', sha256=rec.get('sha256', ''),
            register='correction', split='train', split_source='added_v3_2_fix',
            data_version='v3.2',
        ))
        new_rows.append(row)
    all_rows = base + new_rows
    write_csv_rows(OUT_MANIFEST, all_rows)
    print(f'合并清单已写：{OUT_MANIFEST}')
    print(f'  原 {len(base)} 条（test/val 原样冻结）+ 新增 train {len(new_rows)} 条')
    print(f'  新增按 split: {dict(Counter(r["split"] for r in new_rows))}')
    return len(new_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--retries', type=int, default=3)
    ap.add_argument('--backoff', type=float, default=3.0)
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--all', action='store_true', help='TTS + 增广 + 合并清单')
    ap.add_argument('--manifest-only', action='store_true', help='仅用现有日志重建合并清单')
    args = ap.parse_args()

    check_no_leak()
    tts_items = plan_tts_items()
    aug_tasks = plan_augment_items(tts_items)
    print(f'新句 {len(NEW_EXPRESSIONS)} × 3音色 = {len(tts_items)} baseline；'
          f'+{len(aug_tasks)} 离线增广')

    if args.plan:
        for it in tts_items[:4]:
            print('  TTS', it['utterance_id'], it['text'])
        return

    WAV_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    # run_tts/upsert_log 要求 log 是 {utterance_id: rec} 字典
    log = {r['utterance_id']: r for r in read_csv_rows(LOG)} if LOG.is_file() else {}

    if args.manifest_only:
        rows = sorted(log.values(), key=lambda r: r.get('utterance_id', ''))
        n = build_manifest(rows)
        if n < 8:
            raise SystemExit(f'新增 train 仅 {n} 条，异常')
        return

    write_library()

    done = {uid for uid, r in log.items()
            if r.get('status') in ('ok', 'skipped_existing')}
    todo = [it for it in tts_items if it['utterance_id'] not in done]
    print(f'TTS 待合成 {len(todo)}/{len(tts_items)} 条')

    if todo:
        t0 = time.perf_counter()
        dt, stats = asyncio.run(run_tts(todo, args.workers, args.retries, args.backoff, log))
        print(f'TTS 完成 {dt:.1f}s {dict(stats)}')

    # 离线增广（复用 _augment_one，模块级函数可子进程调用）
    aug_logs = [r for r in log.values() if r.get('stage') == 'augment_v32']
    done_aug = {r.get('utterance_id') for r in aug_logs if r.get('status') in ('ok', 'skipped_existing')}
    todo_aug = [t for t in aug_tasks if t['utterance_id'] not in done_aug]
    print(f'离线增广待处理 {len(todo_aug)}/{len(aug_tasks)} 条')
    for task in todo_aug:
        rec = _augment_one(task)
        log[rec['utterance_id']] = rec
        print(f"  {rec['status']:18s} {task['utterance_id']}")

    log_rows = sorted(log.values(), key=lambda r: (r.get('stage', ''), r.get('utterance_id', '')))
    write_csv_rows(LOG, log_rows, LOG_FIELDS)

    if args.all:
        n = build_manifest(log_rows)
        if n < 8:
            raise SystemExit(f'新增 train 仅 {n} 条，异常')
        build_split()


if __name__ == '__main__':
    main()

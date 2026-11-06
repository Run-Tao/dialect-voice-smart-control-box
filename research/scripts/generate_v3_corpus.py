# -*- coding: utf-8 -*-
"""第三阶段：v3 语料生成器（真实 edge-tts + 离线信号级增广）。

背景与回退
----------
任务原计划：174 表达组 × 3 音色 × 4 条件（rate±18%、pitch±15Hz）≈ 2088 条
全部用 edge-tts 生成。实测本机网络（2026-10-21）：
  - 单条成功请求延迟约 7–35s；
  - 并发 6 时失败率 41.7%（24 条中 10 条 NoAudioReceived）；
  - 并发 1 时失败率仍 33.3%（9 条中 3 条失败）；
  - 有效吞吐约 0.04–0.12 条/秒。
失败率远超任务允许的 5% 阈值，故按任务第 3 条要求**回退为离线信号级增广**：
对 v2 同表达基准音频（语速 0%、音高 0、同一音色）做确定性 WSOLA 变速 /
重采样变调，得到 v3 的 4 个条件变体。

真实 edge-tts 仅用于「没有 v2 音源、无法离线增广」的文本：
  - 18 条新表达 × 3 音色 baseline（54 条）
  - 24 条未知/易混文本 × 3 音色 baseline（72 条，见 generate_unknown_v3.py）
  - 42 个 (表达组, WanLung) 组合在 v2 中缺音源，需先生成 baseline 才能增广（42 条）

生成阶段
--------
  plan     只打印计划与统计
  probe    小规模探针：生成 1 条 rate_slow_18 + 1 条 pitch_high，
           校验 16k 单声道 PCM16、时长/能量正常、与 v2 源 sha256 不同
  tts      edge-tts 真实生成（需 .venv_tts 的 python）
  augment  离线信号级增广（纯 numpy，主环境 python 亦可）
  all      tts -> augment

幂等/续跑：输出已存在且校验通过则跳过；失败单条重试 3 次不中断整体；
日志追加/覆盖写入 outputs/v3_generation_log.csv，重跑不会丢历史。

用法::

    .\\.venv_tts\\Scripts\\python.exe scripts/generate_v3_corpus.py --mode plan
    .\\.venv_tts\\Scripts\\python.exe scripts/generate_v3_corpus.py --mode probe
    .\\.venv_tts\\Scripts\\python.exe scripts/generate_v3_corpus.py --mode all --workers 5
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.dataset_v2 import read_csv_rows, sha256_file, write_csv_rows  # noqa: E402
from src.dataset_v3 import (  # noqa: E402
    EDGE_VOICES, SAMPLE_RATE, V3_CONDITIONS, V3_CONDITION_ORDER, V3_LIBRARY,
    V3_LOG, V3_RAW_ROOT, V3_WAV_ROOT, V2_MANIFEST, analyze_audio,
    augment_condition, build_baseline_source_index, format_detail, load_wav_16k,
    quality_flags_for, resample_sinc, wav_pcm16_ok, write_wav_pcm16,
)

LOG_FIELDS = [
    'stage', 'utterance_id', 'command', 'expression_group_id', 'text', 'voice_id',
    'condition', 'generation_method', 'status', 'attempts', 'duration',
    'sample_rate', 'channels', 'rms', 'sha256', 'file_path', 'augment_from',
    'condition_detail', 'error', 'elapsed_sec', 'generated_at',
]

PROBE_DIR = BASE / 'data' / 'tts_corpus' / 'wav16k_v3' / 'probe'
PROBE_REPORT = BASE / 'data' / 'tts_corpus' / 'wav16k_v3' / 'probe_report.json'

# 条件 -> 文件名 tag（与 v2 的 v0/slow/fast 区分开，避免路径冲突）
CONDITION_TAG = {
    'rate_slow_18': 'slow18',
    'rate_fast_18': 'fast18',
    'pitch_low': 'plow15',
    'pitch_high': 'phigh15',
    'baseline': 'base',
    'src_baseline': 'srcbase',
}

VOICE_TAG = {
    'zh-HK-HiuGaaiNeural': 'zh-HK-HiuGaaiNeural',
    'zh-HK-HiuMaanNeural': 'zh-HK-HiuMaanNeural',
    'zh-HK-WanLungNeural': 'zh-HK-WanLungNeural',
}


# ---------------------------------------------------------------------------
# 生成日志（读-改-写，按 utterance_id 覆盖，保证可续跑）
# ---------------------------------------------------------------------------

def load_log():
    if not V3_LOG.exists():
        return {}
    rows = read_csv_rows(V3_LOG)
    return {r['utterance_id']: r for r in rows if r.get('utterance_id')}


def save_log(log):
    rows = sorted(log.values(), key=lambda r: (r.get('stage', ''), r['utterance_id']))
    write_csv_rows(V3_LOG, rows, LOG_FIELDS)
    return len(rows)


def upsert_log(log, rec):
    log[rec['utterance_id']] = {k: rec.get(k, '') for k in LOG_FIELDS}


def now_iso():
    return datetime.now().isoformat(timespec='seconds')


# ---------------------------------------------------------------------------
# 输入读取
# ---------------------------------------------------------------------------

def load_v2_rows():
    return read_csv_rows(V2_MANIFEST)


def load_v3_library():
    return read_csv_rows(V3_LIBRARY)


def new_expression_rows():
    return [r for r in load_v3_library() if r.get('v3_source') == 'new_v3']


def plan_tts_items():
    """需要真实 edge-tts 的条目（baseline 类）。

    1) tts_new_expr：18 新表达 × 3 音色 baseline
    2) tts_source：v2 音源缺口（WanLung 42 组）的 baseline，仅作离线增广源
    """
    v2 = load_v2_rows()
    usable = [r for r in v2 if r.get('quality_status') != 'excluded']
    index = build_baseline_source_index(usable)
    groups = sorted({r['expression_group_id'] for r in usable})
    cmd_of = {r['expression_group_id']: r['command'] for r in usable}
    text_of = {}
    seen = set()
    for r in usable:
        key = (r['expression_group_id'], r['voice_id'])
        if key in seen:
            continue
        seen.add(key)
        text_of[key] = r['text']

    items = []
    missing = [(g, v) for g in groups for v in EDGE_VOICES if (g, v) not in index]
    for g, v in missing:
        uid = f'{g}_edge_tts_{v}_srcbase_00'
        text = text_of.get((g, v))
        if text is None:
            # 该 (组, 音色) 在 v2 中完全没有音频：从同组其他音色取文本
            text = next(r['text'] for r in usable if r['expression_group_id'] == g)
        items.append(dict(
            stage='tts_source', utterance_id=uid, command=cmd_of[g],
            expression_group_id=g, text=text, voice_id=v,
            condition='src_baseline', rate=0, pitch=0, volume=0,
            out_path=V3_WAV_ROOT / f'{uid}.wav',
            raw_path=V3_RAW_ROOT / 'edge' / f'{uid}.mp3',
        ))

    for r in new_expression_rows():
        for v in EDGE_VOICES:
            uid = f"{r['expression_id']}_edge_tts_{v}_base_00"
            items.append(dict(
                stage='tts_new_expr', utterance_id=uid, command=r['command'],
                expression_group_id=r['expression_id'], text=r['text'],
                voice_id=v, condition='baseline', rate=0, pitch=0, volume=0,
                out_path=V3_WAV_ROOT / f'{uid}.wav',
                raw_path=V3_RAW_ROOT / 'edge' / f'{uid}.mp3',
            ))
    return items, index, missing


def plan_augment_items(log):
    """离线增广条目：

    1) v2 全部 (表达组, 音色) 的 baseline 源 -> 4 个 v3 条件（522 源 × 4 = 2088）
    2) 新表达的真实 baseline -> rate_slow_18（54 条）
    """
    v2 = [r for r in load_v2_rows() if r.get('quality_status') != 'excluded']
    index = build_baseline_source_index(v2)

    def source_path(row):
        p = Path(row['audio_path'])
        return p if p.is_absolute() else (BASE / 'data' / 'tts_corpus' / p)

    augmented_sources = {}
    for (g, v), row in index.items():
        augmented_sources[(g, v)] = dict(
            source_id=row['utterance_id'], source_path=str(source_path(row)),
            group=g, voice=v, command=row['command'], text=row['text'],
            group_kind='v2_expression')

    # 缺口 baseline 来自本轮 tts_source 日志
    for uid, r in log.items():
        if r.get('stage') != 'tts_source' or r.get('status') not in ('ok', 'skipped_existing'):
            continue
        key = (r['expression_group_id'], r['voice_id'])
        if key in augmented_sources:
            continue
        augmented_sources[key] = dict(
            source_id=uid, source_path=str(V3_WAV_ROOT / f'{uid}.wav'),
            group=r['expression_group_id'], voice=r['voice_id'],
            command=r['command'], text=r['text'], group_kind='v2_expression')

    tasks = []
    for (g, v), src in sorted(augmented_sources.items()):
        for cond in V3_CONDITION_ORDER:
            uid = f"{g}_edge_tts_{v}_{CONDITION_TAG[cond]}_00"
            tasks.append(dict(
                stage='augment', utterance_id=uid, command=src['command'],
                expression_group_id=g, text=src['text'], voice_id=v,
                condition=cond, source_id=src['source_id'],
                source_path=src['source_path'], group_kind=src['group_kind'],
                out_path=str(V3_WAV_ROOT / f'{uid}.wav')))

    # 新表达 baseline -> 离线变慢 18%
    for uid, r in sorted(log.items()):
        if r.get('stage') != 'tts_new_expr' or r.get('status') not in ('ok', 'skipped_existing'):
            continue
        g = r['expression_group_id']
        v = r['voice_id']
        cond = 'rate_slow_18'
        new_uid = f"{g}_edge_tts_{v}_{CONDITION_TAG[cond]}_00"
        tasks.append(dict(
            stage='augment_new_expr', utterance_id=new_uid, command=r['command'],
            expression_group_id=g, text=r['text'], voice_id=v,
            condition=cond, source_id=uid,
            source_path=str(V3_WAV_ROOT / f'{uid}.wav'),
            group_kind='new_expression',
            out_path=str(V3_WAV_ROOT / f'{new_uid}.wav')))
    return tasks, augmented_sources


# ---------------------------------------------------------------------------
# 音频格式转换（mp3 -> 16k 单声道 PCM16）
# ---------------------------------------------------------------------------

def convert_mp3_to_wav16k(src_mp3, dst_wav):
    import soundfile as sf
    audio, sr = sf.read(str(src_mp3), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = resample_sinc(audio, SAMPLE_RATE / float(sr))
    write_wav_pcm16(dst_wav, audio)
    return dst_wav


# ---------------------------------------------------------------------------
# edge-tts 真实生成
# ---------------------------------------------------------------------------

def _tts_settings(rate, pitch, volume):
    r = f'{int(round(rate * 100)):+d}%'
    p = f'{int(round(pitch)):+d}Hz'
    vol = f'{int(round(volume * 100)):+d}%'
    return r, p, vol


async def _tts_attempt(item, timeout=90):
    import edge_tts
    rate, pitch, volume = _tts_settings(item['rate'], item['pitch'], item['volume'])
    item['raw_path'].parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = item['raw_path'].with_name(item['raw_path'].name + '.part')
    try:
        await asyncio.wait_for(
            edge_tts.Communicate(item['text'], item['voice_id'],
                                 rate=rate, pitch=pitch, volume=volume).save(str(tmp_mp3)),
            timeout=timeout)
        if tmp_mp3.stat().st_size < 256:
            raise RuntimeError(f'mp3 too small: {tmp_mp3.stat().st_size}')
        os.replace(str(tmp_mp3), str(item['raw_path']))
    finally:
        if tmp_mp3.exists():
            try:
                tmp_mp3.unlink()
            except OSError:
                pass
    item['out_path'].parent.mkdir(parents=True, exist_ok=True)
    convert_mp3_to_wav16k(item['raw_path'], item['out_path'])


def _existing_ok(item, log):
    """已存在且校验通过则跳过；返回可复用记录或 None。"""
    out = item['out_path']
    if not wav_pcm16_ok(out):
        return None
    prev = log.get(item['utterance_id'])
    if prev and prev.get('status') in ('ok', 'skipped_existing') and prev.get('sha256'):
        if prev['sha256'] == sha256_file(out):
            return prev
    # 文件有效但日志缺失/哈希不符：采纳现有文件（如实记录为 skipped_existing）
    return 'adopt'


def _tts_record(item, status, attempts, elapsed, error=''):
    stats = analyze_audio(item['out_path']) if Path(item['out_path']).exists() else None
    status_q = status
    reasons = ''
    if stats is not None and status in ('ok', 'skipped_existing'):
        q, qr = quality_flags_for(stats)
        if q == 'excluded':
            status_q = 'quality_excluded'
            reasons = qr
        elif q == 'flagged':
            status_q = 'quality_flagged'
            reasons = qr
    return {
        'stage': item['stage'], 'utterance_id': item['utterance_id'],
        'command': item['command'], 'expression_group_id': item['expression_group_id'],
        'text': item['text'], 'voice_id': item['voice_id'],
        'condition': item['condition'], 'generation_method': 'edge_tts',
        'status': status_q, 'attempts': attempts,
        'duration': round(stats['duration'], 3) if stats else '',
        'sample_rate': stats['sample_rate'] if stats else '',
        'channels': stats['channels'] if stats else '',
        'rms': round(stats['rms'], 6) if stats else '',
        'sha256': sha256_file(item['out_path']) if Path(item['out_path']).exists() else '',
        'file_path': str(Path(item['out_path']).relative_to(BASE)).replace('\\', '/'),
        'augment_from': '', 'condition_detail': f'edge_tts rate={item["rate"]} pitch={item["pitch"]}',
        'error': (error + (';' + reasons if reasons else ''))[:500],
        'elapsed_sec': round(elapsed, 3), 'generated_at': now_iso(),
    }


async def _run_one(item, sem, log, retries, backoff):
    async with sem:
        t0 = time.perf_counter()
        ex = _existing_ok(item, log)
        if ex == 'adopt':
            rec = _tts_record(item, 'skipped_existing', 0, time.perf_counter() - t0)
            upsert_log(log, rec)
            return rec
        if isinstance(ex, dict):
            rec = _tts_record(item, 'skipped_existing', 0, time.perf_counter() - t0)
            rec['sha256'] = ex.get('sha256') or rec['sha256']
            upsert_log(log, rec)
            return rec
        last_err = ''
        for attempt in range(1, retries + 1):
            try:
                await _tts_attempt(item)
                rec = _tts_record(item, 'ok', attempt, time.perf_counter() - t0)
                upsert_log(log, rec)
                return rec
            except Exception as exc:  # 单条失败不中断整体
                last_err = f'{type(exc).__name__}: {repr(exc)[:300]}'
                if attempt < retries:
                    await asyncio.sleep(backoff * attempt)
        rec = _tts_record(item, 'fail', retries, time.perf_counter() - t0, last_err)
        upsert_log(log, rec)
        return rec


async def run_tts(items, workers, retries, backoff, log):
    workers = max(1, min(int(workers), 6))
    sem = asyncio.Semaphore(workers)
    done = Counter()

    async def wrapped(i, item):
        rec = await _run_one(item, sem, log, retries, backoff)
        done[rec['status']] += 1
        n = sum(done.values())
        if n % 10 == 0 or rec['status'] in ('fail', 'quality_excluded', 'quality_flagged'):
            print(f"  [{n}/{len(items)}] {rec['status']} {rec['utterance_id']} "
                  f"attempts={rec['attempts']} {rec['elapsed_sec']}s "
                  f"{rec.get('error', '')[:80]}", flush=True)

    t0 = time.perf_counter()
    await asyncio.gather(*[wrapped(i, it) for i, it in enumerate(items)])
    dt = time.perf_counter() - t0
    nfail = done.get('fail', 0) + done.get('quality_excluded', 0)
    print(f'edge-tts 阶段完成：{len(items)} 条，耗时 {dt:.1f}s，状态 {dict(done)}，'
          f'失败/排除 {nfail}，失败率 {nfail / max(1, len(items)):.1%}', flush=True)
    return dt, done


# ---------------------------------------------------------------------------
# 离线信号级增广
# ---------------------------------------------------------------------------

def _augment_one(task):
    """单个增广任务（可在子进程运行，必须模块级函数）。"""
    t0 = time.perf_counter()
    out = Path(task['out_path'])
    if wav_pcm16_ok(out):
        stats = analyze_audio(out)
        return {**{k: task[k] for k in ('stage', 'utterance_id', 'command',
                                       'expression_group_id', 'text', 'voice_id',
                                       'condition')},
                'generation_method': 'offline_signal_aug', 'status': 'skipped_existing',
                'attempts': 0, 'duration': round(stats['duration'], 3),
                'sample_rate': stats['sample_rate'], 'channels': stats['channels'],
                'rms': round(stats['rms'], 6), 'sha256': sha256_file(out),
                'file_path': str(out.relative_to(BASE)).replace('\\', '/'),
                'augment_from': task['source_id'], 'condition_detail': '',
                'error': '', 'elapsed_sec': round(time.perf_counter() - t0, 3),
                'generated_at': now_iso()}
    x, sr = load_wav_16k(task['source_path'])
    if sr != SAMPLE_RATE:
        x = resample_sinc(x, SAMPLE_RATE / float(sr))
    y, detail = augment_condition(x, task['condition'], voice=task['voice_id'])
    write_wav_pcm16(out, y)
    stats = analyze_audio(out)
    q, qr = quality_flags_for(stats)
    status = 'ok'
    if q == 'excluded':
        status = 'quality_excluded'
    elif q == 'flagged':
        status = 'quality_flagged'
    return {**{k: task[k] for k in ('stage', 'utterance_id', 'command',
                                   'expression_group_id', 'text', 'voice_id',
                                   'condition')},
            'generation_method': 'offline_signal_aug', 'status': status,
            'attempts': 1, 'duration': round(stats['duration'], 3),
            'sample_rate': stats['sample_rate'], 'channels': stats['channels'],
            'rms': round(stats['rms'], 6), 'sha256': sha256_file(out),
            'file_path': str(out.relative_to(BASE)).replace('\\', '/'),
            'augment_from': task['source_id'],
            'condition_detail': format_detail(detail),
            'error': qr if q != 'ok' else '',
            'elapsed_sec': round(time.perf_counter() - t0, 3),
            'generated_at': now_iso()}


def run_augment(tasks, workers, overwrite, log):
    if overwrite:
        for t in tasks:
            p = Path(t['out_path'])
            if p.exists():
                p.unlink()
    workers = max(1, min(int(workers), 12))
    t0 = time.perf_counter()
    done = Counter()
    if workers == 1:
        results = []
        for i, t in enumerate(tasks):
            r = _augment_one(t)
            results.append(r)
            done[r['status']] += 1
            if (i + 1) % 100 == 0:
                print(f'  [{i + 1}/{len(tasks)}] {dict(done)}', flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_augment_one, tasks, chunksize=8)):
                results.append(r)
                done[r['status']] += 1
                if (i + 1) % 200 == 0:
                    print(f'  [{i + 1}/{len(tasks)}] {dict(done)}', flush=True)
    for r in results:
        upsert_log(log, r)
    dt = time.perf_counter() - t0
    nbad = done.get('quality_excluded', 0) + done.get('fail', 0)
    print(f'离线增广阶段完成：{len(tasks)} 条，耗时 {dt:.1f}s，状态 {dict(done)}，'
          f'排除/失败 {nbad}', flush=True)
    return dt, done


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------

def run_probe(workers):
    """生成 1 条 rate_slow_18 与 1 条 pitch_high，校验格式/时长/能量/sha 差异。"""
    v2 = [r for r in load_v2_rows() if r.get('quality_status') != 'excluded']
    index = build_baseline_source_index(v2)
    # 选第一条基准源（ALL_OFF_01 / HiuGaai），保证 v2 同文本音频存在
    key = sorted(index)[0]
    row = index[key]
    src_path = BASE / 'data' / 'tts_corpus' / row['audio_path']
    x, sr = load_wav_16k(src_path)
    src_sha = sha256_file(src_path)
    src_stats = analyze_audio(src_path)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'generated_at': now_iso(),
        'source_utterance_id': row['utterance_id'],
        'source_audio_path': row['audio_path'],
        'source_sha256': src_sha,
        'source_stats': {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in src_stats.items()},
        'conditions': {},
        'network_note': '真实 edge-tts 基准生成由 tts 阶段执行；本探针校验离线增广产物',
    }
    problems = []
    for cond in ['rate_slow_18', 'pitch_high']:
        t0 = time.perf_counter()
        y, detail = augment_condition(x, cond, voice=row['voice_id'])
        out = PROBE_DIR / f"{row['utterance_id']}_{CONDITION_TAG[cond]}_probe.wav"
        write_wav_pcm16(out, y)
        sha = sha256_file(out)
        stats = analyze_audio(out)
        q, qr = quality_flags_for(stats)
        import wave
        with wave.open(str(out), 'rb') as wf:
            fmt = {'sample_rate': wf.getframerate(), 'channels': wf.getnchannels(),
                   'sampwidth': wf.getsampwidth()}
        expect_ratio = 1.0 / (1.0 + V3_CONDITIONS[cond][1]) if V3_CONDITIONS[cond][0] == 'rate' else 1.0
        ratio = stats['duration'] / src_stats['duration']
        checks = {
            'format_16k_mono_pcm16': (fmt['sample_rate'] == 16000 and fmt['channels'] == 1
                                      and fmt['sampwidth'] == 2),
            'sha_differs_from_v2_source': sha != src_sha,
            'duration_ratio_within_2pct': abs(ratio - expect_ratio) <= 0.02 * expect_ratio,
            'duration_gt_0.35s': stats['duration'] > 0.35,
            'rms_gt_0.002': stats['rms'] > 0.002,
            'quality_not_excluded': q != 'excluded',
        }
        ok = all(checks.values())
        if not ok:
            problems.append({cond: checks})
        report['conditions'][cond] = {
            'ok': ok, 'checks': checks, 'path': str(out.relative_to(BASE)).replace('\\', '/'),
            'sha256': sha, 'format': fmt, 'duration_in': round(src_stats['duration'], 4),
            'duration_out': round(stats['duration'], 4), 'duration_ratio': round(ratio, 6),
            'expected_ratio': round(expect_ratio, 6), 'rms': round(stats['rms'], 6),
            'peak': round(stats['peak'], 6), 'quality': q, 'quality_reasons': qr,
            'detail': detail, 'elapsed_sec': round(time.perf_counter() - t0, 3),
        }
        print(f"probe {cond}: ok={ok} checks={checks} dur_ratio={ratio:.4f} "
              f"rms={stats['rms']:.4f} sha={sha[:12]}", flush=True)
    report['ok'] = not problems
    PROBE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('probe report:', PROBE_REPORT)
    if problems:
        raise SystemExit(f'探针校验失败: {problems}')
    return report


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def mode_plan():
    items, index, missing = plan_tts_items()
    print(f'v2 基准源（可离线增广的 (组,音色)）: {len(index)}')
    print(f'v2 音源缺口: {len(missing)} -> 需真实 TTS baseline')
    by_stage = Counter(i['stage'] for i in items)
    print(f'TTS 条目: {len(items)} {dict(by_stage)}')
    # 预估增广条目数
    fake_log = {i['utterance_id']: {'stage': i['stage'], 'status': 'ok',
                                    'expression_group_id': i['expression_group_id'],
                                    'voice_id': i['voice_id'], 'command': i['command'],
                                    'text': i['text']} for i in items}
    tasks, sources = plan_augment_items(fake_log)
    print(f'离线增广源: {len(sources)} (组×音色)')
    print(f'增广条目: {len(tasks)} {dict(Counter(t["stage"] for t in tasks))}')


def mode_tts(args, log):
    items, index, missing = plan_tts_items()
    if args.limit:
        items = items[:args.limit]
    print(f'TTS 计划 {len(items)} 条（缺口 {len(missing)} 组音色 + 新表达）', flush=True)
    dt, done = asyncio.run(run_tts(items, args.workers, args.retries,
                                   args.backoff, log))
    save_log(log)
    print(f'日志已写：{V3_LOG}，累计 {len(log)} 条', flush=True)
    return dt, done


def mode_augment(args, log):
    tasks, sources = plan_augment_items(log)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f'增广计划 {len(tasks)} 条，源 {len(sources)} 个', flush=True)
    dt, done = run_augment(tasks, args.workers, args.overwrite, log)
    save_log(log)
    print(f'日志已写：{V3_LOG}，累计 {len(log)} 条', flush=True)
    return dt, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='plan',
                    choices=['plan', 'probe', 'tts', 'augment', 'all'])
    ap.add_argument('--workers', type=int, default=5,
                    help='并发数（TTS 阶段上限 6，增广阶段上限 12）')
    ap.add_argument('--retries', type=int, default=3, help='单条 TTS 重试次数')
    ap.add_argument('--backoff', type=float, default=3.0, help='重试退避基数（秒）')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    V3_WAV_ROOT.mkdir(parents=True, exist_ok=True)
    V3_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    (V3_RAW_ROOT / 'edge').mkdir(parents=True, exist_ok=True)

    if args.mode == 'plan':
        mode_plan()
        return
    if args.mode == 'probe':
        run_probe(args.workers)
        return

    log = load_log()
    t0 = time.perf_counter()
    if args.mode in ('tts', 'all'):
        mode_tts(args, log)
    if args.mode in ('augment', 'all'):
        mode_augment(args, log)
    print(f'总耗时 {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()

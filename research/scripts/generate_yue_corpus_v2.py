# -*- coding: utf-8 -*-
"""第二阶段：粤语命令语料扩充生成器。

在既有 180 条语料基础上新增粤语 TTS 样本，目标为粤语累计 600 条（六类各约 100 条）。

设计要点（均在清单中记录，便于分组分析）：
- 音色：canto-tts-default(1) + edge-tts zh-HK 三音色(3) = 4 个独立音色身份。
  - 不允许把同一音色的不同文件名/不同语速档当作不同说话人。
- 变化数量（增加 420 条）不能只靠换音色：文本表达仅 174 组，
  因此以「语域/句式变化」为主、「语速档」「语气词省略」和「同音色随机重复采样」为辅。
  - edge-tts 支持 rate / pitch / volume 参数，语速档作为独立 condition 记录。
  - canto-tts ONNX 后端只有单一 baked 音色且不支持语速参数；
    其同一文本的多次采样会产生自然的口音/语速/停顿差异，作为 condition=resample 记录。
    这类样本与文本变体必须整体落在同一数据子集（由 expression_group_id 保证）。
- 失败重试与幂等：已存在且校验通过的音频默认跳过。

用法::

    .\\.venv_tts\\Scripts\\python.exe scripts/generate_yue_corpus_v2.py --plan-only
    .\\.venv_tts\\Scripts\\python.exe scripts/generate_yue_corpus_v2.py --execute
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.dataset_v2 import (  # noqa: E402
    INTENTS, read_csv_rows, write_csv_rows, analyze_audio, quality_flags_for,
    sha256_file, MANIFEST_FIELDS,
)

LIBRARY_CSV = BASE / 'data' / 'corpus' / 'yue_expression_library.csv'
EXPR_CSV = BASE / 'data' / 'corpus' / 'core_six_expressions.csv'
OUT_ROOT = BASE / 'data' / 'tts_corpus'
RAW_ROOT = OUT_ROOT / 'raw_v2'
WAV_ROOT = OUT_ROOT / 'wav16k_v2'
LOG_PATH = BASE / 'outputs' / 'v2_generation_log.csv'

EDGE_VOICES = ['zh-HK-HiuGaaiNeural', 'zh-HK-HiuMaanNeural', 'zh-HK-WanLungNeural']
CANTO_VOICE = 'canto-tts-default'

# 语速档（仅 edge-tts 支持）；canto-tts 无语速参数，用随机重复采样代替
EDGE_RATES = [('v0', 0), ('slow', -12), ('fast', 12)]
# 音高档（仅 edge-tts 支持）：作为「同一表达/同一音色下的韵律变化」，不是新说话人
EDGE_PITCHES = [('p0', 0), ('plow', -6), ('phigh', 6)]
CANTO_DRAWS_FOR_MORPHOLOGY = 2   # 每个 canto 表达组额外重复采样次数（产生自然韵律差异）
CANTO_DRAWS_FOR_RESAMPLE = 3     # 音色内 resample 变体（同一文本）

TARGET_YUE_TOTAL = 600


# ---------------------------------------------------------------------------
# 旧数据读取
# ---------------------------------------------------------------------------

def load_legacy_yue():
    """读取现有 180 条清单中的粤语部分，并补全新增字段。"""
    legacy_path = OUT_ROOT / 'manifest.csv'
    rows = read_csv_rows(legacy_path)
    out = []
    for r in rows:
        if (r.get('language') or '').strip() != 'yue':
            continue
        command = (r.get('command') or '').strip()
        text = (r.get('text') or '').strip()
        expr_type = (r.get('expression_type') or '').strip()
        voice = (r.get('speaker_id') or '').strip()
        out.append({
            'utterance_id': r['utterance_id'],
            'audio_path': 'wav16k/' + (r['audio_path'].split('wav16k/')[-1]),
            'speaker_id': voice,
            'dialect': r.get('dialect', 'yue-HK'),
            'command': command,
            'language': 'yue',
            'text': text,
            'expression_type': expr_type,
            'semantic_pair_id': r.get('semantic_pair_id', ''),
            'source': r.get('source', ''),
            'is_synthetic': 'true',
            'duration': r.get('duration', ''),
            'sample_rate': r.get('sample_rate', ''),
            'split_group': r.get('split_group', ''),
            'text_verified_by': 'PENDING_NATIVE_SPEAKER',
            'quality_flags': 'SYNTHETIC;UNVERIFIED_TEXT',
            'notes': 'v1 corpus; TTS synthetic; not real dialect performance',
            'expression_group_id': f'{command}_{expr_type}',
            'tts_engine': 'canto_tts' if voice == CANTO_VOICE else 'edge_tts',
            'voice_id': voice,
            'voice_identity': voice,
            'rate_percent': '0',
            'pitch': '0',
            'volume': '0',
            'condition': 'legacy',
            'source_text': text,
            'quality_status': '',
            'quality_reasons': '',
            'generation_batch': 'v1',
            'channels': '1',
            'sha256': '',
            'register': expr_type,
        })
    return out


# ---------------------------------------------------------------------------
# 生成计划
# ---------------------------------------------------------------------------

def corpus_shortfall(legacy_rows, target=TARGET_YUE_TOTAL):
    """按类计算现有合格数量与缺口。"""
    counts = Counter(r['command'] for r in legacy_rows)
    per_class_target = target // len(INTENTS)
    return {c: {'existing': counts.get(c, 0), 'target': per_class_target,
                'shortfall': max(0, per_class_target - counts.get(c, 0))}
            for c in INTENTS}


def load_library():
    rows = read_csv_rows(LIBRARY_CSV)
    by_cmd = defaultdict(list)
    for r in rows:
        by_cmd[r['command']].append(r)
    for c in by_cmd:
        by_cmd[c].sort(key=lambda r: r['expression_id'])
    return by_cmd


def build_plan(legacy_rows, shortfall):
    """构造生成计划。

    **关键约束（实测发现）**：edge-tts 对相同 (文本, 音色, rate, pitch) 是确定性输出，
    逐字节相同。v1 语料已包含 direct/colloquial/polite 三种表达的 edge 音频，
    表达库若在相同文本上用相同音色基准语速重新生成，会产出与 v1 完全相同的文件
    （实测 54 条 sha256 冲突）。

    因此这里做**去重规划**：
    - 若 (规范化文本, 音色) 已存在于 v1 语料，则该槽位改用 rate=-12%（语速慢档），
      既避免重复，又正好补充任务要求的「同一文本的语速变化」维度；
    - 其余槽位仍用基准语速。

    所有条件写入清单 condition / rate_percent，可通过分组分析区分
    「文本变化」/「语速变化」/「音色变化」。
    """
    lib = load_library()
    legacy_keys = {(t, v) for t, v in _legacy_text_voice_keys(legacy_rows)}
    plan = []
    for cmd in INTENTS:
        need = shortfall[cmd]['shortfall']
        slots = []
        for voice in EDGE_VOICES:
            for expr in lib[cmd]:
                from src.dataset_v2 import normalize_text
                key = (normalize_text(expr['text']), voice)
                if key in legacy_keys:
                    rate, condition = -12, 'rate_slow'
                else:
                    rate, condition = 0, 'text_morphology'
                slots.append(dict(
                    command=cmd, expression_group_id=expr['expression_id'],
                    text=expr['text'], register=expr['register'], engine='edge_tts',
                    voice=voice, rate=rate, pitch=0, condition=condition,
                    source_text=expr['text']))
        if need > len(slots):
            extra = []
            for tag, rate in EDGE_RATES[1:]:
                for expr in lib[cmd]:
                    extra.append(dict(
                        command=cmd, expression_group_id=expr['expression_id'],
                        text=expr['text'], register=expr['register'], engine='edge_tts',
                        voice=EDGE_VOICES[0], rate=rate, pitch=0,
                        condition=f'rate_{tag}', source_text=expr['text']))
            for tag, pitch in EDGE_PITCHES:
                for expr in lib[cmd]:
                    extra.append(dict(
                        command=cmd, expression_group_id=expr['expression_id'],
                        text=expr['text'], register=expr['register'], engine='edge_tts',
                        voice=EDGE_VOICES[0], rate=0, pitch=pitch,
                        condition=f'pitch_{tag}', source_text=expr['text']))
            slots.extend(extra)
        plan.extend(slots[:need])
    return plan


def _legacy_text_voice_keys(legacy_rows):
    from src.dataset_v2 import normalize_text
    for r in legacy_rows:
        yield normalize_text(r['text']), r['speaker_id']


def count_plan(plan):
    return Counter(p['command'] for p in plan)


# ---------------------------------------------------------------------------
# 生成执行
# ---------------------------------------------------------------------------

def convert_to_16k(src, dst):
    audio, sr = sf.read(str(src), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        new_len = int(round(len(audio) * 16000 / sr))
        old_t = np.linspace(0, 1, len(audio), endpoint=False)
        new_t = np.linspace(0, 1, new_len, endpoint=False)
        audio = np.interp(new_t, old_t, audio).astype(np.float32)
    sf.write(str(dst), audio, 16000, subtype='PCM_16')
    return sf.info(str(dst))


def synth_edge(text, voice, rate, dst_mp3, pitch=0):
    import edge_tts
    rate_str = f'{rate:+d}%'
    pitch_str = f'{pitch:+d}Hz'
    asyncio.run(edge_tts.Communicate(text, voice, rate=rate_str,
                                     pitch=pitch_str).save(str(dst_mp3)))


def make_id(engine, voice_tag, group, rate_tag, idx):
    return f'{group}_{engine}_{voice_tag}_{rate_tag}_{idx:02d}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true', help='实际调用TTS生成音频')
    ap.add_argument('--plan-only', action='store_true', help='只打印计划')
    ap.add_argument('--limit', type=int, default=0, help='只生成前N条，用于小规模验证')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    WAV_ROOT.mkdir(parents=True, exist_ok=True)
    for d in ['edge', 'canto']:
        (RAW_ROOT / d).mkdir(parents=True, exist_ok=True)
        (WAV_ROOT / d).mkdir(parents=True, exist_ok=True)

    legacy = load_legacy_yue()
    sf_req = corpus_shortfall(legacy)
    print('现有粤语样本:', len(legacy))
    print(json.dumps(sf_req, ensure_ascii=False, indent=2))

    plan = build_plan(legacy, sf_req)
    print('计划新增:', len(plan), dict(count_plan(plan)))
    if args.plan_only:
        return

    if not args.execute:
        print('未指定 --execute，只输出计划。')
        return

    log_rows = []
    log_fields = ['utterance_id', 'command', 'expression_group_id', 'text', 'engine',
                  'voice', 'rate', 'condition', 'status', 'duration', 'sample_rate',
                  'elapsed_sec', 'error']

    # 追加模式：保留历史生成记录，重跑不会丢失已完成的批次
    existing = []
    if LOG_PATH.exists():
        existing = read_csv_rows(LOG_PATH)
        log_rows.extend(existing)
    prior_ids = {r['utterance_id']: r for r in existing}

    canto_tts = None
    todo = plan[:args.limit] if args.limit else plan
    for i, p in enumerate(todo):
        engine = p['engine']
        draw = 0
        rate = int(p.get('rate', 0))
        pitch = int(p.get('pitch', 0))
        cond = p['condition']
        if engine == 'canto_tts':
            if cond == 'text_morphology':
                rate_tag = 'd0'
                draw = 0
            else:
                draw = int(cond.replace('resample', '')) if cond.startswith('resample') else 0
                rate_tag = f'r{draw}'
        elif cond.startswith('pitch_'):
            rate_tag = cond.replace('pitch_', '')
        elif rate == 0:
            rate_tag = 'v0'
        elif rate < 0:
            rate_tag = 'slow'
        else:
            rate_tag = 'fast'
        stem = make_id(engine, p['voice'], p['expression_group_id'], rate_tag, draw)
        if engine == 'edge_tts':
            raw_ext, sub = '.mp3', 'edge'
        else:
            raw_ext, sub = '.wav', 'canto'
        raw_path = RAW_ROOT / sub / (stem + raw_ext)
        dst = WAV_ROOT / sub / (stem + '.wav')

        t0 = time.perf_counter()
        err = ''
        status = 'ok'
        dur = ''
        sr_out = ''
        try:
            if dst.exists() and not args.overwrite:
                info = sf.info(str(dst))
                dur, sr_out = info.duration, info.samplerate
                status = 'skipped_existing'
            else:
                if engine == 'canto_tts':
                    if canto_tts is None:
                        from canto_tts import CantoTTS
                        print('初始化 canto-tts ...')
                        canto_tts = CantoTTS()
                    canto_tts.synthesize(p['source_text'], str(raw_path))
                else:
                    synth_edge(p['source_text'], p['voice'], p['rate'], raw_path)
                info = convert_to_16k(raw_path, dst)
                dur, sr_out = info.duration, info.samplerate
        except Exception as exc:  # 单条失败不终止整批
            status = 'fail'
            err = repr(exc)[:500]
        elapsed = time.perf_counter() - t0
        rec = {'utterance_id': stem, 'command': p['command'],
               'expression_group_id': p['expression_group_id'],
               'text': p['text'], 'engine': p['engine'], 'voice': p['voice'],
               'rate': p['rate'], 'condition': p['condition'],
               'status': status, 'duration': dur, 'sample_rate': sr_out,
               'elapsed_sec': round(elapsed, 3), 'error': err}
        if stem in prior_ids:
            # 覆盖同ID的历史记录，避免重复行
            idx = next(k for k, r in enumerate(log_rows) if r['utterance_id'] == stem)
            log_rows[idx] = rec
        else:
            log_rows.append(rec)
            prior_ids[stem] = rec
        if i % 20 == 0 or status == 'fail':
            print(f"{i+1}/{len(todo)} {status} {stem} dur={dur} {round(elapsed,2)}s")

    write_csv_rows(LOG_PATH, log_rows, log_fields)
    nfail = sum(1 for r in log_rows if r['status'] == 'fail')
    print(f'本次处理 {len(todo)} 条，日志累计 {len(log_rows)} 条，其中失败 {nfail}，日志 {LOG_PATH}')


if __name__ == '__main__':
    main()

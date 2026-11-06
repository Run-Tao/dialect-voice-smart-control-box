# -*- coding: utf-8 -*-
"""第三阶段：未知 / 易混指令测试集（v3）生成器。

目的
----
评估「拒识（unknown rejection）」与「误识别风险」：用 24 条**不在六类命令内**、
但语感接近的粤语文本（含硬负例 開冷氣 / 關電視 / 調暗啲燈 / 開窗 等），
每文本 3 个 edge 粤语音色（baseline），共 72 条 16kHz 单声道 PCM16 音频。

划分
----
按文本对半分：对 24 条文本按规范化文本指纹排序，前 12 条 -> calibration，
后 12 条 -> test；同一文本的 3 个音色整体落在同一侧（避免文本级泄漏）。

输出
----
  data/tts_corpus/unknown_v3/<utterance_id>.wav   （音频）
  data/tts_corpus/unknown_v3.csv                  （清单）
  outputs/v3_generation_log.csv                   （与主生成器共用日志，stage=unknown）

用法::

    .\\.venv_tts\\Scripts\\python.exe scripts/generate_unknown_v3.py
    .\\.venv_tts\\Scripts\\python.exe scripts/generate_unknown_v3.py --plan
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset_v2 import (  # noqa: E402
    quality_flags_for, read_csv_rows, sha256_file, write_csv_rows,
)
from src.dataset_v3 import (  # noqa: E402
    EDGE_VOICES, UNKNOWN_MANIFEST, UNKNOWN_ROOT, V2_MANIFEST, V3_LIBRARY,
    V3_RAW_ROOT, V3_WAV_ROOT, analyze_audio, normalize_text, text_fingerprint,
)

from generate_v3_corpus import (  # noqa: E402
    LOG_FIELDS, load_log, now_iso, run_tts, save_log, upsert_log,
)

# 24 条易混/未知文本：均不属于 LIGHT_ON / LIGHT_OFF / CURTAIN_OPEN / CURTAIN_CLOSE /
# ALL_OFF / STATUS_ALL 六类。分层标注：
#   label = hard_negative：复用六类命令的「開/關/拉/熄/調」方向词，或对象就在灯光/窗帘/窗
#           语义场内，仅设备或动作维度不同 -> 最易被误识别为六类之一（22 条）；
#   label = general_unknown：不含设备通断方向词、属另外的调节维度（音量），
#           词面与六类命令距离最远（2 条）。
# negative_type 供拒识实验进一步按设备类型分层报告。
UNKNOWN_TEXTS = [
    ('開冷氣', 'hard_negative', 'device_onoff_aircon'),
    ('關冷氣', 'hard_negative', 'device_onoff_aircon'),
    ('開電視', 'hard_negative', 'device_onoff_tv'),
    ('關電視', 'hard_negative', 'device_onoff_tv'),
    ('開風扇', 'hard_negative', 'device_onoff_fan'),
    ('關風扇', 'hard_negative', 'device_onoff_fan'),
    ('調暗啲燈', 'hard_negative', 'light_brightness'),
    ('調光啲燈', 'hard_negative', 'light_brightness'),
    ('開窗', 'hard_negative', 'window_vs_curtain'),
    ('關窗', 'hard_negative', 'window_vs_curtain'),
    ('開門', 'hard_negative', 'device_onoff_door'),
    ('關門', 'hard_negative', 'device_onoff_door'),
    ('開抽濕機', 'hard_negative', 'device_onoff_dehumidifier'),
    ('關抽濕機', 'hard_negative', 'device_onoff_dehumidifier'),
    ('開熱水爐', 'hard_negative', 'device_onoff_water_heater'),
    ('關熱水爐', 'hard_negative', 'device_onoff_water_heater'),
    ('較大聲啲', 'general_unknown', 'volume_adjust'),
    ('較細聲啲', 'general_unknown', 'volume_adjust'),
    ('開音樂', 'hard_negative', 'device_onoff_music'),
    ('關音樂', 'hard_negative', 'device_onoff_music'),
    ('開抽氣扇', 'hard_negative', 'device_onoff_exhaust_fan'),
    ('關抽氣扇', 'hard_negative', 'device_onoff_exhaust_fan'),
    ('開雪櫃', 'hard_negative', 'device_onoff_fridge'),
    ('關雪櫃', 'hard_negative', 'device_onoff_fridge'),
]

MANIFEST_FIELDS_UNKNOWN = [
    'utterance_id', 'audio_path', 'text', 'text_norm', 'text_group_id',
    'text_split', 'label', 'negative_type', 'voice_id', 'command', 'condition',
    'tts_engine', 'generation_method', 'sha256', 'duration', 'sample_rate',
    'channels', 'quality_status', 'quality_reasons', 'notes',
]


def assign_text_split(texts):
    """按规范化文本 sha256 指纹排序后对半分（确定性，文本级互斥）。"""
    order = sorted(texts, key=lambda t: text_fingerprint(t))
    half = len(order) // 2
    return {t: ('calibration' if i < half else 'test')
            for i, t in enumerate(order)}


def plan_items():
    texts = [t for t, _, _ in UNKNOWN_TEXTS]
    assert len(texts) == 24, len(texts)
    assert len({normalize_text(t) for t in texts}) == 24, '未知集内文本重复'
    split_of = assign_text_split(texts)
    items = []
    for idx, (text, label, neg_type) in enumerate(UNKNOWN_TEXTS):
        gid = f'UNK_V3_{idx + 1:02d}'
        for v in EDGE_VOICES:
            uid = f'{gid}_{v}'
            items.append(dict(
                stage='unknown', utterance_id=uid, command='UNKNOWN',
                expression_group_id=gid, text=text, voice_id=v,
                condition='baseline', rate=0, pitch=0, volume=0,
                out_path=UNKNOWN_ROOT / f'{uid}.wav',
                raw_path=V3_RAW_ROOT / 'unknown' / f'{uid}.mp3',
                text_split=split_of[text], label=label, negative_type=neg_type,
            ))
    return items


def check_disjoint_from_main():
    """未知集文本不得与主语料（v2 清单 + v3 表达库）规范化文本重复。"""
    known = set()
    for r in read_csv_rows(V2_MANIFEST):
        known.add(normalize_text(r['text']))
    for r in read_csv_rows(V3_LIBRARY):
        known.add(normalize_text(r['text']))
    overlap = {t for t, _, _ in UNKNOWN_TEXTS if normalize_text(t) in known}
    if overlap:
        raise ValueError(f'未知集文本与主语料重复: {sorted(overlap)}')
    return len(known)


def write_unknown_manifest(items):
    log = load_log()
    rows = []
    for it in items:
        rec = log.get(it['utterance_id'])
        if not rec or rec.get('status') not in ('ok', 'skipped_existing',
                                                'quality_flagged'):
            continue
        stats = analyze_audio(it['out_path'])
        q, qr = quality_flags_for(stats)
        rows.append({
            'utterance_id': it['utterance_id'],
            'audio_path': str(Path(it['out_path']).relative_to(BASE)).replace('\\', '/'),
            'text': it['text'],
            'text_norm': normalize_text(it['text']),
            'text_group_id': it['expression_group_id'],
            'text_split': it['text_split'],
            'label': it['label'],
            'negative_type': it['negative_type'],
            'voice_id': it['voice_id'],
            'command': 'UNKNOWN',
            'condition': 'baseline',
            'tts_engine': 'edge_tts',
            'generation_method': 'edge_tts',
            'sha256': sha256_file(it['out_path']),
            'duration': round(stats['duration'], 3),
            'sample_rate': stats['sample_rate'],
            'channels': stats['channels'],
            'quality_status': 'ok' if q == 'ok' else q,
            'quality_reasons': qr,
            'notes': 'v3 未知/易混指令测试集；edge-tts baseline；非六类命令，不参与训练',
        })
    rows.sort(key=lambda r: (r['text_group_id'], r['voice_id']))
    write_csv_rows(UNKNOWN_MANIFEST, rows, MANIFEST_FIELDS_UNKNOWN)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--retries', type=int, default=3)
    ap.add_argument('--backoff', type=float, default=3.0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    n_known = check_disjoint_from_main()
    items = plan_items()
    by_split = {}
    for it in items:
        by_split.setdefault(it['text_split'], set()).add(it['text'])
    label_counts = Counter(t[1] for t in UNKNOWN_TEXTS)
    print(f'已知主语料规范化文本 {n_known} 条；未知集 {len(UNKNOWN_TEXTS)} 文本 × '
          f'{len(EDGE_VOICES)} 音色 = {len(items)} 条')
    print('文本划分:', {k: len(v) for k, v in by_split.items()},
          '分层:', dict(label_counts))
    if args.plan:
        for it in items:
            print(it['utterance_id'], it['text'], it['label'], it['negative_type'],
                  it['voice_id'], it['text_split'])
        return

    UNKNOWN_ROOT.mkdir(parents=True, exist_ok=True)
    V3_WAV_ROOT.mkdir(parents=True, exist_ok=True)
    (V3_RAW_ROOT / 'unknown').mkdir(parents=True, exist_ok=True)
    log = load_log()
    t0 = time.perf_counter()
    dt, done = asyncio.run(run_tts(items[:args.limit] if args.limit else items,
                                   args.workers, args.retries, args.backoff, log))
    save_log(log)
    rows = write_unknown_manifest(items)
    print(f'未知集完成：{len(rows)} 条写入 {UNKNOWN_MANIFEST}；'
          f'TTS {dt:.1f}s，状态 {dict(done)}')
    print(f'总耗时 {time.perf_counter() - t0:.1f}s，日志累计 {len(log)} 条')


if __name__ == '__main__':
    main()

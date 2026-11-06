# -*- coding: utf-8 -*-
"""第三阶段：v3 数据工程的公共工具。

模块职责
--------
1. 定义 v3 统一的清单字段、条件定义与路径常量。
2. 实现**离线信号级增广**（变速 / 变调），用于在 TTS 网络不可靠时
   对 v2 同表达音频生成 v3 条件变体：
   - rate_slow_18 / rate_fast_18：WSOLA 时域伸缩（保音高，只改语速）
   - pitch_low / pitch_high：重采样改调 + WSOLA 还原时长（保时长，只改音高）
   所有增广都是确定性的（无随机噪声），同一输入必得同一输出，便于复现与测试。
3. 提供 16kHz 单声道 PCM16 WAV 的读写、f0 估计、质量检查复用等工具。

依赖约束：只依赖 numpy 与标准库，主环境（D:\\anaconda3\\python.exe）与
TTS 隔离环境（.venv_tts）均可导入；不依赖 edge_tts / soundfile / scipy。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from pathlib import Path

import numpy as np

from src.dataset_v2 import (  # noqa: F401  （部分符号再导出给上层脚本）
    INTENTS, MANIFEST_FIELDS, VOICE_IDENTITY, analyze_audio, normalize_text,
    quality_flags_for, read_csv_rows, sha256_file, text_hash, write_csv_rows,
)

BASE = Path(__file__).resolve().parents[1]
OUT_ROOT = BASE / 'data' / 'tts_corpus'
V3_WAV_ROOT = OUT_ROOT / 'wav16k_v3' / 'edge'
V3_RAW_ROOT = OUT_ROOT / 'raw_v3'
V3_LOG = BASE / 'outputs' / 'v3_generation_log.csv'
V2_MANIFEST = OUT_ROOT / 'manifest_yue_v2.csv'
V2_SPLIT = BASE / 'data' / 'splits' / 'track_a_expression_split.csv'
V2_LIBRARY = BASE / 'data' / 'corpus' / 'yue_expression_library.csv'
V3_LIBRARY = BASE / 'data' / 'corpus' / 'yue_expression_library_v3.csv'
V3_MANIFEST = OUT_ROOT / 'manifest_yue_v3.csv'
V3_SPLIT = BASE / 'data' / 'splits' / 'track_a_expression_split_v3.csv'
UNKNOWN_ROOT = OUT_ROOT / 'unknown_v3'
UNKNOWN_MANIFEST = OUT_ROOT / 'unknown_v3.csv'
V3_AUDIT = BASE / 'outputs' / 'v3_data_audit.json'
V3_DOC = BASE / 'docs' / 'data_v3.md'

EDGE_VOICES = ['zh-HK-HiuGaaiNeural', 'zh-HK-HiuMaanNeural', 'zh-HK-WanLungNeural']
SAMPLE_RATE = 16000

# v3 条件定义：name -> (kind, param, 人类可读说明)
V3_CONDITIONS = {
    'rate_slow_18': ('rate', -0.18, '语速 -18%（时长 ×1.2195）'),
    'rate_fast_18': ('rate', +0.18, '语速 +18%（时长 ×0.8475）'),
    'pitch_low': ('pitch', -15.0, '音高 -15Hz（时长不变）'),
    'pitch_high': ('pitch', +15.0, '音高 +15Hz（时长不变）'),
}
V3_CONDITION_ORDER = ['rate_slow_18', 'rate_fast_18', 'pitch_low', 'pitch_high']

# 各音色的默认基频（仅当自动估计失败时使用；HiuGaai/HiuMaan 女声，WanLung 男声）
DEFAULT_F0 = {
    'zh-HK-HiuGaaiNeural': 220.0,
    'zh-HK-HiuMaanNeural': 210.0,
    'zh-HK-WanLungNeural': 125.0,
}

# v2 清单字段 + v2 追加的 register/split/group_alias + v3 追加字段
MANIFEST_V3_FIELDS = MANIFEST_FIELDS + [
    'register', 'split', 'group_alias',
    # v3 追加
    'condition_detail', 'generation_method', 'augment_from', 'augment_params',
    'text_norm', 'split_source',
]


# ---------------------------------------------------------------------------
# 16kHz 单声道 PCM16 WAV 读写
# ---------------------------------------------------------------------------

def load_wav_16k(path):
    """读取 16k WAV；立体声取均值；返回 float32 [-1,1] 与采样率。"""
    from src.audio_io import load_wav
    return load_wav(path)


def write_wav_pcm16(path, audio, sr=SAMPLE_RATE) -> Path:
    """写 16kHz 单声道 PCM16 WAV。先写临时文件再原子替换，避免半截文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.clip(x, -1.0, 1.0)
    pcm = np.rint(x * 32767.0).astype('<i2')
    tmp = path.with_name(path.name + '.tmp')
    with wave.open(str(tmp), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    if path.exists():
        path.unlink()
    os.replace(str(tmp), str(path))
    return path


def wav_pcm16_ok(path) -> bool:
    """快速校验：存在且为 16kHz 单声道 PCM16、非空。"""
    path = Path(path)
    if not path.exists() or path.stat().st_size < 128:
        return False
    try:
        with wave.open(str(path), 'rb') as wf:
            return (wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1
                    and wf.getsampwidth() == 2 and wf.getnframes() > 0)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 重采样（窗函数 sinc 插值，纯 numpy）
# ---------------------------------------------------------------------------

def resample_sinc(x, ratio, taps=16):
    """按 ratio 重采样：输出长度 = round(len(x)*ratio)，ratio>1 变长（变慢/降调）。

    使用 Kaiser 窗 sinc 插值，比线性插值更少高频混叠，且完全确定性。
    """
    x = np.asarray(x, dtype=np.float64)
    n_in = len(x)
    if n_in == 0:
        return np.zeros(0, dtype=np.float32)
    n_out = max(1, int(round(n_in * ratio)))
    if abs(ratio - 1.0) < 1e-9:
        return x.astype(np.float32)
    half = taps // 2
    offsets = np.arange(-half + 1, half + 1)
    kaiser = np.kaiser(taps, 8.0)
    # 源坐标（input index = out_index / ratio）
    pos = np.arange(n_out, dtype=np.float64) / float(ratio)
    base = np.floor(pos).astype(np.int64)
    frac = pos - base
    idx = np.clip(base[:, None] + offsets[None, :], 0, n_in - 1)
    w = np.sinc(offsets[None, :] - frac[:, None]) * kaiser[None, :]
    w /= np.sum(w, axis=1, keepdims=True)
    y = np.sum(x[idx] * w, axis=1)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# WSOLA 时域伸缩（保音高变速）
# ---------------------------------------------------------------------------

def _hann_periodic(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def wsola_time_stretch(x, alpha, frame_ms=50.0, hop_ms=12.5,
                       search_ms=12.5, step=8):
    """WSOLA 时间伸缩：输出长度 ≈ len(x) * alpha（alpha>1 变慢/变长）。

    保音高，只改时长，正是 edge-tts rate 参数的效果。纯 numpy、确定性。
    """
    x = np.asarray(x, dtype=np.float64)
    n_in = len(x)
    n_out = max(1, int(round(n_in * float(alpha))))
    if n_in == 0:
        return np.zeros(0, dtype=np.float32)
    if abs(alpha - 1.0) < 1e-9:
        return x.astype(np.float32)

    frame = int(round(frame_ms * SAMPLE_RATE / 1000.0))
    hop_s = int(round(hop_ms * SAMPLE_RATE / 1000.0))
    search = int(round(search_ms * SAMPLE_RATE / 1000.0))
    hop_a = hop_s / float(alpha)
    frame = max(32, frame)
    hop_s = max(8, hop_s)

    win = _hann_periodic(frame)
    pad = frame + search + 1
    xp = np.concatenate([x, np.zeros(pad, dtype=np.float64)])
    out = np.zeros(n_out + frame + hop_s, dtype=np.float64)
    norm = np.zeros_like(out)

    def ola(start_in, out_pos):
        seg = xp[start_in:start_in + frame]
        out[out_pos:out_pos + frame] += seg * win
        norm[out_pos:out_pos + frame] += win

    prev_start = 0
    ola(0, 0)
    k = 1
    # 预先构造候选偏移模板
    offsets = np.arange(-search, search + 1, step)
    while True:
        out_pos = k * hop_s
        if out_pos >= n_out:
            break
        nominal = int(round(k * hop_a))
        # 上一个片段平移一个合成 hop 后的“自然延续”
        cont = xp[prev_start + hop_s:prev_start + hop_s + frame]
        cont0 = cont - cont.mean()
        cn = float(np.linalg.norm(cont0)) + 1e-9

        lo = max(0, nominal - search)
        hi = min(len(xp) - frame, nominal + search)
        if hi <= lo:
            best = min(max(0, nominal), len(xp) - frame)
        else:
            starts = np.arange(lo, hi + 1, step)
            if starts.size == 0 or (nominal >= lo and nominal <= hi):
                starts = np.unique(np.append(starts, min(max(nominal, lo), hi)))
            cand = xp[starts[:, None] + np.arange(frame)[None, :]]
            c0 = cand - cand.mean(axis=1, keepdims=True)
            cnorms = np.linalg.norm(c0, axis=1) + 1e-9
            corr = (c0 @ cont0) / (cnorms * cn)
            best = int(starts[int(np.argmax(corr))])
        best = max(0, min(best, len(xp) - frame))
        ola(best, out_pos)
        prev_start = best
        k += 1

    out = out[:n_out]
    norm = norm[:n_out]
    nz = norm > 1e-6
    out[nz] /= norm[nz]
    # 末尾不足一帧的部分能量可能偏低，但语音通常已结束；统一做峰值保护
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = out * (0.999 / peak)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# f0 估计与变调
# ---------------------------------------------------------------------------

def estimate_f0(x, sr=SAMPLE_RATE, fmin=60.0, fmax=400.0,
                frame_ms=40.0, hop_ms=20.0):
    """自相关法估计基频中位数；静音/清音帧跳过。失败返回 None。"""
    x = np.asarray(x, dtype=np.float64)
    frame = int(round(frame_ms * sr / 1000.0))
    hop = int(round(hop_ms * sr / 1000.0))
    lag_min = max(2, int(sr / fmax))
    lag_max = int(sr / fmin)
    vals = []
    for i in range(0, max(1, len(x) - frame + 1), hop):
        f = x[i:i + frame]
        if len(f) < frame:
            break
        if float(np.sqrt(np.mean(f * f))) < 0.01:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, 'full')[frame - 1:]
        if ac.size <= lag_max or ac[0] <= 0:
            continue
        seg = ac[lag_min:lag_max]
        j = int(np.argmax(seg))
        if seg[j] > 0.3 * ac[0]:
            vals.append(sr / float(lag_min + j))
    if not vals:
        return None
    return float(np.median(vals))


def pitch_shift_hz(x, delta_hz, f0=None, voice=None):
    """把音高平移 delta_hz（Hz），时长保持不变。

    实现：先按 m=(f0+delta)/f0 重采样（调 ×m，时长 ÷m），
    再 WSOLA 以 alpha=m 拉伸回原时长。
    返回 (y, params)；params 记录 f0 估计值、倍率等，写入清单可审计。
    """
    x = np.asarray(x, dtype=np.float32)
    if f0 is None:
        f0 = estimate_f0(x)
    f0_source = 'estimated'
    if not f0 or not (40.0 <= float(f0) <= 600.0):
        f0 = DEFAULT_F0.get(voice, 200.0)
        f0_source = 'default_by_voice'
    f0 = float(f0)
    target = max(50.0, f0 + float(delta_hz))
    m = target / f0
    y = resample_sinc(x, 1.0 / m)
    y = wsola_time_stretch(y, m)
    # 长度对齐到原长度（不同 alpha 下可能差 1-2 个样本）
    n = len(x)
    if len(y) > n:
        y = y[:n]
    elif len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y), dtype=np.float32)])
    return y.astype(np.float32), {
        'f0_est_hz': round(f0, 2), 'f0_source': f0_source,
        'pitch_multiplier': round(m, 5),
        'target_f0_hz': round(target, 2),
    }


def time_scale(x, speed_factor):
    """语速倍率 speed_factor：>1 变快，<1 变慢；音高不变。"""
    x = np.asarray(x, dtype=np.float32)
    speed_factor = float(speed_factor)
    alpha = 1.0 / speed_factor
    y = wsola_time_stretch(x, alpha)
    return y.astype(np.float32), {'time_scale': round(alpha, 5),
                                  'speed_factor': round(speed_factor, 5)}


def augment_condition(source_audio, condition, voice=None):
    """对源音频施加 v3 条件，返回 (y, detail)。

    detail 是可直接 JSON 序列化的字典，用于清单与审计。
    """
    if condition not in V3_CONDITIONS:
        raise KeyError(f'未知 v3 条件: {condition}')
    kind, param, desc = V3_CONDITIONS[condition]
    x = np.asarray(source_audio, dtype=np.float32)
    if kind == 'rate':
        speed = 1.0 + float(param)          # +18% -> 1.18, -18% -> 0.82
        y, params = time_scale(x, speed)
        params['target_rate_percent'] = int(round(float(param) * 100))
    elif kind == 'pitch':
        y, params = pitch_shift_hz(x, float(param), voice=voice)
        params['pitch_delta_hz'] = float(param)
    else:  # pragma: no cover
        raise ValueError(kind)
    detail = {'method': 'offline_signal_aug', 'kind': kind,
              'condition': condition, 'desc': desc, **params}
    return y, detail


def format_detail(detail) -> str:
    """把 detail 字典压成稳定的单行字符串，便于写入 CSV。"""
    return json.dumps(detail, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


# ---------------------------------------------------------------------------
# 清单 / 划分辅助
# ---------------------------------------------------------------------------

def load_v2_split_map(path=V2_SPLIT):
    rows = read_csv_rows(path)
    return {r['expression_group_id']: r['subset'] for r in rows}


def load_v2_manifest(path=V2_MANIFEST):
    return read_csv_rows(path)


def baseline_source_key(row):
    """判断某行是否可作为「同一 (表达组, 音色) 的基准音频」用于离线增广。

    基准 = 语速 0%、音高 0 的 edge-tts 音频（v2 text_morphology 或 v1 legacy）。
    """
    if (row.get('tts_engine') or '') != 'edge_tts':
        return False
    try:
        rate = float(row.get('rate_percent') or 0)
        pitch = float(row.get('pitch') or 0)
    except ValueError:
        return False
    return rate == 0.0 and pitch == 0.0


def build_baseline_source_index(rows):
    """(expression_group_id, voice_id) -> row（基准音频所在行）。

    若同一键出现多条基准行，取字典序最小的 utterance_id 以保证确定性。
    """
    idx = {}
    for r in rows:
        if r.get('quality_status') == 'excluded':
            continue
        if not baseline_source_key(r):
            continue
        key = (r['expression_group_id'], r['voice_id'])
        cur = idx.get(key)
        if cur is None or r['utterance_id'] < cur['utterance_id']:
            idx[key] = r
    return idx


def manifest_v3_row_defaults(row):
    """把任意行补齐为 v3 清单字段（缺失字段为空串），便于合并写出。"""
    return {k: row.get(k, '') for k in MANIFEST_V3_FIELDS}


def split_manifest_v3_hash(path=None) -> str:
    return sha256_file(path or V3_SPLIT)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def text_fingerprint(s: str) -> str:
    return hashlib.sha256(normalize_text(s).encode('utf-8')).hexdigest()[:16]


def is_audio_usable(path):
    """质量检查 + 非排除判定（复用 v2 阈值，保证口径一致）。"""
    if not Path(path).exists():
        return 'excluded', 'missing_file', None
    stats = analyze_audio(path)
    status, reasons = quality_flags_for(stats)
    return status, reasons, stats


def duration_ratio(audio_a, audio_b):
    la = len(audio_a) / SAMPLE_RATE
    lb = len(audio_b) / SAMPLE_RATE
    return la / lb if lb else 0.0

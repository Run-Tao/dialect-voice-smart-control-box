# -*- coding: utf-8 -*-
"""第二阶段：粤语命令语料扩充的数据集构建、清洗、泄漏审计与固定划分。

设计原则
--------
1. 不改动 `src/` 里已被端到端控制链路使用的现有接口，只新增模块。
2. 清单在既有字段之后追加字段；旧清单读入时新字段为空串，保持向后兼容。
3. 文本表达组 (expression_group_id) 是所有划分的原子单位：
   同一表达的全部声学变体（音色 / 语速 / 语气词省略）必须整体落在同一子集。
4. 划分结果写入固定文件并记录 sha256，实验只读取该文件，不重新随机划分。
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

INTENTS = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']

# 音色身份：不同 speaker_id 可能不是独立音色，必须单独给出 voice_identity
VOICE_IDENTITY = {
    'canto-tts-default': 'canto-tts-default',
    'zh-HK-HiuGaaiNeural': 'edge-zh-HK-HiuGaaiNeural',
    'zh-HK-HiuMaanNeural': 'edge-zh-HK-HiuMaanNeural',
    'zh-HK-WanLungNeural': 'edge-zh-HK-WanLungNeural',
}

# 第一阶段清单列（与 scripts/generate_tts_corpus.py 输出保持一致并追加）
MANIFEST_FIELDS = [
    'utterance_id', 'audio_path', 'speaker_id', 'dialect', 'command', 'language',
    'text', 'expression_type', 'semantic_pair_id', 'source', 'is_synthetic',
    'duration', 'sample_rate', 'split_group', 'text_verified_by', 'quality_flags',
    'notes',
    # 第二阶段追加
    'expression_group_id', 'tts_engine', 'voice_id', 'voice_identity',
    'rate_percent', 'pitch', 'volume', 'condition', 'source_text',
    'quality_status', 'quality_reasons', 'generation_batch', 'channels', 'sha256',
]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path) -> list:
    path = Path(path)
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows, fieldnames=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else MANIFEST_FIELDS
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    return path


def text_hash(s: str) -> str:
    return hashlib.sha256((s or '').strip().encode('utf-8')).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 音频质量检查
# ---------------------------------------------------------------------------

def analyze_audio(path):
    """读取16k单声道WAV并返回质量统计。使用项目既有 src.audio_io.load_wav。"""
    from src.audio_io import load_wav
    import wave

    audio, sr = load_wav(path)
    with wave.open(str(path), 'rb') as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
    n = len(audio)
    dur = n / sr if sr else 0.0
    if n == 0:
        return {'sample_rate': sr, 'channels': channels, 'sampwidth': sampwidth,
                'duration': 0.0, 'rms': 0.0, 'peak': 0.0, 'silence_ratio': 1.0,
                'clipping_ratio': 0.0, 'samples': 0}
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    silence = float(np.mean(np.abs(audio) < 0.01))
    clipping = float(np.mean(np.abs(audio) >= 0.999))
    # 有效语音占比：以帧级能量相对峰值判定
    frame = int(0.02 * sr)
    hop = int(0.01 * sr)
    if n >= frame:
        idx = range(0, n - frame + 1, hop)
        e = np.array([np.sum(audio[i:i + frame].astype(np.float64) ** 2) for i in idx])
        thr = max(float(np.max(e)) * 10 ** (-3.5), 1e-12)
        speech = float(np.mean(e >= thr))
    else:
        speech = 0.0
    return {'sample_rate': sr, 'channels': channels, 'sampwidth': sampwidth,
            'duration': dur, 'rms': rms, 'peak': peak, 'silence_ratio': silence,
            'clipping_ratio': clipping, 'speech_frame_ratio': speech, 'samples': n}


def quality_flags_for(stats, expected_sample_rate=16000):
    """返回 (status, reasons)。status in {'ok','flagged','excluded'}。

    阈值刻意宽松，只用于排除明显无效的音频；所有标记都会写入清单可审计。
    """
    reasons = []
    status = 'ok'
    if stats['samples'] == 0:
        return 'excluded', 'empty_audio'
    if stats['sample_rate'] != expected_sample_rate:
        reasons.append(f"sample_rate={stats['sample_rate']}")
        status = 'excluded'
    if stats['channels'] != 1:
        reasons.append(f"channels={stats['channels']}")
        status = 'excluded'
    if stats['duration'] < 0.35:
        reasons.append('too_short_lt_0.35s')
        status = 'excluded'
    elif stats['duration'] < 0.5:
        reasons.append('short_lt_0.5s')
        if status == 'ok':
            status = 'flagged'
    if stats['duration'] > 12.0:
        reasons.append('too_long_gt_12s')
        status = 'excluded'
    if stats['rms'] < 0.002:
        reasons.append('near_silent')
        status = 'excluded'
    if stats['silence_ratio'] > 0.90:
        reasons.append('high_silence_gt_0.90')
        if status == 'ok':
            status = 'flagged'
    if stats['clipping_ratio'] > 0.01:
        reasons.append('clipping_gt_1pct')
        if status == 'ok':
            status = 'flagged'
    if stats.get('speech_frame_ratio', 1.0) < 0.15:
        reasons.append('low_speech_ratio_lt_0.15')
        if status == 'ok':
            status = 'flagged'
    if stats['peak'] <= 0.0:
        reasons.append('all_zero')
        status = 'excluded'
    return status, ';'.join(reasons)


# ---------------------------------------------------------------------------
# 文本规范化与近重复检测
# ---------------------------------------------------------------------------

_PUNCT = set('，。、；：！？,.!?;:　 \t\n\r"\'“”‘’()（）[]【】')


def normalize_text(s: str) -> str:
    """文本规范化：去标点、去空白、繁简统一（仅用于重复检测，不改清单原值）。"""
    s = (s or '').strip()
    out = []
    for ch in s:
        if ch in _PUNCT:
            continue
        out.append(_TRAD2SIMP.get(ch, ch))
    return ''.join(out)


# 繁简映射：仅收录「繁简字形不同但指同一词」的字，用于重复检测。
# 注意：粤语特有字（係/啲/咩/嘢/睇/咗/唔/冇/佢/呢/嗰/嘅/嚟/喺/啱/㗎/喇/噉）保留原字形，
# 因为它们简繁同形且有语言学意义，错误映射会改变文本含义。
_TRAD2SIMP = {
    '狀': '状',
    '燈': '灯',
    '簾': '帘',
    '態': '态',
    '備': '备',
    '開': '开',
    '關': '关',
    '問': '问',
    '請': '请',
    '講': '讲',
    '視': '视',
    '閂': '闩',
    '曬': '晒',
    '裝': '装',
    '電': '电',
    '氣': '气',
    '當': '当',
    '錯': '错',
    '雙': '双',
    '願': '愿',
    '讓': '让',
    '給': '给',
    '從': '从',
    '還': '还',
    '總': '总',
    '與': '与',
    '個': '个',
    '們': '们',
    '麼': '么',
    '樣': '样',
    '準': '准',
    '設': '设',
    '淨': '净',
    '熱': '热',
    '飲': '饮',
    '飯': '饭',
    '聽': '听',
    '買': '买',
    '賣': '卖',
    '錢': '钱',
    '時': '时',
    '間': '间',
    '鐘': '钟',
    '風': '风',
    '溫': '温',
    '調': '调',
    '機': '机',
    '聲': '声',
    '響': '响',
    '頭': '头',
    '車': '车',
    '門': '门',
    '牆': '墙',
    '樓': '楼',
    '層': '层',
    '廳': '厅',
    '廚': '厨',
    '衛': '卫',
    '書': '书',
    '報': '报',
    '紙': '纸',
    '筆': '笔',
    '畫': '画',
    '戲': '戏',
    '樂': '乐',
    '學': '学',
    '歸': '归',
}


def char_ngrams(s: str, n: int = 2) -> set:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: str, b: str, n: int = 2) -> float:
    A, B = char_ngrams(a, n), char_ngrams(b, n)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def phoneme_signature(text: str):
    """用 canto-tts 的 Jyutping 前端给出粤语音节序列。

    该检查的意义是客观的“可发音性/粤语音节覆盖”检查，不是自然度判定；
    naturalness 仍由人工规则审查给出，并在清单中标注复核人与状态。
    """
    from canto_tts.backends.base import TTSBackend
    try:
        s = TTSBackend.to_phoneme(text)
    except Exception:
        return None
    return s


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------

def _hash_seed(*parts) -> int:
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode('utf-8')).hexdigest()
    return int(h[:12], 16)


def row_expression_group(row) -> str:
    """取表达组ID：优先显式字段，否则用 (command, 规范化文本) 兜底。"""
    g = (row.get('expression_group_id') or '').strip()
    if g:
        return g
    return f"{row.get('command', '')}::{normalize_text(row.get('text', ''))}"


def group_predicate(row) -> str:
    """表达组的属性：语域/长度/语速档，用于分层。"""
    return (row.get('condition') or 'text_morphology').strip()


def assign_expression_groups(groups, seed=42, train_ratio=0.70, val_ratio=0.15):
    """按表达组做分层随机划分（轨道A：未见表达）。

    分层键 = (command, register)。同一表达组的所有样本整体进入同一子集。
    返回 {group_id: 'train'|'val'|'test'}。
    """
    by_key = defaultdict(list)
    for gid, info in groups.items():
        key = (info['command'], info.get('register', ''))
        by_key[key].append(gid)

    assignment = {}
    for key in sorted(by_key):
        gids = sorted(by_key[key], key=lambda g: _hash_seed(seed, key, g))
        n = len(gids)
        if n == 1:
            assignment[gids[0]] = 'train'
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 1
        if n_train + n_val >= n:
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test
        n_test = n - n_train - n_val
        if n < 3:
            assignment[gids[0]] = 'train'
            assignment[gids[1]] = 'test'
            continue
        for i, gid in enumerate(gids):
            if i < n_train:
                assignment[gid] = 'train'
            elif i < n_train + n_val:
                assignment[gid] = 'val'
            else:
                assignment[gid] = 'test'
    return assignment


def split_file_hash(path) -> str:
    return sha256_file(path)


def save_split_manifest(path, assignment, extra=None):
    rows = []
    for gid in sorted(assignment):
        a = assignment[gid]
        rows.append({'expression_group_id': gid,
                     'subset': a['subset'], 'command': a['command'],
                     'register': a['register'], 'n_samples': a['n_samples'],
                     'voices': ';'.join(sorted(a['voices'])),
                     'text': a.get('text', '')})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['expression_group_id', 'subset', 'command',
                                          'register', 'n_samples', 'voices', 'text'])
        w.writeheader()
        w.writerows(rows)
    payload = {'split_file': str(path), 'sha256': sha256_file(path)}
    if extra:
        payload.update(extra)
    return payload

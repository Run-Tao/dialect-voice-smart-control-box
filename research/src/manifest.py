# -*- coding: utf-8 -*-
"""数据清单读取、字段检查和路径检查。"""
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ['utterance_id', 'audio_path', 'speaker_id', 'dialect', 'command']
ALIASES = {'utterance_id': ['utt_id'], 'command': ['intent_id']}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestRecord:
    utterance_id: str
    audio_path: str
    speaker_id: str
    dialect: str
    command: str
    duration: str = ''
    sample_rate: str = ''
    language: str = ''
    text: str = ''
    expression_type: str = ''
    semantic_pair_id: str = ''
    source: str = ''
    is_synthetic: str = ''
    split_group: str = ''
    text_verified_by: str = ''
    quality_flags: str = ''
    source_model: str = ''
    notes: str = ''
    # 以下为第二阶段扩充字段；旧清单缺失时为空字符串，保证向后兼容
    expression_group_id: str = ''      # 文本表达组ID，同组的所有声学变体必须整体归属同一数据子集
    tts_engine: str = ''               # canto_tts / edge_tts / ''
    voice_id: str = ''                 # 具体发音人音色ID
    voice_identity: str = ''           # 独立音色身份，用于未见音色实验分组
    rate_percent: str = ''             # 语速参数（edge-tts 为百分比，canto-tts 为 fixed）
    pitch: str = ''                    # 音高参数
    volume: str = ''                   # 音量参数
    condition: str = ''                # text_morphology / rate（同一音色内的条件类型）
    source_text: str = ''              # 允许省略语气词后的合成文本；与 text 相同表示未省略
    quality_status: str = ''           # ok / excluded
    quality_reasons: str = ''          # 排除或标记原因，分号分隔
    generation_batch: str = ''         # 生成批次
    channels: str = ''                 # 声道数
    sha256: str = ''                   # 音频内容哈希，用于完全重复检测


def load_manifest(path, check_files=True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"清单文件不存在: {path}")

    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ManifestError("清单为空或没有表头")
        missing = []
        for x in REQUIRED_FIELDS:
            aliases = ALIASES.get(x, [])
            if x not in reader.fieldnames and not any(a in reader.fieldnames for a in aliases):
                missing.append(x)
        if missing:
            raise ManifestError(f"清单缺少字段: {missing}")
        rows = list(reader)

    if not rows:
        raise ManifestError("清单没有数据行")

    records = []
    errors = []
    seen = set()
    for i, row in enumerate(rows, start=2):
        uid = (row.get('utterance_id') or row.get('utt_id') or '').strip()
        if not uid:
            errors.append(f"第{i}行: utterance_id为空")
            continue
        if uid in seen:
            errors.append(f"第{i}行: utterance_id重复: {uid}")
            continue
        seen.add(uid)

        vals = {}
        for k in REQUIRED_FIELDS:
            v = row.get(k)
            if not v:
                for a in ALIASES.get(k, []):
                    v = row.get(a)
                    if v:
                        break
            vals[k] = (v or '').strip()
        if any(not vals[k] for k in REQUIRED_FIELDS):
            errors.append(f"第{i}行: 必填字段存在空值")
            continue

        raw_path = Path(vals['audio_path'])
        audio_path = raw_path if raw_path.is_absolute() else (path.parent / raw_path)
        audio_path = audio_path.resolve()
        if check_files and not audio_path.exists():
            errors.append(f"第{i}行: 音频文件不存在: {audio_path}")

        records.append(ManifestRecord(
            utterance_id=uid,
            audio_path=str(audio_path),
            speaker_id=vals['speaker_id'],
            dialect=vals['dialect'],
            command=vals['command'],
            duration=(row.get('duration') or '').strip(),
            sample_rate=(row.get('sample_rate') or '').strip(),
            language=(row.get('language') or '').strip(),
            text=(row.get('text') or '').strip(),
            expression_type=(row.get('expression_type') or '').strip(),
            semantic_pair_id=(row.get('semantic_pair_id') or '').strip(),
            source=(row.get('source') or '').strip(),
            is_synthetic=(row.get('is_synthetic') or '').strip(),
            split_group=(row.get('split_group') or '').strip(),
            text_verified_by=(row.get('text_verified_by') or '').strip(),
            quality_flags=(row.get('quality_flags') or '').strip(),
            source_model=(row.get('source_model') or '').strip(),
            notes=(row.get('notes') or '').strip(),
            expression_group_id=(row.get('expression_group_id') or '').strip(),
            tts_engine=(row.get('tts_engine') or '').strip(),
            voice_id=(row.get('voice_id') or '').strip(),
            voice_identity=(row.get('voice_identity') or '').strip(),
            rate_percent=(row.get('rate_percent') or '').strip(),
            pitch=(row.get('pitch') or '').strip(),
            volume=(row.get('volume') or '').strip(),
            condition=(row.get('condition') or '').strip(),
            source_text=(row.get('source_text') or '').strip(),
            quality_status=(row.get('quality_status') or '').strip(),
            quality_reasons=(row.get('quality_reasons') or '').strip(),
            generation_batch=(row.get('generation_batch') or '').strip(),
            channels=(row.get('channels') or '').strip(),
            sha256=(row.get('sha256') or '').strip(),
        ))

    if errors:
        raise ManifestError("\n".join(errors))
    return records


def manifest_summary(records):
    speakers = sorted({r.speaker_id for r in records})
    commands = sorted({r.command for r in records})
    return {
        'num_utterances': len(records),
        'num_speakers': len(speakers),
        'num_commands': len(commands),
        'speakers': speakers,
        'commands': commands,
        'counts_by_speaker': dict(Counter(r.speaker_id for r in records)),
        'counts_by_command': dict(Counter(r.command for r in records)),
    }

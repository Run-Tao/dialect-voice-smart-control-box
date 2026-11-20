# -*- coding: utf-8 -*-
"""演示音频目录：自动挑选「未见表达」的测试音频与一条未知指令音频。

数据来源（均为只读）：
  * data/splits/track_a_expression_split.csv   —— subset=test 的表达组（未参与训练）
  * data/tts_corpus/manifest_yue_v2.csv         —— audio_path 相对 data/tts_corpus/
  * data/tts_corpus/unknown_v3.csv（Lead 后续提供，可选）
  * outputs/unknown_tts_log.csv                 —— 二期未知指令 TTS 日志（回退）
  * data/tts_unknown/                           —— 二期未知指令音频目录（回退）

选择策略：每个命令优先挑 canto-tts-default 音色、表达组序号靠前的测试音频；
保证六个按钮各对应一条真实存在、且属于 test 划分的 WAV。不做模型打分，
避免演示台启动时批量推理拖慢速度。
"""
import csv
from pathlib import Path

try:  # 复用控制服务里的中文命令名，避免两份文案漂移
    from src.control_server_v3 import COMMAND_LABELS, SUPPORTED_COMMANDS
except Exception:  # pragma: no cover - 独立运行/导入路径异常时的回退
    COMMAND_LABELS = {
        'LIGHT_ON': '開燈', 'LIGHT_OFF': '關燈',
        'CURTAIN_OPEN': '打開窗簾', 'CURTAIN_CLOSE': '關閉窗簾',
        'ALL_OFF': '全屋關閉', 'STATUS_ALL': '查詢全屋狀態',
    }
    SUPPORTED_COMMANDS = ('LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN',
                          'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL')

COMMAND_ORDER = tuple(SUPPORTED_COMMANDS)
# 演示时优先使用与既有 DTW 演示一致的 TTS 音色（该音色样本通常更稳定）
PREFERRED_VOICES = ('canto-tts-default',)


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _as_path(base, raw, roots):
    """把清单里的相对/绝对路径解析成存在的文件路径。"""
    raw = (raw or '').strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else None
    for root in roots:
        candidate = Path(root) / p
        if candidate.exists():
            return candidate
    return None


def _test_groups(base):
    rows = _read_csv(base / 'data' / 'splits' / 'track_a_expression_split.csv')
    return {r['expression_group_id']: r for r in rows if r.get('subset') == 'test'}


def _command_candidates(base, test_groups):
    """返回 {command: [候选音频 dict, ...]}，候选均已排序。"""
    candidates = {cmd: [] for cmd in COMMAND_ORDER}
    manifest = _read_csv(base / 'data' / 'tts_corpus' / 'manifest_yue_v2.csv')
    corpus_dir = base / 'data' / 'tts_corpus'
    for row in manifest:
        command = row.get('command')
        if command not in candidates or row.get('split') != 'test':
            continue
        group_id = row.get('expression_group_id') or row.get('split_group') or ''
        group = test_groups.get(group_id)
        if not group or group.get('command') != command:
            continue
        audio = _as_path(base, row.get('audio_path'), [corpus_dir, base])
        if audio is None:
            continue
        candidates[command].append({
            'kind': 'command',
            'command': command,
            'label': COMMAND_LABELS.get(command, command),
            'text': group.get('text') or row.get('text') or '',
            'register': group.get('register') or row.get('register') or '',
            'group': group_id,
            'voice': row.get('voice_identity') or row.get('speaker_id') or '',
            'audio_path': str(audio),
            'source': 'track_a_expression_split:test + manifest_yue_v2',
        })

    # 清单缺失时的兜底：直接按文件名命令前缀在语料目录里找 WAV
    for command in COMMAND_ORDER:
        if candidates[command]:
            continue
        for wav in sorted((base / 'data' / 'tts_corpus').rglob('*.wav')):
            if wav.name.startswith(command + '_'):
                candidates[command].append({
                    'kind': 'command',
                    'command': command,
                    'label': COMMAND_LABELS.get(command, command),
                    'text': '',
                    'register': '',
                    'group': '',
                    'voice': '',
                    'audio_path': str(wav),
                    'source': 'corpus_glob_fallback',
                })
                break

    for command in COMMAND_ORDER:
        def sort_key(item):
            voice_rank = 0
            for i, voice in enumerate(PREFERRED_VOICES):
                if item['voice'] == voice:
                    voice_rank = i
                    break
            else:
                voice_rank = len(PREFERRED_VOICES)
            return (voice_rank, item['group'], item['audio_path'])
        candidates[command].sort(key=sort_key)
    return candidates


def _unknown_candidates(base):
    """未知指令音频候选，按优先级返回存在的文件。"""
    found = []
    corpus_dir = base / 'data' / 'tts_corpus'

    # 1) Lead 的 v3 未知指令清单（字段名可能是 audio_path / path / wav_path）
    for row in _read_csv(corpus_dir / 'unknown_v3.csv'):
        raw = row.get('audio_path') or row.get('path') or row.get('wav_path')
        status = (row.get('status') or 'ok').lower()
        if status not in ('ok', ''):
            continue
        audio = _as_path(base, raw, [corpus_dir, base])
        if audio is not None:
            found.append({
                'kind': 'unknown',
                'command': None,
                'label': '未知指令（應拒識）',
                'text': row.get('text') or row.get('transcript') or row.get('source_text') or '',
                'audio_path': str(audio),
                'source': 'data/tts_corpus/unknown_v3.csv',
            })
            break

    # 2) 二期未知指令生成日志
    if not found:
        for row in _read_csv(base / 'outputs' / 'unknown_tts_log.csv'):
            status = (row.get('status') or 'ok').lower()
            if status not in ('ok', ''):
                continue
            audio = _as_path(base, row.get('audio_path'), [base, corpus_dir])
            if audio is not None:
                found.append({
                    'kind': 'unknown',
                    'command': None,
                    'label': '未知指令（應拒識）',
                    'text': row.get('text') or '',
                    'audio_path': str(audio),
                    'source': 'outputs/unknown_tts_log.csv',
                })
                break

    # 3) 二期未知指令语料目录（manifest.csv 或直接 glob）
    if not found:
        unknown_dir = base / 'data' / 'tts_unknown'
        for row in _read_csv(unknown_dir / 'manifest.csv'):
            raw = row.get('audio_path') or row.get('path') or row.get('wav_path')
            audio = _as_path(base, raw, [unknown_dir, base, unknown_dir / 'wav16k'])
            if audio is not None:
                found.append({
                    'kind': 'unknown',
                    'command': None,
                    'label': '未知指令（應拒識）',
                    'text': row.get('text') or row.get('transcript') or '',
                    'audio_path': str(audio),
                    'source': 'data/tts_unknown/manifest.csv',
                })
                break
    if not found:
        wavs = sorted((base / 'data' / 'tts_unknown').rglob('*.wav'))
        if wavs:
            found.append({
                'kind': 'unknown',
                'command': None,
                'label': '未知指令（應拒識）',
                'text': '',
                'audio_path': str(wavs[0]),
                'source': 'data/tts_unknown/wav16k',
            })
    return found


def build_demo_catalog(base_dir=None):
    """构建演示台音频目录。

    返回 dict：``commands``（六个命令各一条样本）、``unknown``（未知音频或 None）、
    ``warnings``（缺数据时的中文提示）、``base_dir``。
    """
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    warnings = []
    test_groups = _test_groups(base)
    if not test_groups:
        warnings.append('未找到 data/splits 的 test 划分，命令按钮将退化为按文件名挑选。')
    candidates = _command_candidates(base, test_groups)

    commands = []
    for command in COMMAND_ORDER:
        items = candidates.get(command) or []
        if not items:
            warnings.append(f'未找到可用的測試音頻：{command}')
            continue
        chosen = dict(items[0])
        chosen['candidate_count'] = len(items)
        commands.append(chosen)

    unknown_list = _unknown_candidates(base)
    unknown = unknown_list[0] if unknown_list else None
    if unknown is None:
        warnings.append('未找到未知指令測試音頻，未知按鈕將不可用。')

    return {
        'base_dir': str(base),
        'commands': commands,
        'unknown': unknown,
        'candidates': {cmd: len(candidates.get(cmd) or []) for cmd in COMMAND_ORDER},
        'warnings': warnings,
    }


if __name__ == '__main__':  # 便于单独检查挑选结果
    import json
    import sys
    catalog = build_demo_catalog(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps(catalog, ensure_ascii=False, indent=2))

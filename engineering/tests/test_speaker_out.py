# -*- coding: utf-8 -*-
"""普通话播报层测试。

speaker_out.py 此前零测试。这里不碰真实声卡：mock 模式只验证文本与分支，
非 mock 模式注入 fake ``_player`` 验证「选了哪个 wav」，因此不依赖 sounddevice。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import speaker_out
from src.speaker_out import MandarinSpeaker

REAL_AUDIO_DIR = ROOT / 'data' / 'audio' / 'mandarin'


# ----------------------------------------------------------------------
# 路径与可用性
# ----------------------------------------------------------------------
def test_paths_are_derived_from_audio_dir(tmp_path):
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=True)
    assert sp.path_for('LIGHT_ON') == tmp_path / 'LIGHT_ON.wav'
    assert sp.asset_path('ACK_DONE') == tmp_path / 'ACK_DONE.wav'


def test_available_reflects_file_existence(tmp_path):
    (tmp_path / 'LIGHT_ON.wav').write_bytes(b'RIFF')
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=True)
    assert sp.available('LIGHT_ON') is True
    assert sp.available('LIGHT_OFF') is False
    assert sp.available_asset('LIGHT_ON') is True


def test_real_assets_cover_commands_and_feedback():
    """部署资产必须齐全，否则现场会静默不播。"""
    sp = MandarinSpeaker(audio_dir=REAL_AUDIO_DIR, mock=True)
    from src.commands import COMMANDS
    missing = [c for c in COMMANDS if not sp.available(c)]
    assert missing == [], f'缺少命令音频: {missing}'
    for asset in ('ACK_DONE', 'ACK_FAIL', 'REJECT_UNKNOWN', 'PRETRIGGER'):
        assert sp.available_asset(asset), f'缺少反馈音频: {asset}'


# ----------------------------------------------------------------------
# mock 模式
# ----------------------------------------------------------------------
def test_mock_speak_returns_true_without_touching_player(tmp_path, monkeypatch):
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=True)

    def boom(path):
        raise AssertionError('mock 模式不应调用播放器')

    sp._player = boom
    assert sp.speak('LIGHT_ON') is True


def test_mock_speak_unknown_command_returns_false(tmp_path):
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=True)
    assert sp.speak('NOT_A_COMMAND') is False


def test_wake_word_is_prefixed_to_spoken_text(tmp_path, capsys):
    sp = MandarinSpeaker(audio_dir=tmp_path, wake_word='小智小智', mock=True)
    sp.speak('LIGHT_ON')
    assert '小智小智打开灯' in capsys.readouterr().out


def test_feedback_assets_are_all_available_in_mock(tmp_path):
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=True)
    assert sp.speak_done() is True
    assert sp.speak_error() is True
    assert sp.speak_reject() is True
    assert sp.speak_pretrigger() is True


# ----------------------------------------------------------------------
# 非 mock：注入 fake player
# ----------------------------------------------------------------------
def test_speak_plays_matching_wav(tmp_path):
    (tmp_path / 'LIGHT_ON.wav').write_bytes(b'RIFF')
    played = []
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=False)
    sp._player = lambda path: played.append(Path(path))
    assert sp.speak('LIGHT_ON') is True
    assert played == [tmp_path / 'LIGHT_ON.wav']


def test_speak_missing_file_returns_false_and_does_not_play(tmp_path):
    played = []
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=False)
    sp._player = lambda path: played.append(Path(path))
    assert sp.speak('LIGHT_ON') is False
    assert played == []


def test_speak_asset_plays_feedback_wav(tmp_path):
    (tmp_path / 'ACK_DONE.wav').write_bytes(b'RIFF')
    played = []
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=False)
    sp._player = lambda path: played.append(Path(path))
    assert sp.speak_done() is True
    assert played == [tmp_path / 'ACK_DONE.wav']


def test_speak_clarify_uses_dialog_asset_name(tmp_path, capsys):
    from runtime import dialog

    expected = dialog.clarify_asset(['LIGHT_ON', 'LIGHT_OFF'])
    (tmp_path / f'{expected}.wav').write_bytes(b'RIFF')
    played = []
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=False)
    sp._player = lambda path: played.append(Path(path))
    assert sp.speak_clarify(['LIGHT_ON', 'LIGHT_OFF']) is True
    assert played == [tmp_path / f'{expected}.wav']


def test_speak_clarify_text_lists_both_candidates(tmp_path, capsys):
    sp = MandarinSpeaker(audio_dir=tmp_path, mock=True)
    sp.speak_clarify(['LIGHT_ON', 'LIGHT_OFF'])
    out = capsys.readouterr().out
    assert '打开灯' in out and '关灯' in out


# ----------------------------------------------------------------------
# 播放器构造
# ----------------------------------------------------------------------
def test_make_player_raises_helpful_error_without_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sounddevice', None)
    monkeypatch.setitem(sys.modules, 'soundfile', None)
    with pytest.raises(ImportError) as exc:
        speaker_out._make_player()
    assert 'sounddevice' in str(exc.value)

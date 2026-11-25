# -*- coding: utf-8 -*-
"""演示入口测试。

只测排版与编排逻辑，不加载模型、不碰声卡：大字卡的对齐是录屏时最容易
翻车的地方，必须锁死；语料相关用例在缺语料的机器上自动跳过。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_demo


# ----------------------------------------------------------------------
# 排版
# ----------------------------------------------------------------------
def test_text_width_counts_cjk_as_two():
    assert run_demo._text_width('abc') == 3
    assert run_demo._text_width('打开灯') == 6


def test_center_pads_to_exact_width():
    for text in ('', 'abc', '打开灯', '查询设备状态'):
        assert run_demo._text_width(run_demo._center(text, 20)) == 20


def test_render_card_has_uniform_display_width():
    card = run_demo.render_card('CURTAIN_OPEN', 0.987, True, 21.4)
    lines = card.splitlines()
    assert len(lines) == 7
    widths = {run_demo._text_width(line) for line in lines}
    assert len(widths) == 1, f'卡片各行宽度不一致: {widths}'


def test_render_card_shows_command_and_verdict():
    card = run_demo.render_card('LIGHT_ON', 0.951, True, 19.0)
    assert '打开灯' in card
    assert '已执行' in card
    assert '0.951' in card
    assert '19 ms' in card


def test_render_card_marks_rejection():
    card = run_demo.render_card('', 0.31, False, 12.0)
    assert '未识别' in card
    assert '已拒识' in card


# ----------------------------------------------------------------------
# 语料收集
# ----------------------------------------------------------------------
def test_collect_files_accepts_single_file(tmp_path):
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'RIFF')
    assert run_demo.collect_files(wav) == [wav]


def test_collect_files_walks_directory(tmp_path):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'a.wav').write_bytes(b'RIFF')
    (tmp_path / 'sub' / 'b.wav').write_bytes(b'RIFF')
    assert [p.name for p in run_demo.collect_files(tmp_path)] == ['a.wav', 'b.wav']


def test_one_per_command_returns_six_when_corpus_present():
    if run_demo.corpus_dir() is None:
        pytest.skip('本机没有演示语料')
    picked = run_demo.one_per_command()
    assert len(picked) == 6
    assert [name for name, _ in picked] == list(run_demo.COMMANDS)
    for command, path in picked:
        assert path.name.startswith(command + '_')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_list_mode_exits_zero(capsys):
    assert run_demo.main(['--list']) == 0
    out = capsys.readouterr().out
    assert 'LIGHT_ON' in out


def test_replay_without_audio_returns_2(capsys):
    assert run_demo.main(['--mode', 'replay']) == 2
    assert '需要 --audio' in capsys.readouterr().err


def test_replay_with_missing_path_returns_2(tmp_path, capsys):
    assert run_demo.main(['--mode', 'replay', '--audio', str(tmp_path / 'nope.wav')]) == 2
    assert '没有找到 WAV' in capsys.readouterr().err

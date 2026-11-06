# -*- coding: utf-8 -*-
"""对话层测试：去抖窗口 + 澄清状态机。

澄清状态机最容易出的两个错，这里都钉住：

1. 追问不能连着来（用户重说时**不许**再追问，否则变成问-答-再问的死循环）；
2. 澄清预算按「一轮」算，不按「整个会话」算——一轮结束后额度要清零，
   下一条独立的歧义语句仍然可以再澄清一次。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.dialog import (COMMAND_ZH, ClarifyDialog, Debouncer,
                            clarify_asset)
from src.commands import COMMANDS


# ----------------------------------------------------------------------
# 去抖
# ----------------------------------------------------------------------
def test_debouncer_suppresses_same_command_in_window():
    now = [1000.0]
    d = Debouncer(window_ms=2000, clock=lambda: now[0])
    assert d.allow('LIGHT_ON')
    assert not d.allow('LIGHT_ON')
    assert d.allow('LIGHT_OFF')          # 不同命令不受影响
    now[0] += 2.0
    assert d.allow('LIGHT_ON')           # 窗口过后放行


def test_debouncer_suppress_ms_reports_elapsed():
    now = [1000.0]
    d = Debouncer(window_ms=2000, clock=lambda: now[0])
    assert d.suppress_ms('LIGHT_ON') == -1.0
    d.allow('LIGHT_ON')
    now[0] += 0.5
    assert d.suppress_ms('LIGHT_ON') == pytest.approx(500.0)


def test_debouncer_reset_clears_history():
    d = Debouncer(window_ms=2000)
    d.allow('LIGHT_ON')
    d.reset()
    assert d.allow('LIGHT_ON')


# ----------------------------------------------------------------------
# 澄清
# ----------------------------------------------------------------------
def test_clarify_prompt_text_follows_candidate_order():
    """问句按传入顺序念（声学 top-1 在前），资产名按字典序（同一对固定同一个文件）。"""
    d = ClarifyDialog(max_clarify=1)
    prompt = d.request(['CURTAIN_OPEN', 'CURTAIN_CLOSE'])
    assert prompt is not None
    assert prompt.text == '你是要打开窗帘，还是关上窗帘？'
    assert prompt.candidates == ['CURTAIN_OPEN', 'CURTAIN_CLOSE']
    assert prompt.asset == 'CLARIFY_CURTAIN_CLOSE_CURTAIN_OPEN.wav'
    assert prompt.attempt == 1

    d2 = ClarifyDialog(max_clarify=1)
    prompt2 = d2.request(['CURTAIN_CLOSE', 'CURTAIN_OPEN'])
    assert prompt2.text == '你是要关上窗帘，还是打开窗帘？'
    assert prompt2.asset == prompt.asset


def test_clarify_asset_is_sorted_and_deterministic():
    a = clarify_asset(['CURTAIN_CLOSE', 'CURTAIN_OPEN'])
    b = clarify_asset(['CURTAIN_OPEN', 'CURTAIN_CLOSE'])
    assert a == b == 'CLARIFY_CURTAIN_CLOSE_CURTAIN_OPEN.wav'


def test_clarify_is_not_repeated_while_awaiting_retry():
    """用户正在重说时不能再追问。"""
    d = ClarifyDialog(max_clarify=1)
    assert d.request(['CURTAIN_OPEN', 'CURTAIN_CLOSE']) is not None
    assert d.awaiting_retry
    assert d.request(['CURTAIN_OPEN', 'CURTAIN_CLOSE']) is None


def test_retry_resets_the_budget():
    """一轮澄清结束后额度清零，下一条独立语句还能再澄清。"""
    d = ClarifyDialog(max_clarify=1)
    d.request(['CURTAIN_OPEN', 'CURTAIN_CLOSE'])
    d.on_retry(success=True)
    assert not d.awaiting_retry
    assert d.used == 0
    assert d.request(['CURTAIN_OPEN', 'CURTAIN_CLOSE']) is not None


def test_failed_retry_also_ends_the_round():
    d = ClarifyDialog(max_clarify=1)
    d.request(['CURTAIN_OPEN', 'CURTAIN_CLOSE'])
    d.on_retry(success=False)
    assert d.pending is None
    assert not d.awaiting_retry


def test_clarify_needs_two_candidates():
    d = ClarifyDialog(max_clarify=1)
    assert d.request(['LIGHT_ON']) is None
    assert d.request([]) is None


def test_exhausted_dialog_refuses():
    d = ClarifyDialog(max_clarify=0)
    assert d.exhausted
    assert d.request(['CURTAIN_OPEN', 'CURTAIN_CLOSE']) is None


def test_command_zh_matches_command_table():
    """runtime/dialog.py 的普通话说法必须与 src/commands.py 一致。

    两处各改一处的话，澄清问句会念错命令名。
    """
    assert set(COMMAND_ZH) == set(COMMANDS)
    for name, entry in COMMANDS.items():
        assert COMMAND_ZH[name] == entry['zh'], name

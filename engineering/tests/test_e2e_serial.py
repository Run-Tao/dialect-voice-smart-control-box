# -*- coding: utf-8 -*-
"""端到端串口闭环：PC 下发命令 -> 固件状态机 -> ACK/STATE -> PC 状态。

用 fake serial 把 SerialExecutor 的串口读写桥接到 runtime.serial_protocol 的
MockEsp32Controller（与 C++ 固件同协议），不依赖真实串口与硬件。
目的：证明「能烧录然后使用」的软件链路本身是通的，并暴露回执处理缺陷。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.serial_protocol import MockEsp32Controller, build_command
from src import control


class Esp32Bridge:
    """串口替身：write 时把字节喂给固件状态机，readline 返回其响应。"""

    def __init__(self, *args, **kwargs):
        self.dev = MockEsp32Controller()
        self.rx: list[str] = []
        self.written: list[bytes] = []
        self.timeout = 1.0

    def write(self, payload: bytes):
        self.written.append(payload)
        resp = self.dev.handle_line(payload.decode('ascii'))
        self.rx.extend(l for l in resp.strip().split('\n') if l)
        return len(payload)

    def readline(self) -> bytes:
        return (self.rx.pop(0) + '\n').encode('ascii') if self.rx else b''

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


@pytest.fixture
def bridge(monkeypatch):
    module = types.ModuleType('serial')
    module.Serial = Esp32Bridge
    monkeypatch.setitem(sys.modules, 'serial', module)
    return module


def _executor():
    ex = control.SerialExecutor('COM_TEST', seq_start=0)
    return ex, ex.ser


def test_e2e_light_on_then_status_all(bridge):
    ex, ser = _executor()
    ex.dispatch('LIGHT_ON')
    ack = ex.wait_ack()
    assert 'ACK' in ack['detail']
    assert ser.dev.state['light'] == 'on'

    ex.dispatch('STATUS_ALL')
    ex.wait_ack()
    # STATE 应紧随 ACK 到达，能被下一次 read 读到
    state_line = ser.readline().decode('ascii').strip()
    assert state_line == 'STATE,2,LIGHT=on,CURTAIN=closed'


def test_e2e_all_off_resets_state(bridge):
    ex, ser = _executor()
    ex.dispatch('LIGHT_ON')
    ex.wait_ack()
    ex.dispatch('CURTAIN_OPEN')
    ex.wait_ack()
    assert ser.dev.state == {'light': 'on', 'curtain': 'open'}

    ex.dispatch('ALL_OFF')
    ex.wait_ack()
    assert ser.dev.state == {'light': 'off', 'curtain': 'closed'}


def test_e2e_same_seq_is_dropped_by_firmware(bridge):
    """固件对重复 seq 回 DUP 且不改状态——这是 seq 缺陷的底层机制。"""
    ex, ser = _executor()
    ser.write(build_command(7, 'LIGHT_ON').encode('ascii'))
    assert ser.rx[-1] == 'ACK,7,OK'
    ser.write(build_command(7, 'ALL_OFF').encode('ascii'))
    assert ser.rx[-1] == 'ACK,7,DUP'
    assert ser.dev.state['light'] == 'on', 'DUP 命令不得改变状态'


def test_e2e_dup_reaches_caller_as_not_ok(bridge):
    """端到端：撞 seq 时固件回 DUP，SerialExecutor 不把它当成功播报。"""
    ex, ser = _executor()
    ex.seq = 6  # 下一条下发 seq=7，与固件已见的 7 撞车
    ser.write(build_command(7, 'LIGHT_ON').encode('ascii'))
    ser.readline()  # 吃掉第一条 ACK,7,OK

    ex.dispatch('ALL_OFF')       # seq=7 -> DUP
    got = ex.wait_ack()
    assert got['ok'] is False
    assert got['status'] == 'DUP'
    assert ser.dev.state['light'] == 'on', 'ALL_OFF 被 DUP 丢弃，灯仍亮'

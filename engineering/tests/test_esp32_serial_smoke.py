# -*- coding: utf-8 -*-
"""PC 侧 ESP32 串口验收脚本的离线测试。"""
from scripts.esp32_serial_smoke import read_expected, make_seqs


def test_make_seqs_uses_seed_and_increments():
    assert make_seqs(["STATUS_ALL", "LIGHT_ON"], 5000) == [5001, 5002]


def test_make_seqs_distinct_across_runs():
    """两次运行（不同毫秒）seq 不同且都远离 1——不会撞固件的 DUP 窗口。"""
    a = make_seqs(["STATUS_ALL"], 1_700_000_000_000)
    b = make_seqs(["STATUS_ALL"], 1_700_000_000_001)
    assert a != b
    assert all(s > 64 for s in a + b)


def test_make_seqs_wraps_uint32():
    assert make_seqs(["A", "B"], 0xFFFFFFFF) == [0, 1]


class _FakeSerial:
    def __init__(self, lines):
        self.lines = [line.encode("ascii") for line in lines]

    def readline(self):
        return self.lines.pop(0) if self.lines else b""


def test_read_expected_ack_only():
    lines = read_expected(_FakeSerial(["ACK,1,OK"]), 1, False, timeout=0.2)
    assert lines == ["ACK,1,OK"]


def test_read_expected_waits_for_state_after_ack():
    ser = _FakeSerial(["ACK,2,OK", "STATE,2,LIGHT=on,CURTAIN=closed"])
    lines = read_expected(ser, 2, True, timeout=0.2)
    assert lines == ["ACK,2,OK", "STATE,2,LIGHT=on,CURTAIN=closed"]

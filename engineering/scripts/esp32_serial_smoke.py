# -*- coding: utf-8 -*-
"""对真实 ESP32-S3 固件做串口闭环验收。

默认只发送 STATUS_ALL，避免测试时误驱动执行器；``--full`` 才依次测试
灯、窗帘和全部关闭。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.serial_protocol import build_command


def make_seqs(commands, seed_ms: int):
    """给一批命令分配 seq：以毫秒时间戳为起点，避开固件最近 64 个 DUP 记录。

    PC 进程重启而固件不断电时，若从 1 重新计数会被判 DUP 静默丢弃动作。
    用时间戳低 32 位作起点即可跨进程不撞车；同一次运行内连续递增。
    """
    base = int(seed_ms) & 0xFFFFFFFF
    return [(base + i) & 0xFFFFFFFF for i in range(1, len(commands) + 1)]


def read_line(ser, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = ser.readline()
        if line:
            return line.decode("utf-8", errors="replace").strip()
    return ""


def read_expected(ser, seq: int, expect_state: bool, timeout: float):
    lines = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = read_line(ser, max(0.05, deadline - time.monotonic()))
        if not line:
            continue
        lines.append(line)
        if line.startswith(f"ACK,{seq},"):
            if not expect_state:
                return lines
        if expect_state and line.startswith(f"STATE,{seq},"):
            return lines
    raise TimeoutError(f"seq={seq} 未收到预期回复，已收到: {lines}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="例如 COM5 或 /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--full", action="store_true",
                        help="实际执行灯/窗帘动作；默认只查询状态")
    parser.add_argument("--no-boot-wait", action="store_true")
    args = parser.parse_args(argv)

    try:
        import serial
    except ImportError:
        sys.exit("需要 pyserial：python -m pip install pyserial")

    commands = ["STATUS_ALL"]
    if args.full:
        commands = [
            "LIGHT_ON", "STATUS_ALL",
            "CURTAIN_OPEN", "STATUS_ALL",
            "ALL_OFF", "STATUS_ALL",
        ]

    with serial.Serial(args.port, args.baud, timeout=0.1) as ser:
        ser.reset_input_buffer()
        if not args.no_boot_wait:
            time.sleep(2.0)
            ser.reset_input_buffer()

        # 与 SerialExecutor 同理：用毫秒时间戳作 seq 起点，避免固件未重启时
        # 二次运行冒烟脚本被判 DUP（ACK,DUP 但动作不执行）。
        for seq, intent in zip(make_seqs(commands, int(time.time() * 1000)), commands):
            payload = build_command(seq, intent).encode("ascii")
            ser.write(payload)
            lines = read_expected(ser, seq, intent == "STATUS_ALL", args.timeout)
            print(f"{payload.decode().strip()} -> {' | '.join(lines)}")

    print("PASS: ESP32-S3 串口协议闭环")


if __name__ == "__main__":
    main()

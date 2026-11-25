# -*- coding: utf-8 -*-
"""串口客户端：把一条 WAV 发给固件，打印识别结果。

固件协议见 main/app_main.c：
    PING                     -> PONG
    INFO                     -> 模型/内存信息
    RUNRAW + <len><pcm>      -> 识别结果 JSON

用法::

    python firmware/tools/serial_client.py --port COM5 --info
    python firmware/tools/serial_client.py --port COM5 --wav a.wav
    python firmware/tools/serial_client.py --port COM5 --wav a.wav --repeat 5

依赖 pyserial：pip install pyserial
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), 'rb') as wf:
        if wf.getnchannels() != 1:
            sys.exit(f'需要单声道，当前 {wf.getnchannels()} 声道')
        if wf.getframerate() != 16000:
            sys.exit(f'需要 16 kHz，当前 {wf.getframerate()} Hz')
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


def drain(ser, wait=0.3) -> list[str]:
    """读掉启动 banner 与日志，只留下协议行。"""
    time.sleep(wait)
    lines = []
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='replace').strip()
        if line:
            lines.append(line)
    return lines


def send_line(ser, text: str) -> None:
    ser.write((text + '\n').encode('ascii'))
    ser.flush()


def read_json(ser, timeout=10.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = ser.readline().decode('utf-8', errors='replace').strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True, help='例如 COM5 或 /dev/ttyUSB0')
    parser.add_argument('--baud', type=int, default=921600)
    parser.add_argument('--wav', default=None)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--info', action='store_true')
    parser.add_argument('--ping', action='store_true')
    args = parser.parse_args(argv)

    try:
        import serial
    except ImportError:
        sys.exit('需要 pyserial：pip install pyserial')

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(0.2)
    banner = drain(ser, 0.5)
    if banner:
        print('设备输出：', ' | '.join(banner[-3:]))

    if args.ping:
        send_line(ser, 'PING')
        print('PING ->', ser.readline().decode('utf-8', 'replace').strip())

    if args.info:
        send_line(ser, 'INFO')
        info = read_json(ser)
        print('INFO ->', json.dumps(info, ensure_ascii=False) if info else '无响应')

    if args.wav:
        pcm = read_wav(Path(args.wav))
        for i in range(args.repeat):
            send_line(ser, 'RUNRAW')
            ser.write(struct.pack('<i', pcm.size))
            ser.write(pcm.astype('<f4').tobytes())
            ser.flush()
            t0 = time.time()
            result = read_json(ser, timeout=30.0)
            wall = (time.time() - t0) * 1000
            if result is None:
                print(f'[{i+1}] 无响应（{wall:.0f} ms）')
                continue
            print(f'[{i+1}] {result.get("cmd")} conf={result.get("conf")} '
                  f'dev={result.get("ms")} ms wall={wall:.0f} ms T={result.get("T")}')
    ser.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

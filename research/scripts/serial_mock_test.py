# -*- coding: utf-8 -*-
"""无硬件条件下的串口协议模拟测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.serial_protocol import MockEsp32Controller

commands = [
    'CMD,1,LIGHT_ON\n',
    'CMD,2,STATUS_ALL\n',
    'CMD,3,CURTAIN_OPEN\n',
    'CMD,4,ALL_OFF\n',
    'CMD,4,ALL_OFF\n',
    'CMD,5,FAN_ON\n',
    'BADLINE\n',
]
dev = MockEsp32Controller()
for line in commands:
    resp = dev.handle_line(line)
    print('PC>', line.strip())
    print('ESP<', resp.strip().replace('\n', ' | '))
print('最终状态:', dev.state)

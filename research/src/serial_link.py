# -*- coding: utf-8 -*-
"""真实串口适配器；无pyserial时不影响模拟设备。"""
from .serial_protocol import build_command

class SerialLink:
    def __init__(self, port, baudrate=115200, timeout=1.0):
        try:
            import serial
        except ImportError:
            raise ImportError('需要安装 pyserial: pip install pyserial')
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def execute(self, seq, intent_id):
        self.ser.reset_input_buffer()
        self.ser.write(build_command(seq, intent_id).encode('ascii'))
        line = self.ser.readline().decode('ascii', errors='replace').strip()
        if not line:
            raise TimeoutError('ESP32未返回ACK')
        return line + '\n'

    def close(self):
        self.ser.close()

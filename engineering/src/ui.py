# -*- coding: utf-8 -*-
"""控制台/OLED/网页显示。"""
import time


class ConsoleUI:
    def show(self, command, confidence, accepted, angle=None):
        state = '执行' if accepted else '拒识'
        msg = f"[{time.strftime('%H:%M:%S')}] {state} 命令={command} 置信度={confidence:.2f}"
        if angle is not None:
            msg += f" 方向={angle:.0f}deg"
        print(msg)


class OLEDUI:
    def __init__(self, i2c_port=1, address=0x3C):
        self.available = False
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306
            self.device = ssd1306(i2c(port=i2c_port, address=address))
            self.available = True
        except Exception as exc:
            print('OLED未启用:', exc)

    def show(self, command, confidence, accepted):
        if not self.available:
            return
        from luma.core.render import canvas
        with canvas(self.device) as draw:
            draw.text((0, 0), 'Dialect Voice', fill='white')
            draw.text((0, 18), f'{command}', fill='white')
            draw.text((0, 36), f'conf {confidence:.2f}', fill='white')
            draw.text((0, 54), 'OK' if accepted else 'RETRY', fill='white')

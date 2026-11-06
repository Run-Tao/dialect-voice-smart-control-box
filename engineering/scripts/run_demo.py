# -*- coding: utf-8 -*-
"""演示入口：先跑合成测试，再提示硬件步骤。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_synthetic_demo import main as synthetic_main


def main():
    print('=== DialectVoice Engineering Demo ===')
    synthetic_main()
    print('硬件演示请使用 collect_data.py 和 run_monitor.py。')


if __name__ == '__main__':
    main()


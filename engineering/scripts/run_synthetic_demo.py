# -*- coding: utf-8 -*-
"""无需硬件：合成6条命令，测试DTW模板识别。"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import sequence_feature
from src.recognition import TemplateBank


def make_command_tone(freqs, fs=16000, duration=1.2, noise=0.01):
    """把3段不同频率拼接成一条命令，模拟随时间变化的语音。"""
    seg = duration / len(freqs)
    t = np.arange(int(fs * seg)) / fs
    parts = []
    for f in freqs:
        parts.append(np.sin(2 * np.pi * f * t))
    sig = np.concatenate(parts)
    sig += noise * np.random.randn(len(sig))
    return sig


def main():
    fs = 16000
    commands = ['开灯', '关灯', '开风扇', '关风扇', '开空调', '关空调']
    freq_map = {
        '开灯': [300, 600, 900], '关灯': [300, 900, 1200],
        '开风扇': [500, 1000, 1500], '关风扇': [500, 1200, 1800],
        '开空调': [700, 1400, 2100], '关空调': [700, 1600, 2400],
    }
    bank = TemplateBank(commands, temperature=0.2, confidence_threshold=0.4)
    for c in commands:
        bank.add_global(c, sequence_feature(make_command_tone(freq_map[c]), fs))

    correct = 0
    total = 0
    for c in commands:
        for _ in range(5):
            sig = make_command_tone(freq_map[c], fs, noise=0.03)
            feat = sequence_feature(sig, fs)
            pred, conf, _ = bank.predict(feat)
            total += 1
            correct += int(pred == c)
            print(f"真实={c} 预测={pred} 置信度={conf:.2f}")
    print(f"合成数据准确率: {correct}/{total} = {correct / total:.2f}")


if __name__ == '__main__':
    main()

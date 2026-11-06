# -*- coding: utf-8 -*-
"""从 data/raw 生成全局模板库。"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import sequence_feature
from src.recognition import TemplateBank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commands', nargs='+', default=[
        '开灯', '关灯', '开风扇', '关风扇', '开空调', '关空调', '调高温度', '调低温度'
    ])
    parser.add_argument('--raw', default=r'D:\Run_Tao\DialectVoice_Engineering\data\raw')
    parser.add_argument('--out', default=r'D:\Run_Tao\DialectVoice_Engineering\data\models\templates.json')
    parser.add_argument('--fs', type=int, default=16000)
    args = parser.parse_args()

    bank = TemplateBank(args.commands)
    count = 0
    for wav in Path(args.raw).glob('*.wav'):
        name = wav.stem
        matched = None
        for c in args.commands:
            if c in name:
                matched = c
                break
        if matched is None:
            continue
        audio, sr = sf.read(str(wav), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != args.fs:
            print('跳过采样率不匹配:', wav.name)
            continue
        bank.add_global(matched, sequence_feature(audio, args.fs))
        count += 1
    bank.save(args.out)
    print(f'模板数: {count}, 已保存: {args.out}')


if __name__ == '__main__':
    main()

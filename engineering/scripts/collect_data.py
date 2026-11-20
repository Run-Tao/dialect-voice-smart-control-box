# -*- coding: utf-8 -*-
"""采集方言命令 WAV 和 CSV。"""
import argparse
import time
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio_capture import AudioCapture
from src.data_utils import save_wav, append_metadata, make_filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker', default='S1')
    parser.add_argument('--dialect', default='方言A')
    parser.add_argument('--command', required=True)
    parser.add_argument('--background', default='quiet')
    parser.add_argument('--session', default='1')
    parser.add_argument('--seconds', type=float, default=1.5)
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--out', default=r'D:\Run_Tao\DialectVoice_Engineering\data\raw')
    parser.add_argument('--csv', default=r'D:\Run_Tao\DialectVoice_Engineering\data\raw\metadata.csv')
    args = parser.parse_args()

    cap = AudioCapture(samplerate=args.fs, channels=1)
    cap.start()
    print('开始录音...')
    chunks = []
    start = time.time()
    try:
        while time.time() - start < args.seconds:
            chunks.append(cap.read())
    finally:
        cap.stop()

    audio = np.concatenate(chunks, axis=0).reshape(-1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(out_dir.glob('*.wav'))) + 1
    name = make_filename(args.speaker, args.dialect, args.command, args.background, args.session, idx)
    path = out_dir / name
    save_wav(path, audio, args.fs)
    append_metadata(args.csv, {
        'file': str(path), 'speaker': args.speaker, 'dialect': args.dialect,
        'command': args.command, 'background': args.background,
        'session': args.session, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })
    print('已保存:', path)


if __name__ == '__main__':
    main()

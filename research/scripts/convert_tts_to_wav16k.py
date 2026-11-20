# -*- coding: utf-8 -*-
"""把TTS生成文件统一转换为16kHz单声道PCM WAV。"""
from pathlib import Path
import soundfile as sf
import numpy as np

base = Path(r'D:\Run_Tao\DialectVoice_Research')
pairs = [
    (base / 'data' / 'tts_validation' / 'canto', base / 'data' / 'tts_validation' / 'canto_16k'),
    (base / 'data' / 'tts_validation' / 'edge', base / 'data' / 'tts_validation' / 'edge_16k'),
]
for src_dir, dst_dir in pairs:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob('*')):
        if p.suffix.lower() not in ('.wav', '.mp3'):
            continue
        audio, sr = sf.read(str(p), dtype='float32', always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            # 简单线性插值重采样，仅用于课程冒烟，不用于正式实验
            new_len = int(round(len(audio) * 16000 / sr))
            old_t = np.linspace(0, 1, len(audio), endpoint=False)
            new_t = np.linspace(0, 1, new_len, endpoint=False)
            audio = np.interp(new_t, old_t, audio).astype(np.float32)
        out = dst_dir / (p.stem + '.wav')
        sf.write(str(out), audio, 16000, subtype='PCM_16')
        print('converted', p.name, '->', out.name, 'sr=16000')

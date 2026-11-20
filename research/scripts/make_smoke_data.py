# -*- coding: utf-8 -*-
"""生成合成WAV和清单，仅用于代码冒烟测试，不是实验数据。"""
import csv
import wave
from pathlib import Path
import numpy as np

base = Path(r'D:\Run_Tao\DialectVoice_Research')
out_dir = base / 'data' / 'smoke_wavs'
out_dir.mkdir(parents=True, exist_ok=True)
manifest_path = base / 'data' / 'smoke_manifest.csv'

fs = 16000
duration = 1.0
t = np.arange(int(fs * duration)) / fs
commands = ['开灯', '关灯', '开风扇']
freqs = {'开灯': 300, '关灯': 500, '开风扇': 700}
rng = np.random.default_rng(0)

rows = []
for s in range(1, 4):
    speaker = f'S{s}'
    for c in commands:
        for rep in range(1, 3):
            f0 = freqs[c] * (1.0 + 0.03 * (s - 2))
            sig = 0.5 * np.sin(2 * np.pi * f0 * t)
            sig += 0.2 * np.sin(2 * np.pi * (2 * f0) * t)
            sig += 0.01 * rng.standard_normal(len(t))
            sig = np.clip(sig, -1, 1)
            pcm = (sig * 32767).astype('<i2')
            name = f'{speaker}_{c}_{rep}.wav'
            path = out_dir / name
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes(pcm.tobytes())
            rows.append({
                'utterance_id': f'{speaker}_{c}_{rep}',
                'audio_path': f'smoke_wavs/{name}',
                'speaker_id': speaker,
                'dialect': 'synthetic',
                'language': 'synthetic',
                'command': c,
                'duration': duration,
                'sample_rate': fs,
                'notes': 'synthetic smoke test only',
            })

with open(manifest_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'utterance_id', 'audio_path', 'speaker_id', 'dialect', 'language',
        'command', 'duration', 'sample_rate', 'notes'
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f'已生成 {len(rows)} 条合成音频')
print('清单:', manifest_path)

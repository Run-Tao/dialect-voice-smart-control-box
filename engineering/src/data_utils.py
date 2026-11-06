# -*- coding: utf-8 -*-
"""数据保存和元数据。"""
import csv
import time
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None


def save_wav(path, audio, fs):
    if sf is None:
        raise ImportError('请先安装 soundfile: pip install soundfile')
    sf.write(str(path), audio, fs)


def append_metadata(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['file', 'speaker', 'dialect', 'command', 'background', 'session', 'timestamp', 'notes']
    exists = csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in fields})


def make_filename(speaker, dialect, command, background, session, idx):
    safe = command.replace(' ', '')
    return f"{speaker}_{dialect}_{safe}_{background}_s{session}_{idx:02d}.wav"

# -*- coding: utf-8 -*-
"""把 483 条测试集打成一个大 PCM 包，供 C 版批量跑并统计准确率。

产物（默认 firmware/tools/batch/）：
    pcm.f32        所有样本的 float32 PCM 拼接
    index.txt      每行 name,label,numpy_cmd,offset_samples,count

C 侧 host_batch 读这两个文件，逐条 dv_run，统计
「与标签一致率」和「与 numpy 决策一致率」。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.audio_io import load_wav                # noqa: E402
from runtime.v3_model import V3Predictor             # noqa: E402

DEFAULT_MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'
DEFAULT_MANIFEST = (ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus'
                    / 'manifest_yue_v3_1_vad.csv')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--split', default='test')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'batch'))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.manifest, encoding='utf-8-sig'))
            if r['split'] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    model = V3Predictor.load(args.model)

    index_lines = []
    offset = 0
    used = 0
    with open(out / 'pcm.f32', 'wb') as pcm_f:
        for i, row in enumerate(rows, 1):
            try:
                audio, sr = load_wav(row['audio_path'])
            except Exception as exc:
                print(f'[跳过] {row["audio_path"]}: {exc}')
                continue
            if int(sr) != 16000:
                continue
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            pcm_f.write(audio.tobytes())

            try:
                pred = model.predict(row['audio_path'])
                npy_cmd = pred.command
            except Exception:
                npy_cmd = 'ERROR'

            index_lines.append(
                f'{Path(row["audio_path"]).stem},{row["command"]},{npy_cmd},'
                f'{offset},{audio.size}')
            offset += audio.size
            used += 1
            if i % 100 == 0:
                print(f'  {i}/{len(rows)} ...')

    (out / 'index.txt').write_text('\n'.join(index_lines) + '\n', encoding='utf-8')
    print(f'样本 {used} 条，PCM {offset} 点（{offset*4/1024/1024:.1f} MB）')
    print(f'写出 {out}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""为 C 版推理内核生成参考数据（float32 二进制 + 元信息）。

产物目录内容：
    pcm.f32     原始波形（float32，16 kHz 单声道）
    seq.f32     特征序列 [T, 80]（logmel_cmvn 40 + Δ 40）
    input.f32   插值到 256 帧后的网络输入 [80, 256]
    logits.f32  三种子平均的原始 logits [6]
    meta.txt    key=value（T / n_samples / command / confidence）

用法::

    python firmware/tools/dump_reference.py \
        --wav ../DialectVoice_Research/.../LIGHT_ON_01_....wav \
        --out firmware/tools/ref/light_on
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime.audio_io import load_wav                        # noqa: E402
from runtime.v3_cnn_numpy import N_FRAMES, prepare_input     # noqa: E402
from runtime.v3_model import V3Predictor                     # noqa: E402

DEFAULT_MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True)
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = V3Predictor.load(args.model)
    audio, sr = load_wav(args.wav)
    if int(sr) != 16000:
        sys.exit(f'采样率必须是 16000，当前 {sr}')
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)

    seq = model.feature_sequence(args.wav)          # [T, 80] float32
    x = prepare_input(seq, N_FRAMES)                # [1, 80, 256] float32
    logits = np.asarray(model._raw_logits(x), dtype=np.float64).reshape(-1)

    pred = model.predict_features(seq)

    audio.astype(np.float32).tofile(out / 'pcm.f32')
    np.ascontiguousarray(seq, dtype=np.float32).tofile(out / 'seq.f32')
    np.ascontiguousarray(x[0], dtype=np.float32).tofile(out / 'input.f32')
    logits.astype(np.float32).tofile(out / 'logits.f32')

    (out / 'meta.txt').write_text(
        '\n'.join([
            f'n_samples={audio.size}',
            f'T={seq.shape[0]}',
            f'command={pred.command}',
            f'confidence={pred.confidence:.9f}',
        ]) + '\n', encoding='utf-8')

    print(f'wav={Path(args.wav).name}  n={audio.size}  T={seq.shape[0]}')
    print(f'argmax={pred.command}  conf={pred.confidence:.6f}  '
          f'logits={np.round(logits, 4).tolist()}')
    print(f'写出 {out}')


if __name__ == '__main__':
    main()

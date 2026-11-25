# -*- coding: utf-8 -*-
"""把 v3 集成模型导出为 ESP32-S3 可直接编译的 C 头文件。

产物（默认写到 ``firmware/main/model/``）：

- ``dv_model_config.h``：类别名、温度、帧数、量化 scale、mel 滤波器组、hamming 窗；
- ``dv_model_data.h``：三个种子的 int8 权重（BN 已折叠进 conv）。

量化口径与 ``docs/开源智能音箱借用评估_2026-11-06.md`` §8.8 完全一致：
权重 per-output-channel 对称 int8；激活 per-tensor 对称 int8，scale 由训练集
前 N 条校准得到；卷积用 int32 累加后反量化，FC 同理。

用法::

    python firmware/tools/export_c_model.py --calib 300
    python firmware/tools/export_c_model.py --calib 0      # 复用缓存
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FIRMWARE = Path(__file__).resolve().parents[1]
ROOT = FIRMWARE.parent
sys.path.insert(0, str(ROOT))

from runtime.features import mel_filterbank                      # noqa: E402
from runtime.v3_cnn_numpy import (                               # noqa: E402
    CONV_LAYERS, IN_DIM, N_FRAMES, prepare_input,
)
from runtime.v3_model import V3Predictor                         # noqa: E402

DEFAULT_MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'
DEFAULT_MANIFEST = (ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus'
                    / 'manifest_yue_v3_1_vad.csv')
OUT_DIR = FIRMWARE / 'main' / 'model'
CACHE = FIRMWARE / 'tools' / '.act_scale_cache.json'


# ----------------------------------------------------------------------
def fold_bn(params: dict) -> dict:
    """把 BN 折叠进 conv，返回 {name: (w, b)} + fc。"""
    out = {}
    for name, _, _, _ in CONV_LAYERS:
        w = np.asarray(params[name + '.weight'], np.float64)
        b = np.asarray(params[name + '.bias'], np.float64)
        g = np.asarray(params[name + '.bn.weight'], np.float64)
        be = np.asarray(params[name + '.bn.bias'], np.float64)
        mu = np.asarray(params[name + '.bn.running_mean'], np.float64)
        va = np.asarray(params[name + '.bn.running_var'], np.float64)
        s = g / np.sqrt(va + 1e-5)
        out[name] = (w * s[:, None, None], (b - mu) * s + be)
    out['fc'] = (np.asarray(params['fc.weight'], np.float64),
                 np.asarray(params['fc.bias'], np.float64))
    return out


def quantize_per_channel(w: np.ndarray):
    """[O,C,K] -> (int8 [O,C,K], float32 scale [O])。"""
    o = w.shape[0]
    flat = w.reshape(o, -1)
    scale = np.max(np.abs(flat), axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scale.ravel().astype(np.float32)


def conv_float(x, w, b, pad):
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(x, w.shape[2], axis=2)
    return np.einsum('bctk,ock->bot', win, w, optimize=True) + b[None, :, None]


def forward_acts(fp, x):
    """返回各量化点上的激活值：[input, conv0, conv1, pooled]。"""
    ins = [x]
    for name, _, _, k in CONV_LAYERS:
        w, b = fp[name]
        x = np.maximum(conv_float(x, w, b, k // 2), 0.0)
        if name in ('conv0', 'conv1'):
            B, C, T = x.shape
            T2 = T // 2
            x = x[:, :, :T2 * 2].reshape(B, C, T2, 2).max(axis=3)
        ins.append(x)
    pooled = np.concatenate([x.mean(axis=2), x.max(axis=2)], axis=1)
    ins[-1] = pooled
    return ins


# ----------------------------------------------------------------------
def calibrate(model: V3Predictor, manifest: Path, limit: int) -> list[float]:
    """用训练集前 limit 条估激活 per-tensor scale；结果带缓存。"""
    if CACHE.exists():
        cached = json.loads(CACHE.read_text(encoding='utf-8'))
        if cached.get('n') == limit and cached.get('model') == DEFAULT_MODEL.name:
            print(f'复用激活 scale 缓存（n={limit}）')
            return [float(v) for v in cached['act_max']]

    import csv
    rows = [r for r in csv.DictReader(open(manifest, encoding='utf-8-sig'))
            if r['split'] == 'train'][:limit]
    act_max = [0.0] * 4
    used = 0
    for row in rows:
        try:
            seq = model.feature_sequence(row['audio_path'])
        except Exception:
            continue
        x = prepare_input(seq, N_FRAMES)
        for params in model._seed_params:
            for i, act in enumerate(forward_acts(fold_bn(params), x)):
                act_max[i] = max(act_max[i], float(np.max(np.abs(act))))
        used += 1
    print(f'校准完成：{used} 条，act_max={[round(v, 4) for v in act_max]}')
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(
        {'n': limit, 'model': DEFAULT_MODEL.name, 'act_max': act_max},
        ensure_ascii=False, indent=2), encoding='utf-8')
    return act_max


# ----------------------------------------------------------------------
def fmt_floats(arr, per_line=8, precision='%.6ff'):
    values = [precision % float(v) for v in np.asarray(arr).ravel()]
    lines = [', '.join(values[i:i + per_line]) for i in range(0, len(values), per_line)]
    return ',\n    '.join(lines)


def fmt_int8(arr, per_line=32):
    values = [f'{int(v):4d}' for v in np.asarray(arr).ravel()]
    lines = [', '.join(values[i:i + per_line]) for i in range(0, len(values), per_line)]
    return ',\n    '.join(lines)


def emit_config(path: Path, model: V3Predictor, act_scale, mel_fb, hamming):
    classes = ', '.join(f'"{c}"' for c in model.classes_)
    lines = [
        '/* 自动生成，勿手改：python firmware/tools/export_c_model.py */',
        '#ifndef DV_MODEL_CONFIG_H',
        '#define DV_MODEL_CONFIG_H',
        '',
        f'#define DV_N_FRAMES   {N_FRAMES}',
        f'#define DV_IN_DIM     {IN_DIM}',
        f'#define DV_N_CLASSES  {len(model.classes_)}',
        f'#define DV_N_SEEDS    {len(model._seed_params)}',
        f'#define DV_TEMPERATURE {float(model.temperature):.6f}f',
        '',
        'static const char *const DV_CLASSES[DV_N_CLASSES] = {' + classes + '};',
        '',
        '/* 激活 per-tensor 对称 int8 scale（含 1/127 已折入） */',
        'static const float DV_ACT_SCALE[4] = {',
        '    ' + fmt_floats(act_scale, 4),
        '};',
        '',
        '/* 40 x 257 的 mel 滤波器组，直接取自 numpy，避免 C 端重建产生偏差 */',
        'static const float DV_MEL_FB[40][257] = {',
    ]
    for row in mel_fb:
        lines.append('    {' + fmt_floats(row, 8) + '},')
    lines += ['};', '', 'static const float DV_HAMMING[512] = {',
              '    ' + fmt_floats(hamming, 8), '};', '',
              '#endif', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'写出 {path}  ({path.stat().st_size/1024:.0f} KB)')


def emit_data(path: Path, model: V3Predictor):
    lines = [
        '/* 自动生成，勿手改：python firmware/tools/export_c_model.py */',
        '#ifndef DV_MODEL_DATA_H',
        '#define DV_MODEL_DATA_H',
        '',
        '#include <stdint.h>',
        '',
    ]
    for si, params in enumerate(model._seed_params):
        fp = fold_bn(params)
        for name, cin, cout, k in CONV_LAYERS:
            w, b = fp[name]
            q, scale = quantize_per_channel(w)
            lines.append(f'/* seed {si} {name}: [{cout},{cin},{k}] */')
            lines.append(f'static const int8_t DV_S{si}_{name.upper()}_W[{cout * cin * k}] = {{')
            lines.append('    ' + fmt_int8(q))
            lines.append('};')
            lines.append(f'static const float DV_S{si}_{name.upper()}_SCALE[{cout}] = {{')
            lines.append('    ' + fmt_floats(scale, 6))
            lines.append('};')
            lines.append(f'static const float DV_S{si}_{name.upper()}_BIAS[{cout}] = {{')
            lines.append('    ' + fmt_floats(b, 6))
            lines.append('};')
            lines.append('')
        w, b = fp['fc']
        q, scale = quantize_per_channel(w)
        lines.append(f'/* seed {si} fc: [{b.shape[0]},{w.shape[1]}] */')
        lines.append(f'static const int8_t DV_S{si}_FC_W[{q.size}] = {{')
        lines.append('    ' + fmt_int8(q))
        lines.append('};')
        lines.append(f'static const float DV_S{si}_FC_SCALE[{q.shape[0]}] = {{')
        lines.append('    ' + fmt_floats(scale, 6))
        lines.append('};')
        lines.append(f'static const float DV_S{si}_FC_BIAS[{b.shape[0]}] = {{')
        lines.append('    ' + fmt_floats(b, 6))
        lines.append('};')
        lines.append('')
    lines += ['#endif', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'写出 {path}  ({path.stat().st_size/1024/1024:.2f} MB)')


# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(DEFAULT_MODEL))
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--calib', type=int, default=300, help='校准条数；0 表示只用缓存')
    parser.add_argument('--out', default=str(OUT_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = V3Predictor.load(args.model)
    print(f'模型 {Path(args.model).name}：{len(model._seed_params)} 种子 / '
          f'{len(model.classes_)} 类 / T={model.temperature}')

    if args.calib > 0:
        act_max = calibrate(model, Path(args.manifest), args.calib)
    else:
        if not CACHE.exists():
            sys.exit('没有激活 scale 缓存，请先跑 --calib 300')
        act_max = [float(v) for v in
                   json.loads(CACHE.read_text(encoding='utf-8'))['act_max']]
    act_scale = [max(v, 1e-6) / 127.0 for v in act_max]

    mel_fb = mel_filterbank(40, 512, 16000, fmin=50.0, fmax=8000.0)
    hamming = np.hamming(512)

    emit_config(out_dir / 'dv_model_config.h', model, act_scale, mel_fb, hamming)
    emit_data(out_dir / 'dv_model_data.h', model)

    total = sum(q.size for si in range(len(model._seed_params))
                for q in [quantize_per_channel(fold_bn(model._seed_params[si])[n][0])[0]
                          for n, _, _, _ in CONV_LAYERS])
    print(f'int8 权重（不含 FC）约 {total/1024:.0f} KB')


if __name__ == '__main__':
    main()

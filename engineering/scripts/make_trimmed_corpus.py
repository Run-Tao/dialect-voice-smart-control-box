# -*- coding: utf-8 -*-
"""按部署前端（VAD）裁剪语料，生成与线上输入分布一致的训练集。

背景：v3 模型是在「整段 WAV（含 TTS 自带的首尾静音）」上训练的，而线上麦克风
输入经 VAD 端点检测后是一段紧贴语音的音频。两者分布不一致，导致裁剪后
LIGHT_ON/LIGHT_OFF 等短命令识别率从 0.99 掉到 0.17。

解法不是在线上补静音，而是让训练输入与部署输入同源：用**同一个 VAD**
处理训练语料，得到与线上完全一致的音频分布，再重新训练。

示例：
    python scripts/make_trimmed_corpus.py \
        --manifest "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\manifest_yue_v3_1.csv" \
        --out-dir "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\wav16k_v3_1_vad" \
        --out-manifest "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\manifest_yue_v3_1_vad.csv"
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vad import EnergyVad


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def segment_with_vad(audio, sr, vad_kwargs) -> np.ndarray:
    vad = EnergyVad(fs=sr, **vad_kwargs)
    utterances = vad.feed(audio) + vad.flush()
    if not utterances:
        return audio
    return utterances[0]


def write_wav_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    """用标准库写 16bit PCM WAV，避免依赖 soundfile（树莓派上也跑得动）。"""
    import wave
    data = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (data * 32767.0).astype('<i2').tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm)


def resolve_source(raw: str, bases) -> Path:
    """清单里的 audio_path 写法不一（相对项目根、相对 data/tts_corpus），逐个试。"""
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for base in bases:
        cand = (base / p) if not p.is_absolute() else p
        if cand.exists():
            return cand
        # 只给相对项目根写的路径，再拼一次 data/tts_corpus 前缀
        cand = base / 'data' / 'tts_corpus' / p
        if cand.exists():
            return cand
    return Path(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--out-manifest', required=True)
    parser.add_argument('--audio-root', default=None,
                        help='额外的路径搜索根目录，默认用清单所在目录及其父目录')
    parser.add_argument('--pre-roll-ms', type=float, default=300.0)
    parser.add_argument('--energy-margin-db', type=float, default=9.0)
    parser.add_argument('--min-silence-ms', type=float, default=400.0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime.audio_io import load_wav

    manifest = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_manifest = Path(args.out_manifest).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bases = [manifest.parent, manifest.parent.parent, manifest.parent.parent.parent]
    if args.audio_root:
        bases.insert(0, Path(args.audio_root).resolve())

    vad_kwargs = dict(pre_roll_ms=args.pre_roll_ms,
                      energy_margin_db=args.energy_margin_db,
                      min_silence_ms=args.min_silence_ms)

    with open(manifest, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    for extra in ('duration', 'sha256'):
        if extra not in fields:
            fields.append(extra)

    kept = 0
    skipped = 0
    with open(out_manifest, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            src = resolve_source(row['audio_path'], bases)
            if not src.exists():
                skipped += 1
                continue
            dst = out_dir / (src.stem + '.wav')
            if dst.exists() and not args.overwrite:
                seg, sr = load_wav(dst)
            else:
                audio, sr = load_wav(src)
                seg = segment_with_vad(audio, sr, vad_kwargs)
                write_wav_pcm16(dst, seg, sr)
            row['audio_path'] = str(dst).replace('\\', '/')
            row['duration'] = f'{len(seg) / sr:.3f}'
            row['sha256'] = sha256_file(dst)
            writer.writerow(row)
            kept += 1

    print(f'写出 {kept} 条，跳过 {skipped} 条')
    print('音频目录:', out_dir)
    print('清单:', out_manifest)


if __name__ == '__main__':
    main()

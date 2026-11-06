# -*- coding: utf-8 -*-
"""从公开真人粤语语料构建「真实域拒识测试集」（v5 §7.2 / S0）。

为什么要这个测试集：现有 unknown 负样本是 edge-tts 合成的非命令行，
声学域与训练集同源，测出来的误接受率天然偏乐观。真人粤语语音才是真正的
域外负样本，只有它才能给出「真实域 FAR」。

用途限定（不要越界）：拒识测试、硬负样本挖掘、域差距度量。
**不用于命令正样本训练**——那需要逐条标注，且会把测试集污染成训练集。

两条纪律：
1. 同源——切段必须用部署同款 `EnergyVad` + `config.yaml` 的 vad 参数，
   否则测试集分布与线上不一致（PROGRESS.md 已知陷阱第 1 条）。
2. 许可——每个来源必须显式给出许可原文，写入 `provenance.json`。
   未核实许可的语料一旦进入课程公开报告就是分发问题，所以这里强制留痕；
   确实没查清就写 `UNVERIFIED`，构建会给出警告并在 provenance 里标记。

示例：
    python scripts/build_real_testset.py \
        --source mdcc=D:/corpora/MDCC/wav \
        --license mdcc="CC BY-NC-SA 4.0 (https://...)" \
        --out data/real_domain_testset
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import wave
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.vad import EnergyVad
from runtime.audio_io import load_wav

ROOT = Path(__file__).resolve().parents[1]
TARGET_SR = 16000
AUDIO_EXTS = ('.wav', '.flac', '.ogg', '.opus', '.mp3', '.m4a')
SPLIT = 'real_test'

# 命令类文本：命中就剔除。负样本里混进真命令会把 FAR 算低（那其实是正确接受），
# 所以「语料内容不含本项目命令」必须是核查过的，不能默认成立。
DEFAULT_EXCLUDE_PHRASES = (
    '開燈', '熄燈', '關燈', '关灯', '着燈', '開窗簾', '打開窗簾', '拉埋窗簾',
    '拉開窗簾', '關窗簾', '閂窗簾', '全部閂', '全部熄', '全部關', '設備狀態',
)
# 器件词：命中不剔除（维基句子里出现「燈塔」很正常，仍是合法负样本），
# 只计数写进 provenance，作为「负样本里有多少器件相关语境」的交代。
DEVICE_HINTS = ('燈', '灯', '窗簾', '窗帘', '狀態', '状态', '設備', '设备')


class DecodeError(RuntimeError):
    pass


def decode_audio(path: Path):
    """解码任意常见音频为 (float32 单声道, sr)。wave → soundfile → ffmpeg。"""
    if path.suffix.lower() == '.wav':
        try:
            return load_wav(path)
        except Exception:
            pass  # 非 PCM 的 wav（如 IEEE float / A-law）交给下面两条路

    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        try:
            data, sr = sf.read(str(path), dtype='float32', always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return np.ascontiguousarray(data, dtype=np.float32), int(sr)
        except Exception as exc:
            raise DecodeError(f'soundfile 解码失败: {exc}') from exc

    exe = shutil.which('ffmpeg')
    if exe is None:
        raise DecodeError(
            f'{path.suffix} 需要解码器：pip install soundfile（flac/ogg）'
            '或安装 ffmpeg 并加入 PATH（mp3/m4a，Common Voice 用这个）')
    proc = subprocess.run(
        [exe, '-v', 'error', '-i', str(path), '-f', 'f32le', '-ac', '1',
         '-ar', str(TARGET_SR), 'pipe:1'],
        capture_output=True, check=False)
    if proc.returncode != 0:
        raise DecodeError(f'ffmpeg 失败: {proc.stderr.decode("utf-8", "replace")[:200]}')
    return np.frombuffer(proc.stdout, dtype='<f4').astype(np.float32), TARGET_SR


def to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if int(sr) == TARGET_SR:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(TARGET_SR, int(sr))
    return np.asarray(resample_poly(x, TARGET_SR // g, int(sr) // g), dtype=np.float32)


def write_wav_pcm16(path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
                  * 32767.0).astype('<i2')
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return path


def qc(seg: np.ndarray):
    """电平质检：峰值、RMS dB、削波样本占比。"""
    if seg.size == 0:
        return 0.0, -120.0, 1.0
    a = np.abs(seg.astype(np.float64))
    rms = float(np.sqrt(np.mean(a ** 2)))
    return (float(a.max()),
            20.0 * math.log10(rms) if rms > 1e-9 else -120.0,
            float(np.mean(a >= 0.985)))


def parse_pairs(items, flag):
    out = {}
    for item in items or []:
        if '=' not in item:
            sys.exit(f'{flag} 需要 NAME=VALUE 形式，收到: {item}')
        name, value = item.split('=', 1)
        out[name.strip()] = value.strip()
    return out


def load_transcripts(path: Path):
    """读 TSV 转写表 → {wav 文件名: 文本}。

    自动认列：哪一列以 .wav 结尾就是文件名列，取它右边一列当文本。
    这样 FLEURS（id / 文件名 / 文本 / ...）之类的布局不用写死列号。
    """
    out = {}
    with Path(path).open(encoding='utf-8') as fh:
        for line in fh:
            cols = line.rstrip('\n').split('\t')
            idx = next((i for i, c in enumerate(cols)
                        if c.strip().lower().endswith('.wav')), None)
            if idx is None or idx + 1 >= len(cols):
                continue
            out[cols[idx].strip()] = cols[idx + 1].strip()
    return out


def iter_sources(pairs):
    for name, path in pairs.items():
        p = Path(path)
        if not p.exists():
            sys.exit(f'来源 {name} 的路径不存在: {p}')
        files = [p] if p.is_file() else sorted(
            {f for e in AUDIO_EXTS for f in p.rglob(f'*{e}')})
        if not files:
            print(f'[warn] 来源 {name} 下没有音频文件: {p}')
        yield name, p, files


def build_source(name, files, cfg, args, out_dir, manifest_rows, reasons, texts):
    vad_kwargs = cfg.vad_kwargs()
    exclude = () if args.no_text_filter else args.exclude_phrases
    scanned = 0
    kept_files = 0
    for src in files:
        if args.limit and scanned >= args.limit:
            break
        scanned += 1
        text = texts.get(src.name, '')
        if texts and not text:
            reasons['transcript_not_found'] += 1
        if text and any(p in text for p in exclude):
            reasons['command_like_text'] += 1
            if args.verbose:
                print(f'[skip] {src.name}: 转写含命令短语 → {text[:40]}')
            continue
        try:
            audio, sr = decode_audio(src)
        except DecodeError as exc:
            reasons['decode_error'] += 1
            if args.verbose:
                print(f'[skip] {src.name}: {exc}')
            continue
        except Exception as exc:
            reasons['decode_error'] += 1
            if args.verbose:
                print(f'[skip] {src.name}: {type(exc).__name__}: {exc}')
            continue
        audio = to_16k(np.asarray(audio, dtype=np.float32).reshape(-1), sr)
        if audio.size / TARGET_SR > args.max_file_sec:
            reasons['file_too_long'] += 1
            continue

        vad = EnergyVad(fs=TARGET_SR, **vad_kwargs)
        segments = vad.feed(audio) + vad.flush()
        if not segments:
            reasons['no_speech'] += 1
            continue
        kept_files += 1
        for seg_i, seg in enumerate(segments):
            ms = len(seg) / TARGET_SR * 1000.0
            if ms < args.min_ms:
                reasons['too_short'] += 1
                continue
            if ms > args.max_ms:
                reasons['too_long'] += 1
                continue
            peak, rms_db, clip_ratio = qc(seg)
            if rms_db < args.min_rms_db:
                reasons['too_quiet'] += 1
                continue
            if args.drop_clipped and (peak >= args.clip_peak
                                      or clip_ratio > args.max_clip_ratio):
                reasons['clipped'] += 1
                continue

            out_name = f'real_{name}_{len(manifest_rows):06d}.wav'
            out_path = out_dir / name / out_name
            if not args.dry_run:
                write_wav_pcm16(out_path, seg)
            manifest_rows.append({
                'audio_path': str(out_path),
                'split': SPLIT,
                'source_corpus': name,
                'src_file': src.name,
                'seg_index': seg_i,
                'duration_ms': round(ms, 1),
                'peak': round(peak, 4),
                'rms_db': round(rms_db, 2),
                'clip_ratio': round(clip_ratio, 5),
                'text': text,
                'device_hint': int(bool(text) and any(h in text for h in DEVICE_HINTS)),
            })
    return scanned, kept_files


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='append', metavar='NAME=PATH',
                        help='语料来源，可重复；PATH 为目录或单个音频文件')
    parser.add_argument('--license', action='append', metavar='NAME=TEXT',
                        help='来源许可原文（含链接），可重复；未核实写 UNVERIFIED')
    parser.add_argument('--transcripts', action='append', metavar='NAME=PATH.tsv',
                        help='来源转写表（TSV，自动认文件名列），可重复；'
                             '有转写才能核查「内容不含本项目命令」')
    parser.add_argument('--exclude-phrases', default=','.join(DEFAULT_EXCLUDE_PHRASES),
                        help='转写命中即剔除的短语，逗号分隔')
    parser.add_argument('--no-text-filter', action='store_true',
                        help='关掉命令短语剔除（语料无转写时只能这样，需在报告标注未核查）')
    parser.add_argument('--out', default=str(ROOT / 'data' / 'real_domain_testset'))
    parser.add_argument('--config', default=None)
    parser.add_argument('--min-ms', type=float, default=200.0,
                        help='默认等于 VAD 的 min_speech_ms：VAD 能吐出来的都算，'
                             '不按长度挑样本')
    parser.add_argument('--max-ms', type=float, default=15000.0)
    parser.add_argument('--min-rms-db', type=float, default=-120.0,
                        help='默认关闭。按响度筛会把「真实但偏轻」的负样本剔掉，'
                             '那会让 FAR 偏低——QC 数值照样写进 manifest，要切事后切')
    parser.add_argument('--clip-peak', type=float, default=0.999)
    parser.add_argument('--max-clip-ratio', type=float, default=0.01)
    parser.add_argument('--drop-clipped', action='store_true',
                        help='剔除削波片段；默认不剔（削波是真实录音的常态，不是坏数据）')
    parser.add_argument('--max-file-sec', type=float, default=1800.0,
                        help='超过这个时长的单个文件跳过（整文件解码会吃内存）')
    parser.add_argument('--limit', type=int, default=0, help='每个来源最多处理多少个文件')
    parser.add_argument('--dry-run', action='store_true', help='只统计不写盘')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    if not args.source:
        sys.exit('需要至少一个 --source NAME=PATH')
    sources = parse_pairs(args.source, '--source')
    licenses = parse_pairs(args.license, '--license')
    for name in sources:
        if name not in licenses:
            sys.exit(f'来源 {name} 缺少 --license；许可未核实的语料不得进入公开报告。'
                     f'确实没查清就写 --license {name}=UNVERIFIED')
        if licenses[name].upper().startswith('UNVERIFIED'):
            print(f'[warn] 来源 {name} 许可标记为 UNVERIFIED，'
                  '产出只能内部评估，不得写入对外材料')

    cfg = load_config(args.config)
    args.exclude_phrases = tuple(p.strip() for p in args.exclude_phrases.split(',')
                                 if p.strip())
    transcript_pairs = parse_pairs(args.transcripts, '--transcripts')
    transcripts = {}
    for name, tpath in transcript_pairs.items():
        if name not in sources:
            sys.exit(f'--transcripts 的来源 {name} 没有对应的 --source')
        transcripts[name] = load_transcripts(Path(tpath))
        print(f'[{name}] 转写条目 {len(transcripts[name])}')
    if not transcripts and not args.no_text_filter:
        print('[warn] 没有 --transcripts，无法核查语料内容是否含命令；'
              '「内容不含本项目命令」在报告里只能标为未核查')

    out_dir = Path(args.out)
    manifest_rows: list = []
    reasons: Counter = Counter()
    per_source = {}
    for name, path, files in iter_sources(sources):
        n_before = len(manifest_rows)
        scanned, used = build_source(name, files, cfg, args, out_dir,
                                     manifest_rows, reasons,
                                     transcripts.get(name, {}))
        per_source[name] = {
            'path': str(path),
            'license': licenses[name],
            'transcripts': str(transcript_pairs[name]) if name in transcript_pairs else None,
            'files_scanned': scanned,
            'files_with_speech': used,
            'segments': len(manifest_rows) - n_before,
        }
        print(f'[{name}] 扫描 {scanned} 文件 / 有效 {used} → 片段 '
              f'{per_source[name]["segments"]}')

    manifest_path = out_dir / 'manifest.csv'
    provenance = {
        'built': date.today().isoformat(),
        'split': SPLIT,
        'sample_rate': TARGET_SR,
        'vad_kwargs': cfg.vad_kwargs(),
        'filters': {
            'min_ms': args.min_ms, 'max_ms': args.max_ms,
            'min_rms_db': args.min_rms_db, 'clip_peak': args.clip_peak,
            'max_clip_ratio': args.max_clip_ratio,
            'drop_clipped': args.drop_clipped,
            'max_file_sec': args.max_file_sec,
            'text_filter': (not args.no_text_filter) and bool(transcripts),
            'exclude_phrases': [] if args.no_text_filter else list(args.exclude_phrases),
        },
        'selection_note': '默认不按响度/时长/削波挑样本，只剔除解不出来的和 VAD 判无语音的；'
                          '挑样本会把「真实但难听」的负样本剔掉，导致 FAR 偏低。'
                          'QC 数值全部写进 manifest，需要时在评估侧切片。',
        'dry_run': args.dry_run,
        'sources': per_source,
        'segments': len(manifest_rows),
        'total_duration_s': round(sum(r['duration_ms'] for r in manifest_rows) / 1000.0, 1),
        'segments_with_device_hint': sum(r['device_hint'] for r in manifest_rows),
        'segments_text_verified': sum(1 for r in manifest_rows if r['text']),
        'excluded': dict(sorted(reasons.items())),
        'usage': '仅用于拒识测试 / 硬负样本挖掘 / 域差距度量，不用于命令正样本训练',
    }
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('w', encoding='utf-8-sig', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                'audio_path', 'split', 'source_corpus', 'src_file', 'seg_index',
                'duration_ms', 'peak', 'rms_db', 'clip_ratio', 'text', 'device_hint'])
            writer.writeheader()
            writer.writerows(manifest_rows)
        (out_dir / 'provenance.json').write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({'segments': provenance['segments'],
                      'total_duration_s': provenance['total_duration_s'],
                      'excluded': provenance['excluded'],
                      'per_source': {k: v['segments'] for k, v in per_source.items()}},
                     ensure_ascii=False, indent=2))
    if not args.dry_run:
        print('manifest:', manifest_path)
        print('provenance:', out_dir / 'provenance.json')
    return provenance


if __name__ == '__main__':
    main()

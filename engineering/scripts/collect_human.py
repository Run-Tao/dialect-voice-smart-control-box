# -*- coding: utf-8 -*-
"""真人命令正样本采集（会话式，按部署同款 VAD 裁剪，产出评估兼容 manifest）。

背景（R-14 / 项目审查 2026-11-19）：
    全部命令识别数字来自 edge-tts 合成粤语，真人命令正样本 0 条。
    测试报告 §10 预留了「真人命令语料结果」小节，但采集脚本 P2 交付一直为空。
    本脚本把采集工具链补齐：一次运行录完一个说话人的 6 命令 × N 次，
    用与线上一致的 EnergyVad 裁剪，生成 evaluate_pipeline.py 可直接吃的 manifest。

用法（需要真实麦克风 + sounddevice）：
    # 1. 先看麦克风设备号
    python scripts/collect_human.py --list-devices

    # 2. 录一个说话人（S1），每命令 20 次，设备 0
    python scripts/collect_human.py --speaker S1 --device 0 --repeats 20

    # 3. 换人重复：--speaker S2 / S3 ...（同一 out 目录，manifest 追加）

    # 4. 全部录完后按部署 VAD 裁剪（可单独重跑，不重新录音）
    python scripts/collect_human.py --trim-only

    # 5. 评估（真人命令正样本，合成口径之外的新锚点）
    python scripts/evaluate_pipeline.py \
      --manifest data/human/manifest_human_vad.csv --subset test \
      --out docs/pipeline_eval_human.json

关键约定：
    - 只采集「命令正样本」（真人说 6 条命令）；真人非命令负样本已由 FLEURS 公开语料解决。
    - 录音是固定秒数窗口（--seconds），随后用 VAD 切出紧贴语音的片段；
      这与训练语料同源（make_trimmed_corpus.py 用同一 VAD 参数）。
    - manifest 里 is_synthetic=false、speaker_id=真人编号，方便与合成语料区分。
    - 音频写 16 kHz / 16 bit PCM WAV，与部署特征前端一致。

注意：脚本本身不发声，需要真人按提示逐条朗读；AI 侧只能把工具链备齐。
"""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COMMANDS = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']
# 与 config.yaml / make_trimmed_corpus.py 一致的部署 VAD 口径
VAD_KWARGS = dict(pre_roll_ms=300.0, energy_margin_db=9.0, min_silence_ms=400.0)

MANIFEST_FIELDS = [
    'utterance_id', 'audio_path', 'speaker_id', 'dialect', 'command', 'language',
    'text', 'is_synthetic', 'sample_rate', 'duration', 'split', 'source',
    'expression_type', 'text_verified_by', 'sha256', 'notes',
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_wav_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    import wave
    data = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (data * 32767.0).astype('<i2').tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm)


def segment_with_vad(audio, sr):
    from src.vad import EnergyVad
    vad = EnergyVad(fs=sr, **VAD_KWARGS)
    segs = vad.feed(audio) + vad.flush()
    if not segs:
        return None
    # 取最长的一段（真人朗读单命令；多段说明停顿过长，记 notes）
    return max(segs, key=len), len(segs)


def list_devices():
    try:
        import sounddevice as sd
    except ImportError:
        print('请先安装: pip install sounddevice')
        return
    print(sd.query_devices())


def record_one(cap, seconds, countdown=3):
    """录一段固定秒数音频，带倒计提示。返回 float32 数组。"""
    import sounddevice as sd
    for i in range(countdown, 0, -1):
        print(f'  {i}...', flush=True)
        time.sleep(1)
    print('  ● 现在说', flush=True)
    q = cap['queue']
    # 清空残留
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break
    chunks = []
    start = time.time()
    while time.time() - start < seconds:
        try:
            chunks.append(q.get(timeout=1.0))
        except Exception:
            pass
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0).reshape(-1)


def make_capture(sr, device):
    import sounddevice as sd
    import queue
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print('audio status:', status)
        q.put(indata.copy())

    stream = sd.InputStream(samplerate=sr, channels=1, blocksize=1024,
                            device=device, callback=callback)
    stream.start()
    return {'stream': stream, 'queue': q}


def collect(args):
    sr = args.fs
    raw_dir = Path(args.out) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out) / args.manifest

    cap = make_capture(sr, args.device)
    print(f'说话人 {args.speaker}｜设备 {args.device}｜每命令 {args.repeats} 次｜6 条命令')
    print('提示文本见下方；请按屏幕提示朗读对应粤语命令。按 Ctrl+C 可中断（已录的会保留）。')
    try:
        for cmd in COMMANDS:
            for rep in range(1, args.repeats + 1):
                print(f'\n[{cmd} {rep}/{args.repeats}] 「{args.prompt.get(cmd, cmd)}」')
                audio = record_one(cap, args.seconds, args.countdown)
                if audio is None or np.max(np.abs(audio)) < 1e-4:
                    print('  ⚠ 未采到有效音频，跳过本条（可稍后重录）')
                    continue
                uid = f'{args.speaker}_{cmd}_{rep:03d}'
                path = raw_dir / f'{uid}.wav'
                write_wav_pcm16(path, audio, sr)
                print(f'  已存 {path.name}')
    except KeyboardInterrupt:
        print('\n已中断，已录音频保留。')
    finally:
        cap['stream'].stop()
        cap['stream'].close()
    print(f'\n原始音频目录: {raw_dir}')
    print('下一步: 加 --trim-only 用部署 VAD 裁剪并生成 manifest。')


def trim_only(args):
    from runtime.audio_io import load_wav
    raw_dir = Path(args.out) / 'raw'
    vad_dir = Path(args.out) / 'vad'
    vad_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out) / args.manifest

    rows = []
    skipped = 0
    for wav in sorted(raw_dir.glob('*.wav')):
        uid = wav.stem
        # 文件名格式: SPEAKER_COMMAND_REP，命令可能含下划线，取中间段
        parts = uid.split('_')
        rep = parts[-1]
        speaker = parts[0]
        cmd = '_'.join(parts[1:-1])
        if cmd not in COMMANDS:
            # 兼容说话人编号含下划线的情况：从末尾反查命令
            for c in sorted(COMMANDS, key=len, reverse=True):
                if uid.startswith(speaker + '_' + c + '_'):
                    cmd = c
                    rep = uid[len(speaker + '_' + c + '_'):]
                    break
        if cmd not in COMMANDS:
            skipped += 1
            continue

        audio, sr = load_wav(str(wav))
        seg = segment_with_vad(audio, sr)
        if seg is None:
            skipped += 1
            continue
        seg_audio, nseg = seg
        dst = vad_dir / wav.name
        write_wav_pcm16(dst, seg_audio, sr)
        rows.append({
            'utterance_id': f'{uid}_vad',
            'audio_path': str(dst.resolve()).replace('\\', '/'),
            'speaker_id': speaker,
            'dialect': 'yue-HK',
            'command': cmd,
            'language': 'yue',
            'text': args.prompt.get(cmd, ''),
            'is_synthetic': 'false',
            'sample_rate': str(sr),
            'duration': f'{len(seg_audio) / sr:.3f}',
            'split': 'human_test',
            'source': 'human_recorded',
            'expression_type': 'human_natural',
            'text_verified_by': 'speaker',
            'sha256': sha256_file(dst),
            'notes': f'raw={wav.name};vad_segments={nseg}',
        })

    # 追加已有 manifest（多次 --trim-only 不丢历史）
    existing = {}
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                existing[r['utterance_id']] = r
    for r in rows:
        existing[r['utterance_id']] = r

    with open(manifest_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for uid in sorted(existing):
            writer.writerow({k: existing[uid].get(k, '') for k in MANIFEST_FIELDS})

    print(f'VAD 裁剪写出 {len(rows)} 条，跳过 {skipped} 条；manifest 共 {len(existing)} 条')
    print(f'音频目录: {vad_dir}')
    print(f'清单: {manifest_path}')
    print('\n评估命令:')
    print(f'  python scripts/evaluate_pipeline.py --manifest "{manifest_path}" '
          f'--subset human_test --out docs/pipeline_eval_human.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list-devices', action='store_true', help='列出麦克风设备')
    parser.add_argument('--trim-only', action='store_true', help='只做 VAD 裁剪+manifest，不录音')
    parser.add_argument('--speaker', default='S1', help='说话人编号（如 S1/S2/S3）')
    parser.add_argument('--device', default=None, help='麦克风设备号（--list-devices 查）')
    parser.add_argument('--repeats', type=int, default=20, help='每条命令重复次数')
    parser.add_argument('--seconds', type=float, default=2.0, help='每次录音窗口秒数')
    parser.add_argument('--countdown', type=int, default=2, help='开录前倒计秒数')
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--out', default=str(ROOT / 'data' / 'human'))
    parser.add_argument('--manifest', default='manifest_human.csv')
    args = parser.parse_args()

    # 每条命令的提示文本（真人按此朗读；粤语自然表达，与合成语料同义）
    args.prompt = {
        'LIGHT_ON': '開燈',
        'LIGHT_OFF': '熄燈',
        'CURTAIN_OPEN': '打開窗簾',
        'CURTAIN_CLOSE': '拉埋窗簾',
        'ALL_OFF': '全部關晒',
        'STATUS_ALL': '而家咩狀態',
    }

    if args.list_devices:
        list_devices()
        return
    if args.trim_only:
        trim_only(args)
        return
    collect(args)


if __name__ == '__main__':
    main()

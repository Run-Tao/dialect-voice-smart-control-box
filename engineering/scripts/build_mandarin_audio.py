# -*- coding: utf-8 -*-
"""生成播放给智能音箱的普通话指令音频，以及反馈类音频。

需要 edge-tts，跑在隔离环境里：
    D:\\Run_Tao\\DialectVoice_Research\\.venv_tts\\Scripts\\python.exe scripts\\build_mandarin_audio.py

输出为 16kHz 单声道 PCM WAV，放在 data/audio/mandarin/，
文件名与模型类别一致（如 LIGHT_ON.wav）。

反馈类音频（S4(b)/S5）也在这里生成：

- ``ACK_DONE.wav``        动作已下发并收到回执（播「已完成」）；
- ``ACK_FAIL.wav``        动作下发失败；
- ``REJECT_UNKNOWN.wav``  拒识后播报可用命令；
- ``CLARIFY_<A>_<B>.wav`` 黄区澄清的二选一问句；
- ``PRETRIGGER.wav``      Pass-1 预触发提示音（numpy 合成的短促哔声，
  不走 TTS——它要的是「立刻有反馈」，等 TTS 合成反而更慢）。
"""
import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.commands import COMMANDS, spoken_text
from src.speaker_out import FEEDBACK_TEXT

ENGINEERING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ENGINEERING_ROOT / 'data' / 'audio' / 'mandarin'
TARGET_SR = 16000

# 预触发提示音：880 Hz、120 ms 的短促哔声，音量压低（只是给个反馈）。
PRETRIGGER_FREQ = 880.0
PRETRIGGER_MS = 120


async def synthesize(text: str, out_path: Path, voice: str, rate: str, volume: str):
    """合成普通话音频并转成 16k 单声道 WAV，返回时长（秒）。"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
    with tempfile.TemporaryDirectory() as tmp:
        mp3 = Path(tmp) / 'tts.mp3'
        await communicate.save(str(mp3))
        return to_wav16k(mp3, out_path)


def to_wav16k(src: Path, dst: Path) -> float:
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(src), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(sr) != TARGET_SR:
        new_len = int(round(len(audio) * TARGET_SR / sr))
        old_t = np.linspace(0, 1, len(audio), endpoint=False)
        new_t = np.linspace(0, 1, new_len, endpoint=False)
        audio = np.interp(new_t, old_t, audio).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, TARGET_SR, subtype='PCM_16')
    return len(audio) / TARGET_SR


def make_pretrigger(out_path: Path) -> float:
    """生成预触发提示音：短促哔声 + 头尾各 5 ms 静音（避免爆音）。"""
    import numpy as np
    import soundfile as sf
    n = int(TARGET_SR * PRETRIGGER_MS / 1000)
    t = np.arange(n) / TARGET_SR
    tone = 0.25 * np.sin(2 * np.pi * PRETRIGGER_FREQ * t)
    edge = int(TARGET_SR * 0.005)
    fade = np.linspace(0.0, 1.0, edge)
    tone[:edge] *= fade
    tone[-edge:] *= fade[::-1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), tone.astype(np.float32), TARGET_SR, subtype='PCM_16')
    return len(tone) / TARGET_SR


def trim_asset(path: Path) -> float:
    """裁掉 TTS 默认的首尾静音，只留 50 ms 余量，返回裁后时长（秒）。

    edge-tts 的输出固定带 ~170 ms 前导静音 + ~540 ms 尾部静音。
    尾部那段静音对「对话完成延迟」是实打实的预算占用（播完才算完），
    所以反馈类音频要裁。命令类音频沿用已发布的版本，不动——
    ``docs/bench_latency.json`` 里的 speak_ms 是按未裁版本报的。
    """
    import numpy as np
    import soundfile as sf
    from src.vad import trim_silence
    audio, sr = sf.read(str(path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    trimmed, _, _ = trim_silence(audio, sr, margin_db=40.0, pad_ms=50.0)
    sf.write(str(path), trimmed.astype(np.float32), sr, subtype='PCM_16')
    return len(trimmed) / float(sr)


def feedback_texts() -> dict:
    """反馈类音频的「文件名 -> 文本」。"""
    out = {
        'ACK_DONE': FEEDBACK_TEXT['ACK_DONE'],
        'ACK_FAIL': FEEDBACK_TEXT['ACK_FAIL'],
        'REJECT_UNKNOWN': FEEDBACK_TEXT['REJECT_UNKNOWN'],
    }
    # 澄清问句：按 runtime/dialog.py 的候选排序规则生成文件名，
    # 文本用同一套 COMMAND_ZH，两边必须一致（tests/test_dialog.py 有断言）。
    from runtime.dialog import COMMAND_ZH, clarify_asset
    for pair in (('CURTAIN_CLOSE', 'CURTAIN_OPEN'),):
        names = [COMMAND_ZH.get(c, c) for c in pair]
        out[clarify_asset(pair).replace('.wav', '')] = f'你是要{names[0]}，还是{names[1]}？'
    return out


async def main_async(args):
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        sys.exit('缺少 edge-tts，请用 .venv_tts 的 python 运行本脚本')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [(c, spoken_text(c, args.wake_word)) for c in COMMANDS]
    feedback = feedback_texts()
    todo += list(feedback.items())

    for name, text in todo:
        dst = out_dir / f'{name}.wav'
        if dst.exists() and not args.force:
            print(f'跳过（已存在）: {dst.name}')
            continue
        duration = await synthesize(text, dst, args.voice, args.rate, args.volume)
        if name in feedback:
            duration = trim_asset(dst)
        print(f'{name}: "{text}" -> {dst.name}  {duration:.2f}s')

    pre = out_dir / 'PRETRIGGER.wav'
    if pre.exists() and not args.force:
        print(f'跳过（已存在）: {pre.name}')
    else:
        duration = make_pretrigger(pre)
        print(f'PRETRIGGER: 哔声 -> {pre.name}  {duration:.2f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--voice', default='zh-CN-XiaoxiaoNeural')
    parser.add_argument('--rate', default='+0%')
    parser.add_argument('--volume', default='+0%')
    parser.add_argument('--wake-word', default='',
                        help='可选的唤醒词前缀，例如「小爱同学」；留空表示只播指令')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()

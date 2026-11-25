# -*- coding: utf-8 -*-
"""现场演示入口：大字卡 + 连续回放 + 麦克风模式。

v5 方案里把它写成「现场演示入口（连续监听 + 大字显示 + 回放模式）」，此前
这个文件只有 17 行、实质是 ``run_synthetic_demo`` 的包装。这里重写为真正
可拍的演示编排层：

- 不重复实现识别链路，复用 ``src.realtime.DialectEngine``（与 run_replay 同一条路径）；
- 每条结果渲染成终端大字卡，适合录屏 / 投影；
- 麦克风模式不 import ``run_monitor`` 的内部函数，只做子进程转发，
  避免它重构时把演示入口一起带崩。

用法::

    python scripts/run_demo.py --list
    python scripts/run_demo.py --mode demo                 # 六类命令各一条
    python scripts/run_demo.py --mode replay --audio <wav|dir> --limit 5
    python scripts/run_demo.py --mode mic                  # 转交 run_monitor.py

注意：演示默认用 ``LoggingExecutor`` + mock 播报，不碰真实硬件；
要接 ESP32 / 继电器请直接跑 ``run_monitor.py``。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.commands import COMMANDS, zh_text          # noqa: E402
from src.config import load_config                  # noqa: E402
from src.control import LoggingExecutor             # noqa: E402
from src.speaker_out import MandarinSpeaker         # noqa: E402

# 优先用与部署同分布的 VAD 裁剪语料（目录名是 v3_1_vad，别写成 v3_vad）；
# 找不到再退回未裁剪目录
CORPUS_CANDIDATES = [
    ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'wav16k_v3_1_vad',
    ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'wav16k_v3',
]

BOX_WIDTH = 62


# ----------------------------------------------------------------------
# 终端排版
# ----------------------------------------------------------------------
def _char_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1


def _text_width(text: str) -> int:
    return sum(_char_width(ch) for ch in text)


def _center(text: str, width: int = BOX_WIDTH) -> str:
    pad = max(width - _text_width(text), 0)
    left = pad // 2
    return ' ' * left + text + ' ' * (pad - left)


def render_card(command: str, confidence: float, accepted: bool,
                elapsed_ms: float) -> str:
    """把一条识别结果渲染成大写字卡。"""
    label = zh_text(command) if command else '未识别'
    verdict = '已执行' if accepted else '已拒识'
    detail = f'置信度 {confidence:.3f}   {verdict}   {elapsed_ms:.0f} ms'
    line = '─' * BOX_WIDTH
    return '\n'.join([
        f'┌{line}┐',
        f'│{_center("")}│',
        f'│{_center(label)}│',
        f'│{_center("")}│',
        f'│{_center(detail)}│',
        f'│{_center("")}│',
        f'└{line}┘',
    ])


# ----------------------------------------------------------------------
# 语料
# ----------------------------------------------------------------------
def corpus_dir() -> Path | None:
    for path in CORPUS_CANDIDATES:
        if path.is_dir():
            return path
    return None


def collect_files(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(target.rglob('*.wav'))
    # 单文件也要确认存在，否则会把 FileNotFoundError 抛到 load_wav 里
    return [target] if target.exists() else []


def one_per_command() -> list[tuple[str, Path]]:
    """六类命令各挑一条样本，顺序按 COMMANDS 定义。"""
    base = corpus_dir()
    if base is None:
        return []
    picked: list[tuple[str, Path]] = []
    for command in COMMANDS:
        hits = sorted(base.rglob(f'{command}_*.wav'))
        if hits:
            picked.append((command, hits[0]))
    return picked


def print_inventory() -> None:
    base = corpus_dir()
    if base is None:
        print('没找到语料目录，请用 --audio 指定 WAV 或目录。')
        print('候选路径：')
        for path in CORPUS_CANDIDATES:
            print('  -', path)
        return
    print(f'语料目录：{base}')
    for command in COMMANDS:
        hits = sorted(base.rglob(f'{command}_*.wav'))
        print(f'  {command:14s} {zh_text(command):8s} {len(hits):4d} 条')


# ----------------------------------------------------------------------
# 识别链路
# ----------------------------------------------------------------------
def build_engine(cfg, args):
    from src.realtime import DialectEngine
    from src.vad import EnergyVad

    speaker = None if args.no_speak else MandarinSpeaker(
        audio_dir=cfg.broadcast_dir, wake_word=cfg.wake_word, mock=True)
    return DialectEngine.load(
        str(cfg.model_path), str(cfg.thresholds_path),
        executor=LoggingExecutor(),
        speaker=speaker,
        vad=EnergyVad(fs=cfg.sample_rate, **cfg.vad_kwargs()),
        threshold=cfg.threshold,
        gate_path=str(cfg.selective_gate_path) if cfg.selective_gate_path else None,
        pool_path=str(cfg.selective_pool_path) if cfg.selective_pool_path else None,
        selective=cfg.selective_enabled,
        fallback_mode=cfg.selective_fallback_mode,
    )


def run_files(engine, files, fs: int, no_vad: bool) -> dict:
    from runtime.audio_io import load_wav

    accepted = 0
    elapsed_all = []
    for path in files:
        audio, sr = load_wav(path)
        if no_vad:
            segments = [audio]
        else:
            from src.vad import EnergyVad
            vad = EnergyVad(fs=sr)
            segments = vad.feed(audio) + vad.flush()
        if not segments:
            print(f'[跳过] {path.name}：VAD 没切出语音段')
            continue
        t0 = time.perf_counter()
        result = engine.process_utterance(segments[0], sr)
        elapsed = (time.perf_counter() - t0) * 1000.0
        elapsed_all.append(elapsed)
        accepted += int(result.accepted)
        print(f'\n▶ {path.name}')
        print(render_card(getattr(result, 'command', ''),
                          float(getattr(result, 'confidence', 0.0)),
                          bool(result.accepted), elapsed))

    total = len(elapsed_all)
    if total:
        mean_ms = sum(elapsed_all) / total
        p95 = sorted(elapsed_all)[min(total - 1, int(total * 0.95))]
        print(f'\n合计 {total} 条：接受 {accepted}（{accepted / total:.0%}），'
              f'平均 {mean_ms:.0f} ms，P95 {p95:.0f} ms')
    return {'total': total, 'accepted': accepted}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='DialectVoice 演示入口（大字卡 + 回放 / 麦克风）')
    parser.add_argument('--mode', choices=['demo', 'replay', 'mic'],
                        default='demo', help='demo=六类各一条；replay=指定音频；mic=转交 run_monitor')
    parser.add_argument('--audio', default=None, help='replay 模式的 WAV 文件或目录')
    parser.add_argument('--limit', type=int, default=0, help='replay 模式最多跑几条')
    parser.add_argument('--no-vad', action='store_true', help='整段送入，不做端点检测')
    parser.add_argument('--no-speak', action='store_true', help='不播报（只显示大字卡）')
    parser.add_argument('--list', action='store_true', help='列出各命令可用样本数后退出')
    parser.add_argument('--config', default=None, help='配置文件，默认项目根 config.yaml')
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, passthrough = build_parser().parse_known_args(argv)

    if args.list:
        print_inventory()
        return 0

    if args.mode == 'mic':
        # 不 import run_monitor 的内部函数：它正在被并行修改，子进程转发更稳
        cmd = [sys.executable, str(SCRIPTS / 'run_monitor.py'), *passthrough]
        if args.config:
            cmd += ['--config', args.config]
        print('转交实时监听：', ' '.join(cmd))
        return subprocess.call(cmd)

    cfg = load_config(args.config)

    if args.mode == 'replay':
        if not args.audio:
            print('replay 模式需要 --audio <wav|dir>', file=sys.stderr)
            return 2
        files = collect_files(Path(args.audio))
        if args.limit:
            files = files[:args.limit]
        if not files:
            print(f'没有找到 WAV: {args.audio}', file=sys.stderr)
            return 2
    else:
        picked = one_per_command()
        if not picked:
            print('没找到演示语料，请改用 --mode replay --audio <路径>', file=sys.stderr)
            return 2
        files = [path for _, path in picked]

    engine = build_engine(cfg, args)
    gate = '双信号' if getattr(engine, 'gate', None) is not None else '单阈值'
    print(f'模型 {Path(cfg.model_path).name}｜拒识 {gate}｜播报 '
          f'{"关" if args.no_speak else "mock"}｜共 {len(files)} 条')
    run_files(engine, files, cfg.sample_rate, args.no_vad)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

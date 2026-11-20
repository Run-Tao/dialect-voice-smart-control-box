# -*- coding: utf-8 -*-
"""实时识别入口：麦克风 -> VAD -> 分类 -> 拒识 -> 执行 -> 普通话播报。

示例：
    python scripts/run_monitor.py --executor mock_esp32
    python scripts/run_monitor.py --executor serial --port COM5
    python scripts/run_monitor.py --no-speak --executor log
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config, load_config
from src.control import LoggingExecutor, MockEsp32Executor, RelayExecutor, SerialExecutor
from src.realtime import DialectEngine
from src.speaker_out import MandarinSpeaker
from src.vad import EnergyVad

ROOT = Path(__file__).resolve().parents[1]


def build_executor(args):
    if args.executor == 'log':
        return LoggingExecutor(log_path=args.log)
    if args.executor == 'mock_esp32':
        return MockEsp32Executor(log_path=args.log)
    if args.executor == 'relay':
        return RelayExecutor(pin=args.relay_pin, mock=args.mock_relay, log_path=args.log)
    if args.executor == 'serial':
        if not args.port:
            sys.exit('--executor serial 需要 --port，例如 COM5 或 /dev/ttyUSB0')
        return SerialExecutor(port=args.port, log_path=args.log)
    raise ValueError(args.executor)


def build_parser(cfg: Config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='配置文件，默认项目根 config.yaml')
    parser.add_argument('--fs', type=int, default=cfg.sample_rate)
    parser.add_argument('--threshold', type=float, default=cfg.threshold)
    parser.add_argument('--executor', choices=['log', 'mock_esp32', 'relay', 'serial'],
                        default=cfg.executor)
    parser.add_argument('--port', default=None)
    parser.add_argument('--relay-pin', type=int, default=cfg.relay_pin)
    parser.add_argument('--mock-relay', action='store_true', default=True)
    parser.add_argument('--no-speak', action='store_true')
    parser.add_argument('--wake-word', default=cfg.wake_word)
    parser.add_argument('--log', default=str(cfg.log_path))
    parser.add_argument('--model', default=str(cfg.model_path))
    parser.add_argument('--thresholds', default=str(cfg.thresholds_path))
    # S4 双信号拒识开关：默认取配置文件，--no-selective 可临时退回单阈值对照
    parser.add_argument('--gate', default=str(cfg.selective_gate_path or ''),
                        help='双信号规则文件（含阈值与池文件名）')
    parser.add_argument('--pool', default=str(cfg.selective_pool_path or ''),
                        help='kNN 参照池 npz；留空则由规则文件里的 ood.pool 推出')
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument('--selective', dest='selective', action='store_true',
                     default=cfg.selective_enabled, help='启用双信号拒识（默认取配置）')
    sel.add_argument('--no-selective', dest='selective', action='store_false',
                     help='强制退回单阈值 confidence_threshold')
    return parser


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(known.config)

    args = build_parser(cfg).parse_args(argv)

    from src.audio_capture import AudioCapture

    executor = build_executor(args)
    speaker = None if args.no_speak else MandarinSpeaker(
        audio_dir=cfg.broadcast_dir, wake_word=args.wake_word)
    engine = DialectEngine.load(args.model, args.thresholds, executor=executor,
                                speaker=speaker, vad=EnergyVad(fs=args.fs, **cfg.vad_kwargs()),
                                threshold=args.threshold, log_path=args.log,
                                gate_path=args.gate or None, pool_path=args.pool or None,
                                selective=args.selective,
                                fallback_mode=cfg.selective_fallback_mode,
                                window_ms=cfg.streaming_window_ms,
                                hop_ms=cfg.streaming_hop_ms,
                                pretrigger_threshold=cfg.pretrigger_threshold,
                                fast_silence_ms=cfg.fast_silence_ms,
                                fast_endpoint_threshold=cfg.fast_endpoint_threshold,
                                bias=cfg.keyword_bias)

    capture = AudioCapture(samplerate=args.fs, channels=1)
    capture.start()
    rule = ('双信号 ' + '、'.join(f'{k}≥{v:.4f}' for k, v in engine.gate.thresholds.items())
            if engine.gate is not None else f'单阈值 {engine.threshold:.2f}')
    print(f'开始监听（fs={args.fs}，拒识={rule}，配置={args.config or "默认"}），Ctrl+C 退出')
    try:
        for result in engine.run_stream(capture, fs=args.fs):
            state = '执行' if result.accepted else '拒识'
            print(f'[{state}] {result.command} 置信度={result.confidence:.3f} '
                  f'规则={result.rule} 延迟={result.latency_ms:.0f}ms '
                  f'时长={result.duration_sec:.2f}s'
                  f'{(" " + result.reason) if result.reason else ""}')
    except KeyboardInterrupt:
        print('\n已停止')
    finally:
        capture.stop()
        close = getattr(executor, 'close', None)
        if close:
            close()


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""实时识别入口：麦克风 -> 流式两遍识别 -> 三态决策 -> 执行 -> 普通话播报。

默认走**部署口径**（两遍 + 状态先验 + 期望代价决策），与
``scripts/eval_dialog.py --mode stream``、``scripts/bench_perceived_latency.py``
同一套参数，全部取自 ``config.yaml``。``--single-pass`` 可退回 S4 之前的
单遍路径做对照（历史公布指标以它为准）。

示例：
    python scripts/run_monitor.py --executor mock_esp32
    python scripts/run_monitor.py --executor serial --port COM5
    python scripts/run_monitor.py --executor relay --no-mock-relay
    python scripts/run_monitor.py --single-pass --no-speak --executor log
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
    # 安全默认仍是 mock（不碰真 GPIO），但必须能关掉：
    # 原来只有 store_true + default=True，真实继电器永远起不来。
    relay = parser.add_mutually_exclusive_group()
    relay.add_argument('--mock-relay', dest='mock_relay', action='store_true',
                       default=True, help='继电器走 mock（默认，不碰真 GPIO）')
    relay.add_argument('--no-mock-relay', dest='mock_relay', action='store_false',
                       help='驱动真实 GPIO（仅树莓派，需 RPi.GPIO）')
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
    # S4(b) 两遍流式开关：默认取 config 的 streaming.pretrigger。
    # 原来这里没有开关，run_stream 用默认 two_pass=False，
    # 麦克风入口实际跑的是 S4 之前的单遍路径，
    # 报告第 6 节的两遍指标在现场复现不出来。
    tp = parser.add_mutually_exclusive_group()
    tp.add_argument('--two-pass', dest='two_pass', action='store_true',
                    default=cfg.streaming_enabled,
                    help='流式两遍：预触发 + 整句复核 + 三态决策（默认取配置）')
    tp.add_argument('--single-pass', dest='two_pass', action='store_false',
                    help='退回单遍路径，与 S4 之前逐字一致（历史指标口径）')
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
    # S5 状态先验：原来这里不传 state_source，引擎里恒为 None，
    # _read_state() 永远返回 None —— config 的 state_source: mock 与
    # 整个期望代价决策在真实入口完全失效。工厂与 getter 都已存在，只是没接。
    from runtime.state import build_state_source
    state_source = build_state_source(cfg.state_source_kind,
                                      **cfg.state_source_options)
    engine = DialectEngine.load(args.model, args.thresholds, executor=executor,
                                speaker=speaker, vad=EnergyVad(fs=args.fs, **cfg.vad_kwargs()),
                                threshold=args.threshold, log_path=args.log,
                                gate_path=args.gate or None, pool_path=args.pool or None,
                                selective=args.selective,
                                fallback_mode=cfg.selective_fallback_mode,
                                state_source=state_source,
                                decision_costs=cfg.decision_costs,
                                confusable_pairs=cfg.confusable_pairs,
                                confirmation_pairs=cfg.confirmation_pairs,
                                competitor_floor=cfg.competitor_floor,
                                prior_temperature=cfg.prior_temperature,
                                redundancy_weight=cfg.redundancy_weight,
                                prior_override_margin=cfg.prior_override_margin,
                                consistency_margin=cfg.consistency_margin,
                                debounce_ms=cfg.debounce_ms,
                                max_clarify=cfg.max_clarify,
                                red_soft_enabled=cfg.red_soft_enabled,
                                red_soft_min_prob=cfg.red_soft_min_prob,
                                red_soft_min_margin=cfg.red_soft_min_margin,
                                red_soft_tolerances=cfg.red_soft_tolerances,
                                force_confirm_commands=cfg.force_confirm_commands,
                                high_risk_min_ood=cfg.high_risk_min_ood,
                                broadcast_command=cfg.broadcast_command,
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
    path = '两遍流式' if args.two_pass else '单遍（S4 之前口径）'
    # 把口径打在启动横幅上：现场演示与报告第 6 节是否同一口径，一眼可见。
    print(f'开始监听（fs={args.fs}，路径={path}，拒识={rule}，'
          f'状态源={cfg.state_source_kind}，配置={args.config or "默认"}），Ctrl+C 退出')
    try:
        for result in engine.run_stream(capture, fs=args.fs, two_pass=args.two_pass):
            # 两遍路径是三态决策：clarify 既不是执行也不是拒识，
            # 原来一律按 accepted 二分，会把澄清显示成「拒识」。
            if result.action:
                state = {'execute': '执行', 'clarify': '澄清',
                         'reject': '拒识'}.get(result.action, result.action)
                extra = f' 区={result.zone} 一致性={result.consistency or "-"}'
                if result.debounced:
                    extra += ' [去抖抑制]'
                if result.action_latency_ms:
                    extra += f' 动作延迟={result.action_latency_ms:.0f}ms'
            else:
                state = '执行' if result.accepted else '拒识'
                extra = ''
            print(f'[{state}] {result.command} 置信度={result.confidence:.3f} '
                  f'规则={result.rule} 延迟={result.latency_ms:.0f}ms '
                  f'时长={result.duration_sec:.2f}s{extra}'
                  f'{(" " + result.reason) if result.reason else ""}')
    except KeyboardInterrupt:
        print('\n已停止')
    finally:
        capture.stop()
        close = getattr(executor, 'close', None)
        if close:
            close()
        # MQTT 状态源持有连接，不关会留下后台线程。
        sclose = getattr(state_source, 'close', None)
        if sclose:
            sclose()


if __name__ == '__main__':
    main()

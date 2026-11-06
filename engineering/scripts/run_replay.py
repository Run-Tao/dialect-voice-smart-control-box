# -*- coding: utf-8 -*-
"""离线回放：把 WAV 文件（或整个目录）跑完整管线，无需麦克风。

用途：
- 在没有硬件时验证「VAD -> 特征 -> 分类 -> 拒识 -> 执行 -> 播报」全链路；
- 批量跑一批样本统计识别率与拒识率。

示例：
    python scripts/run_replay.py --audio "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\wav16k_v3\\edge" --limit 40
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config, load_config
from src.control import LoggingExecutor, MockEsp32Executor
from src.realtime import DialectEngine
from src.speaker_out import MandarinSpeaker
from src.vad import EnergyVad


def collect_files(path: Path):
    if path.is_dir():
        return sorted(p for p in path.rglob('*.wav'))
    return [path]


def build_parser(cfg: Config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True, help='WAV 文件或目录')
    parser.add_argument('--config', default=None, help='配置文件，默认项目根 config.yaml')
    parser.add_argument('--model', default=str(cfg.model_path))
    parser.add_argument('--thresholds', default=str(cfg.thresholds_path))
    parser.add_argument('--threshold', type=float, default=cfg.threshold,
                        help='置信度阈值，默认取 config.yaml 的 recognition.confidence_threshold')
    parser.add_argument('--fs', type=int, default=cfg.sample_rate)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--no-vad', action='store_true', help='整段送入，不做端点检测')
    parser.add_argument('--no-speak', action='store_true', help='不播报普通话音频')
    parser.add_argument('--executor', choices=['log', 'mock_esp32'], default=cfg.executor
                        if cfg.executor in ('log', 'mock_esp32') else 'log')
    parser.add_argument('--json-out', default=None)
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

    files = collect_files(Path(args.audio))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f'没有找到 WAV: {args.audio}')

    executor = LoggingExecutor() if args.executor == 'log' else MockEsp32Executor()
    speaker = None if args.no_speak else MandarinSpeaker(
        audio_dir=cfg.broadcast_dir, wake_word=cfg.wake_word, mock=True)

    engine = DialectEngine.load(args.model, args.thresholds, executor=executor,
                                speaker=speaker,
                                vad=EnergyVad(fs=args.fs, **cfg.vad_kwargs()),
                                threshold=args.threshold,
                                gate_path=args.gate or None, pool_path=args.pool or None,
                                selective=args.selective,
                                fallback_mode=cfg.selective_fallback_mode)
    if engine.gate is not None:
        print('拒识规则：双信号 '
              + '、'.join(f'{k} ≥ {v:.4f}' for k, v in engine.gate.thresholds.items())
              + f'（池 {engine.gate.pool.shape[0]} 条，k={engine.gate.k}）')
    else:
        print(f'拒识规则：单阈值 {engine.threshold:.4f}')

    import json
    from runtime.audio_io import load_wav

    rows = []
    accepted = 0
    for path in files:
        audio, sr = load_wav(path)
        if args.no_vad:
            result = engine.process_utterance(audio, sr)
        else:
            vad = EnergyVad(fs=sr)
            utterances = vad.feed(audio) + vad.flush()
            if not utterances:
                continue
            result = engine.process_utterance(utterances[0], sr)
        accepted += int(result.accepted)
        row = result.as_dict()
        row['file'] = path.name
        rows.append(row)
        print(f"{path.name}\t{result.command}\t{result.confidence:.3f}\t"
              f"{'接受' if result.accepted else '拒识'}")

    total = len(rows)
    if total:
        print(f'\n文件数 {total}，接受 {accepted}，接受率 {accepted / total:.2%}')
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print('结果已写入:', args.json_out)


if __name__ == '__main__':
    main()

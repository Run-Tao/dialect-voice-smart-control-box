# -*- coding: utf-8 -*-
"""端到端延迟分解基准：VAD / 特征 / 推理 / 播报 四段，输出 P50 / P95。

`realtime` 里记录的 latency_ms 只含特征 + 推理。用户真正感知的等待更长：
说完话 → VAD 静音挂起判定结束 → 特征 → 推理 → 普通话播报播完。
本脚本把这几段拆开量：

- vad_ms      VAD feed+flush 的 CPU 时间
- feature_ms  extract_sequence 的 CPU 时间
- infer_ms    predict_features 的 CPU 时间
- speech_ms   切出的语音段时长
- speak_ms    预录普通话音频的文件时长（不实际播放）
- hangover_ms 静音挂起等待，取 config 的 vad.min_silence_ms（流式下用户说完后必须等这么久）
- user_wait_ms = hangover + feature + infer + speak（用户说完到播报结束的等待）

示例：
    python scripts/bench_latency.py \
        --manifest "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\manifest_yue_v3_1_vad.csv" \
        --subset test --out docs/bench_latency.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.realtime import DialectEngine
from src.vad import EnergyVad
from runtime.audio_io import load_wav
from runtime.features_v3 import extract_sequence

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('vad_ms', 'feature_ms', 'infer_ms', 'gate_ms', 'speech_ms', 'speak_ms',
          'user_wait_ms')


def pct(values, q):
    return round(float(np.percentile(values, q)), 1) if values else None


def bench_one(engine, path, cfg):
    audio, sr = load_wav(path)
    vad = EnergyVad(fs=sr, **cfg.vad_kwargs())
    t0 = time.perf_counter()
    utterances = vad.feed(audio) + vad.flush()
    t1 = time.perf_counter()
    if not utterances:
        return None
    seg = utterances[0]
    seq = extract_sequence(seg, sr, engine.predictor.feature_spec)
    t2 = time.perf_counter()
    # 双信号路径的前向与单阈值路径是**同一次**前向（predict_detailed 只多返回嵌入），
    # 所以 infer_ms 两种口径可比；kNN 判定的开销单独记 gate_ms，不混进 infer_ms。
    if engine.gate is not None:
        pred, extras = engine.predictor.predict_detailed(seq)
        t3 = time.perf_counter()
        prob = np.asarray([pred.scores[c] for c in engine.predictor.classes_],
                          dtype=np.float64)
        decision = engine.gate.decide(prob, (extras or {}).get('embedding'))
        t4 = time.perf_counter()
        gate_ms = (t4 - t3) * 1000.0
        accepted, rule = bool(decision['accepted']), str(decision['rule'])
    else:
        pred = engine.predictor.predict_features(seq)
        t3 = time.perf_counter()
        t4 = t3
        gate_ms = 0.0
        accepted = bool(pred.confidence >= engine.threshold)
        rule = 'single_threshold'

    speak_ms = 0.0
    wav = cfg.broadcast_dir / f'{pred.command}.wav'
    if wav.exists():
        a, s = load_wav(wav)
        speak_ms = len(a) / float(s) * 1000.0
    hangover = float(cfg.vad_kwargs().get('min_silence_ms', 400))
    feature_ms = (t2 - t1) * 1000.0
    infer_ms = (t3 - t2) * 1000.0
    return {
        'file': path.name,
        'command': pred.command,
        'accepted': accepted,
        'rule': rule,
        'vad_ms': round((t1 - t0) * 1000.0, 1),
        'feature_ms': round(feature_ms, 1),
        'infer_ms': round(infer_ms, 1),
        'gate_ms': round(gate_ms, 2),
        'speech_ms': round(len(seg) / float(sr) * 1000.0, 1),
        'speak_ms': round(speak_ms, 1),
        'hangover_ms': hangover,
        'user_wait_ms': round(hangover + feature_ms + infer_ms + gate_ms + speak_ms, 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--subset', default='test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--config', default=None)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'bench_latency.json'))
    # 默认关闭：docs/bench_latency.json 里公布的 P50 2019 ms 是单阈值口径，
    # 被测试报告直接引用；要量双信号的额外开销就显式加 --selective 并另存 --out。
    parser.add_argument('--selective', action='store_true',
                        help='按双信号路径计时（含 kNN 判定开销 gate_ms）')
    parser.add_argument('--gate', default=None)
    parser.add_argument('--pool', default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    gate_path = args.gate or (str(cfg.selective_gate_path) if args.selective else None)
    pool_path = args.pool or (str(cfg.selective_pool_path)
                              if args.selective and cfg.selective_pool_path else None)
    engine = DialectEngine.load(str(cfg.model_path), str(cfg.thresholds_path),
                                gate_path=gate_path, pool_path=pool_path,
                                selective=args.selective)

    import csv
    files = []
    with open(args.manifest, encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('split') != args.subset:
                continue
            p = Path(row['audio_path'])
            if p.exists():
                files.append(p)
    files = sorted(files)
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f'manifest 中没有 {args.subset} 子集的文件')

    for p in files[:args.warmup]:
        bench_one(engine, p, cfg)

    rows = [r for r in (bench_one(engine, p, cfg) for p in files) if r]
    summary = {stage: {'p50': pct([r[stage] for r in rows], 50),
                       'p95': pct([r[stage] for r in rows], 95)}
               for stage in STAGES}
    summary['hangover_ms'] = rows[0]['hangover_ms']
    report = {
        'n': len(rows),
        'subset': args.subset,
        'manifest': args.manifest,
        'model': str(cfg.model_path),
        # 计时口径必须写进工件：gate_ms 在单阈值口径下恒为 0，
        # 两种口径的 user_wait_ms 不能混着比。
        'rejection_rule': (engine.gate.as_dict() if engine.gate is not None
                           else {'kind': 'single_threshold',
                                 'threshold': round(float(engine.threshold), 6)}),
        'summary': summary,
        'rows': rows,
        'note': 'vad/feature/infer 为 CPU 时间；gate 为双信号 kNN 判定开销（单阈值口径恒为 0）；'
                'user_wait 含静音挂起与播报时长，为离线估算，真实值需在目标硬件上实测。',
    }
    print(json.dumps({'n': report['n'], 'summary': summary}, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding='utf-8')
    print('已写入:', args.out)


if __name__ == '__main__':
    main()

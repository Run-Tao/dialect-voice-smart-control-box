# -*- coding: utf-8 -*-
"""感知延迟基准：动作 / 预触发 / 对话完成 三段延迟（P50 / P95）。

对应 v5 方案设计 §9 的验收口径：

| 指标 | 目标 | 本脚本怎么量的 |
|------|------|----------------|
| 动作延迟 | P95 ≤ 300 ms | 用户停止说话 -> 动作下发完成 |
| 预触发反馈 | P95 ≤ 150 ms | 见下面的口径说明 |
| 对话完成 | P95 ≤ 1.2 s | 动作下发 -> 用户听到「已完成」播完 |

口径说明（很重要，别把三段混着比）
----------------------------------
**动作延迟 = 音频时间 + CPU 时间**：

- 音频时间 = 从「VAD 判定最后一帧是语音」到「切段」之间的静音挂起。
  这一段是**音频本身**要等的时间，不是代码慢；预触发命中后静音挂起
  从 400 ms 压到 ``fast_silence_ms``（120 ms），这是 300 ms 预算能成立的前提。
- CPU 时间 = Pass-2 整句复核 + 双信号门限 + 决策层 + 执行器下发。

真实设备上音频是持续流入的，CPU 时间与后续音频**并行**，所以
「用户说完 -> 动作下发」的墙钟时间就是上面两项之和。

**预触发反馈延迟**分两个口径，都报出来：

- ``pretrigger_cpu_ms``：本块音频进来到预触发事件产生的处理耗时。
  这是代码能压的部分，也是 §9 里 150 ms 目标真正对应的量。
- ``pretrigger_audio_ms``：从用户开口到预触发产生的**音频时间**。
  Pass-1 用 1 s 滑窗打分且要求连续 2 个窗证据一致（``min_hits=2``），
  所以这个量**按构造就 ≥ 1.1 s**，与 150 ms 目标不在同一个口径上，
  不能拿它去判 150 ms 的验收。

**对话完成延迟** = 动作延迟 + ACK 等待 + 「已完成」播报时长。
普通话命令音频不计入：两遍路径里执行器直接控设备，播报只是确认，
而且是异步的（动作优先）。

用法::

    python scripts/bench_perceived_latency.py \
        --manifest ../DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv \
        --subset test --out docs/bench_perceived_latency.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.audio_io import load_wav
from src.config import load_config
from src.realtime import DialectEngine

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = ROOT.parent / 'DialectVoice_Research'
CORPUS_ROOT = RESEARCH_ROOT / 'data' / 'tts_corpus'


def resolve_audio(rel: str) -> Optional[Path]:
    """清单里的 audio_path 写法不统一：绝对路径、相对项目根、相对语料目录。"""
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    for base in (RESEARCH_ROOT, CORPUS_ROOT):
        cand = base / p
        if cand.exists():
            return cand
    return None


def stream_manifest_path(manifest: Path) -> Path:
    """推导流式回放该用的**原始**音频清单。

    裁剪语料（VAD 已经裁好）本身含 300 ms pre-roll，再喂进 VAD 会**叠加**
    一段 pre-roll，模型看到的输入分布和线上不一样（实测 argmax 0.975 -> 0.65）。
    流式回放必须用裁剪前的原始音频，让 VAD 自己切。
    命名约定：裁剪版清单文件名带 ``_vad`` 后缀，去掉即原始清单。
    """
    p = manifest.with_name(manifest.stem.replace('_vad', '') + manifest.suffix)
    if not p.exists():
        sys.exit(f'找不到原始音频清单: {p}\n'
                 f'流式回放必须用裁剪前的原始音频（否则 pre-roll 会叠加），'
                 f'请用 --stream-manifest 显式指定')
    return p

# 句前/句尾填充必须用**数字静音**（全 0），不能用电平噪声。
# 原因：VAD 的噪声底是自适应的，只在判为非语音的帧上更新，而语料 wav 自带
# 数字静音（-120 dB）。加一段 -60 dB 的底噪进去，噪声底会被语料的静音拖到
# -120 dB，判决门限跟着掉到 -111 dB，于是**填充的底噪自己变成了语音**，
# 语音段永远切不出来（实测：第 8 条起全部切不出段）。
# 数字静音与语料静音同电平，噪声底和门限一起收敛到同一处，不会自激。
PAD_VALUE = 0.0


def pct(values, q):
    return round(float(np.percentile(values, q)), 1) if values else None


def stat(values) -> Dict[str, Optional[float]]:
    if not values:
        return {'p50': None, 'p95': None, 'mean': None, 'max': None, 'n': 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        'p50': round(float(np.percentile(arr, 50)), 1),
        'p95': round(float(np.percentile(arr, 95)), 1),
        'mean': round(float(arr.mean()), 1),
        'max': round(float(arr.max()), 1),
        'n': int(arr.size),
    }


class _BenchExecutor:
    """离线替身执行器：记录下发并立刻给回执，不接硬件。"""

    async_ack = False

    def __init__(self):
        self.dispatched: List[str] = []

    def dispatch(self, command: str):
        self.dispatched.append(command)
        return {'command': command, 'ok': True, 'detail': 'bench'}

    def wait_ack(self, timeout=None):
        return {'ok': True, 'detail': 'bench'}


class _BenchSpeaker:
    """只记录播了哪些音频，不真的发声（离线跑）。"""

    def __init__(self, audio_dir: Path):
        self.audio_dir = Path(audio_dir)
        self.spoken: List[str] = []
        self.durations: Dict[str, float] = {}

    def _dur(self, asset: str) -> float:
        if asset in self.durations:
            return self.durations[asset]
        p = self.audio_dir / f'{asset}.wav'
        ms = 0.0
        if p.exists():
            audio, sr = load_wav(p)
            ms = len(audio) / float(sr) * 1000.0
        self.durations[asset] = ms
        return ms

    def speak(self, command: str) -> bool:
        self.spoken.append(command)
        return True

    def speak_done(self, command: str = '') -> bool:
        self.spoken.append('ACK_DONE')
        return True

    def speak_error(self, command: str = '') -> bool:
        self.spoken.append('ACK_FAIL')
        return True

    def speak_reject(self) -> bool:
        self.spoken.append('REJECT_UNKNOWN')
        return True

    def speak_clarify(self, candidates) -> bool:
        self.spoken.append('CLARIFY')
        return True

    def asset_ms(self, asset: str) -> float:
        return self._dur(asset)


def make_silence(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return np.full(int(n_samples), PAD_VALUE, dtype=np.float32)


def build_engine(cfg, gate_enabled: bool) -> DialectEngine:
    from src.vad import EnergyVad
    engine = DialectEngine.load(
        str(cfg.model_path), str(cfg.thresholds_path),
        gate_path=(str(cfg.selective_gate_path) if gate_enabled else None),
        pool_path=(str(cfg.selective_pool_path)
                   if gate_enabled and cfg.selective_pool_path else None),
        selective=gate_enabled,
        vad=EnergyVad(fs=cfg.sample_rate, **cfg.vad_kwargs()),
        state_source=_mock_state(),
        decision_costs=cfg.decision_costs,
        confusable_pairs=cfg.confusable_pairs,
        competitor_floor=cfg.competitor_floor,
        prior_temperature=cfg.prior_temperature,
        redundancy_weight=cfg.redundancy_weight,
        prior_override_margin=cfg.prior_override_margin,
        debounce_ms=cfg.debounce_ms,
        max_clarify=cfg.max_clarify,
        window_ms=cfg.streaming_window_ms,
        hop_ms=cfg.streaming_hop_ms,
        pretrigger_threshold=cfg.pretrigger_threshold,
        fast_silence_ms=cfg.fast_silence_ms,
        fast_endpoint_threshold=cfg.fast_endpoint_threshold,
        bias=cfg.keyword_bias,
        broadcast_async=False,
        broadcast_command=cfg.broadcast_command,
    )
    engine.executor = _BenchExecutor()
    return engine


def _mock_state():
    import importlib
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module('runtime.state').MockStateSource()


def corpus_reference_segment(audio, fs: int, vad_kwargs) -> Optional[np.ndarray]:
    """用**默认挂起**的 VAD 切同一段音频：这是训练语料的切法。

    语料是用 ``scripts/make_trimmed_corpus.py`` 生成的，那里的 VAD 用默认
    ``min_silence_ms``（400 ms -> 13 帧 = 390 ms）。部署路径在快速端点命中后
    把挂起压到 120 ms，两者切点不同：句内只要有超过 120 ms 的停顿，部署路径
    就会**提前切段**，把后半句留给下一段——而模型从没见过这种被截断的输入。
    """
    from src.vad import EnergyVad
    vad = EnergyVad(fs=fs, **vad_kwargs)
    segs = vad.feed(audio) + vad.flush()
    return segs[0] if segs else None


def bench_one(engine, speaker, path: Path, fs: int, block_frames: int,
              lead_ms: float, tail_ms: float, rng: np.random.Generator,
              vad_kwargs=None) -> Optional[dict]:
    audio, sr = load_wav(path)
    if sr != fs:
        return None
    lead = make_silence(int(fs * lead_ms / 1000.0), rng)
    tail = make_silence(int(fs * tail_ms / 1000.0), rng)
    stream = np.concatenate([lead, np.asarray(audio, dtype=np.float32), tail])

    engine.reset_stream()
    engine.debouncer.reset()

    pretrigger: Dict[str, object] = {}

    def on_pretrigger(event):
        pretrigger.update({
            'command': event.command,
            'sample_pos': int(event.sample_pos),
            'stream_pos': int(engine._stream_pos),
            'cpu_ms': float(event.latency_ms),
            'evidence': float(event.evidence),
        })

    engine.on_pretrigger = on_pretrigger

    results = []
    for i in range(0, stream.size, block_frames):
        results.extend(engine.process_stream_block(stream[i:i + block_frames], fs))

    if not results:
        return None
    res = results[0]

    vad = engine.vad
    speech_start = int(vad.speech_start_sample)
    speech_end = int(vad.speech_end_sample)
    cut = int(vad.cut_sample)
    if speech_end < 0 or cut < 0:
        return None

    # 音频时间：VAD 判定的「最后一帧语音」到「切段」。
    audio_wait_ms = (cut - speech_end) / float(fs) * 1000.0
    # CPU 时间：切段后这一轮的复核 + 门限 + 决策 + 下发。
    cpu_ms = float(res.latency_ms) + float(res.action_latency_ms)

    pre_audio_ms = None
    pre_cpu_ms = None
    if pretrigger:
        pre_cpu_ms = float(pretrigger['cpu_ms'])
        # 预触发事件的位置取事件发生那一刻的累计样本数，
        # 和 VAD 的 speech_start_sample 同一个坐标系。
        pre_pos = int(pretrigger['stream_pos'])
        if speech_start >= 0 and pre_pos >= speech_start:
            pre_audio_ms = (pre_pos - speech_start) / float(fs) * 1000.0

    ack_ms = 0.0
    ack_done_ms = speaker.asset_ms('ACK_DONE')

    # 语料对齐：部署路径切出的段 vs 默认挂起切出的段（训练语料的条件）。
    corpus_len_ms = None
    truncated = False
    if vad_kwargs is not None:
        ref = corpus_reference_segment(np.asarray(audio, dtype=np.float32), fs,
                                       vad_kwargs)
        if ref is not None:
            corpus_len_ms = round(len(ref) / float(fs) * 1000.0, 1)
            # 一个 VAD 帧（30 ms）以内的差别不算截断。
            truncated = bool(float(res.duration_sec) * 1000.0
                             < corpus_len_ms - 30.0)

    return {
        'file': path.name,
        'command': res.command,
        'action': res.action,
        'zone': res.zone,
        'executed': bool(res.executed),
        'gate_accepted': bool(res.gate_accepted),
        'pass1_command': res.pass1_command,
        'consistency': res.consistency,
        'audio_wait_ms': round(max(audio_wait_ms, 0.0), 1),
        'cpu_verify_ms': round(float(res.latency_ms), 1),
        'cpu_dispatch_ms': round(float(res.action_latency_ms), 2),
        'action_latency_ms': round(max(audio_wait_ms, 0.0) + cpu_ms, 1),
        'pretrigger_fired': bool(pretrigger),
        'pretrigger_command': pretrigger.get('command', ''),
        'pretrigger_audio_ms': (round(pre_audio_ms, 1)
                                 if pre_audio_ms is not None else None),
        'pretrigger_cpu_ms': (round(pre_cpu_ms, 1)
                              if pre_cpu_ms is not None else None),
        'ack_ms': round(ack_ms, 1),
        'ack_done_audio_ms': round(ack_done_ms, 1),
        'dialog_complete_ms': round(max(audio_wait_ms, 0.0) + cpu_ms
                                    + ack_ms + ack_done_ms, 1),
        'n_segments': len(results),
        'stream_segment_ms': round(float(res.duration_sec) * 1000.0, 1),
        'corpus_segment_ms': corpus_len_ms,
        'truncated': truncated,
    }


def load_rows(manifest: Path, subset: str, limit: int) -> List[dict]:
    rows = []
    with open(manifest, encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            if subset and row.get('split') != subset:
                continue
            p = resolve_audio(row['audio_path'])
            if p is not None:
                rows.append({'path': p, 'command': row.get('command', ''),
                             'utterance_id': row.get('utterance_id', '')})
    rows.sort(key=lambda r: r['utterance_id'])
    if limit:
        rows = rows[:limit]
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--stream-manifest', default=None,
                        help='流式回放用的**原始**音频清单；默认由 --manifest '
                             '去掉 _vad 后缀推导。裁剪语料含 pre-roll，'
                             '再进 VAD 会叠加，模型输入分布与线上不一致')
    parser.add_argument('--subset', default='test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--config', default=None)
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'bench_perceived_latency.json'))
    parser.add_argument('--block-frames', type=int, default=1024)
    parser.add_argument('--lead-ms', type=float, default=0.0,
                        help='句前额外填充。用原始音频时必须为 0：'
                             '语料自带的静音就是「麦克风听到的样子」，'
                             '再补一段会改变 VAD 的噪声底轨迹，'
                             '切出的段与线上不一致')
    parser.add_argument('--tail-ms', type=float, default=0.0,
                        help='句尾额外填充。用原始音频时必须为 0：'
                             '原始音频自带尾部静音，足够切段；'
                             '补静音会让静音挂起提前结束，与线上不一致')
    parser.add_argument('--no-gate', action='store_true',
                        help='不启用双信号门限（对照口径）')
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    manifest = Path(args.manifest)
    manifest = (Path(args.stream_manifest) if args.stream_manifest
                else stream_manifest_path(manifest))
    rows = load_rows(manifest, args.subset, args.limit)
    if not rows:
        sys.exit('manifest 中没有可用样本')

    engine = build_engine(cfg, not args.no_gate)
    speaker = _BenchSpeaker(cfg.broadcast_dir)
    engine.speaker = speaker

    rng = np.random.default_rng(20261103)
    fs = cfg.sample_rate
    vad_kwargs = cfg.vad_kwargs()

    for row in rows[:args.warmup]:
        bench_one(engine, speaker, row['path'], fs, args.block_frames,
                  args.lead_ms, args.tail_ms, rng, vad_kwargs)

    records = []
    for row in rows:
        rec = bench_one(engine, speaker, row['path'], fs, args.block_frames,
                        args.lead_ms, args.tail_ms, rng, vad_kwargs)
        if rec is not None:
            rec['utterance_id'] = row['utterance_id']
            rec['truth'] = row['command']
            records.append(rec)

    if not records:
        sys.exit('没有样本切出语音段')

    executed = [r for r in records if r['executed']]
    action_lat = [r['action_latency_ms'] for r in executed]
    pre_cpu = [r['pretrigger_cpu_ms'] for r in records
               if r['pretrigger_cpu_ms'] is not None]
    pre_audio = [r['pretrigger_audio_ms'] for r in records
                 if r['pretrigger_audio_ms'] is not None]
    dialog_lat = [r['dialog_complete_ms'] for r in executed]

    # 语料对齐：被快速端点提前切段的样本，以及它们的执行结果。
    trunc = [r for r in records if r.get('truncated')]
    trunc_exec = [r for r in trunc if r['executed']]
    trunc_wrong = [r for r in trunc_exec if r['command'] != r['truth']]
    kept_exec = [r for r in records
                 if not r.get('truncated') and r['executed']]
    kept_wrong = [r for r in kept_exec if r['command'] != r['truth']]
    corpus_alignment = {
        'n_compared': len(records),
        'n_truncated': len(trunc),
        'truncated_rate': (round(len(trunc) / float(len(records)), 4)
                           if records else None),
        'n_split_into_multi_segment': sum(
            1 for r in records if r.get('n_segments', 1) > 1),
        'truncated_executed': len(trunc_exec),
        'truncated_wrong_command': len(trunc_wrong),
        'truncated_wrong_pairs': sorted(
            f"{r['truth']}->{r['command']}" for r in trunc_wrong),
        'kept_executed': len(kept_exec),
        'kept_wrong_command': len(kept_wrong),
        'kept_wrong_rate': (round(len(kept_wrong) / float(len(kept_exec)), 4)
                            if kept_exec else None),
        'truncated_wrong_rate': (round(len(trunc_wrong) / float(len(trunc_exec)), 4)
                                 if trunc_exec else None),
        'note': '截断 = 部署路径切出的语音段比「默认挂起切法」（训练语料的切法）'
                '短超过 30 ms（一个 VAD 帧）。快速端点把静音挂起压到 120 ms，'
                '句内超过 120 ms 的停顿就会提前切段。',
    }

    report = {
        'n': len(records),
        'n_executed': len(executed),
        'subset': args.subset,
        'manifest': str(manifest),
        'model': str(cfg.model_path),
        'gate_enabled': not args.no_gate,
        'stream': {
            'block_frames': args.block_frames,
            'fs': fs,
            'lead_ms': args.lead_ms,
            'tail_ms': args.tail_ms,
            'window_ms': cfg.streaming_window_ms,
            'hop_ms': cfg.streaming_hop_ms,
            'pretrigger_threshold': cfg.pretrigger_threshold,
            'fast_endpoint_threshold': cfg.fast_endpoint_threshold,
            'fast_silence_ms': cfg.fast_silence_ms,
        },
        'action_latency_ms': stat(action_lat),
        'action_latency_breakdown_ms': {
            'audio_hangover': stat([r['audio_wait_ms'] for r in executed]),
            'cpu_verify_gate_decision': stat([r['cpu_verify_ms'] for r in executed]),
            'cpu_dispatch': stat([r['cpu_dispatch_ms'] for r in executed]),
        },
        'pretrigger_cpu_ms': stat(pre_cpu),
        'pretrigger_audio_ms': stat(pre_audio),
        'dialog_complete_ms': stat(dialog_lat),
        'pretrigger_fire_rate': round(
            len(pre_cpu) / float(len(records)), 4) if records else None,
        'corpus_alignment': corpus_alignment,
        'acceptance': {
            'action_p95_le_300ms': (
                stat(action_lat)['p95'] is not None
                and stat(action_lat)['p95'] <= 300.0),
            'pretrigger_cpu_p95_le_150ms': (
                stat(pre_cpu)['p95'] is not None
                and stat(pre_cpu)['p95'] <= 150.0),
            'dialog_complete_p95_le_1200ms': (
                stat(dialog_lat)['p95'] is not None
                and stat(dialog_lat)['p95'] <= 1200.0),
        },
        'note': '动作延迟 = 静音挂起（音频时间）+ 复核/门限/决策/下发（CPU 时间）；'
                '预触发分 CPU 与音频两个口径，音频口径按构造 ≥ window_ms + hop_ms，'
                '不能与 150 ms 目标比；对话完成含「已完成」播报时长，不含普通话命令音频'
                '（两遍路径由执行器直接控设备，播报异步且只是确认）。'
                '合成语料，非真人性能。',
        'rows': records,
    }

    printable = {k: v for k, v in report.items() if k != 'rows'}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding='utf-8')
    print('已写入:', args.out)


if __name__ == '__main__':
    main()

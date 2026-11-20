# -*- coding: utf-8 -*-
"""会话级评估：任务成功率、澄清率、危险误动作（S5 验收口径）。

对应 v5 方案设计 §10 的 S5 验收「易混对任务成功率提升」与 §9 的「澄清率 ≤ 15%」。

三种口径放在同一批样本、同一份模型工件上对比：

- ``argmax``      单遍 argmax 准确率——历史公布的基线口径（测试集 0.882）；
- ``single_gate`` 单遍 + 双信号门限：门限放行且 argmax 正确——
  S0/S3 公布的「覆盖率」口径，**只算一次成功，没有对话补救**；
- ``dialog``      两遍 + 决策层 + 澄清 + 重说——会话级成功率。

重说样本怎么取（这是本脚本最关键、也最容易误导人的假设）
----------------------------------------------------------
澄清之后用户重说，本脚本取**同一命令的另一个表达组**的样本，
即模拟「用户换一种说法再说一遍」。这样测的是**系统**在拿到第二次机会后
能不能纠正自己，而不是「同一段音频再算一遍」。

代价：这个假设偏乐观。实测同一表达组的兄弟样本准确率只有 0.358
（错误集中在 CURTAIN_OPEN_14、CURTAIN_CLOSE_15 这两个表达组），
真实用户原样重复时澄清收益会明显低于本脚本报出的数字。
``--retry-mode same`` 可以跑同一个悲观口径做对照。

两种模式
--------
``--mode offline``（默认）：直接把切好的语音段送进两遍路径，``candidate=None``，
即**没有 Pass-1 候选**——没有关键词偏置，也没有两遍一致性规则。
这是「只有对话层与决策层」的消融口径。

``--mode stream``：走**部署路径**（音频分块喂进 ``process_stream_block``）：
Pass-1 滑窗预触发 -> 快速端点 -> Pass-2 整句复核（带关键词偏置）->
两遍一致性 -> 决策层 -> 执行/澄清/拒识。S5 的验收数字应当用这个口径，
因为线上跑的就是它。

用法::

    python scripts/eval_dialog.py \
        --manifest ../DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv \
        --subset test --out docs/dialog_eval.json

    python scripts/eval_dialog.py --mode stream \
        --manifest ../DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv \
        --subset test --out docs/dialog_eval_stream.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from runtime.audio_io import load_wav
from src.commands import COMMANDS
from src.config import load_config
from src.realtime import DialectEngine
from src.vad import EnergyVad

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

# 流式回放参数：与 scripts/bench_perceived_latency.py 一致。
# 句前/句尾**不额外补静音**：原始音频自带的静音就是「麦克风听到的样子」，
# 再补一段会改变 VAD 的噪声底轨迹，切出的段与线上不一致（实测会明显掉覆盖率）。
STREAM_LEAD_MS = 0.0
STREAM_TAIL_MS = 0.0
STREAM_BLOCK = 1024


def pct(values, q):
    if not values:
        return None
    import numpy as np
    return round(float(np.percentile(values, q)), 4)


def rate(num, den):
    return round(float(num) / float(den), 4) if den else None

def is_dangerous(command: str) -> bool:
    """危险动作 = 会改变设备状态（查询类不算）。"""
    entry = COMMANDS.get(command)
    return bool(entry) and entry['actuator'] != 'none'


def load_manifest(path: Path, subset: str, limit: int = 0) -> List[dict]:
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            if subset and row.get('split') != subset:
                continue
            p = resolve_audio(row['audio_path'])
            if p is not None:
                row['_path'] = str(p)
                rows.append(row)
    rows.sort(key=lambda r: r['utterance_id'])
    if limit:
        rows = rows[:limit]
    return rows


class _CountingExecutor:
    """只记录下发了哪些命令的替身执行器（离线评估不接真实硬件）。"""

    async_ack = False

    def __init__(self):
        self.dispatched: List[str] = []

    def dispatch(self, command: str):
        self.dispatched.append(command)
        return {'command': command, 'ok': True, 'detail': 'eval'}

    def wait_ack(self, timeout=None):
        return {'ok': True, 'detail': 'eval'}


def build_engine(cfg, gate_enabled: bool) -> DialectEngine:
    state = _runtime_state().MockStateSource()
    engine = DialectEngine.load(
        str(cfg.model_path), str(cfg.thresholds_path),
        gate_path=(str(cfg.selective_gate_path) if gate_enabled else None),
        pool_path=(str(cfg.selective_pool_path)
                   if gate_enabled and cfg.selective_pool_path else None),
        selective=gate_enabled,
        vad=EnergyVad(fs=cfg.sample_rate, **cfg.vad_kwargs()),
        state_source=state,
        decision_costs=cfg.decision_costs,
        confusable_pairs=cfg.confusable_pairs,
        competitor_floor=cfg.competitor_floor,
        prior_temperature=cfg.prior_temperature,
        redundancy_weight=cfg.redundancy_weight,
        prior_override_margin=cfg.prior_override_margin,
        consistency_margin=cfg.consistency_margin,
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
    engine.executor = _CountingExecutor()
    return engine

def _runtime_state():
    import importlib
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module('runtime.state')


def pick_retry(rows: List[dict], index: int, retry_mode: str) -> Optional[dict]:
    """给第 ``index`` 条找一条重说样本。

    ``group``：同一命令的**另一个表达组**（默认，模拟换一种说法）；
    ``same``：同一表达组的另一条（悲观口径，模拟原样重复）。
    """
    truth = rows[index]['command']
    same_group = rows[index].get('expression_group_id', '')
    candidates = [r for r in rows if r['command'] == truth and r is not rows[index]]
    if retry_mode == 'same':
        group = [r for r in candidates if r.get('expression_group_id') == same_group]
        return group[0] if group else None
    others = [r for r in candidates if r.get('expression_group_id') != same_group]
    return others[0] if others else None


def run_segment(engine, audio, sr: int, mode: str):
    """跑一条语音，返回 ``UtteranceResult``；切不出段时返回 None。

    ``offline``：直接送两遍路径，没有 Pass-1 候选。
    ``stream``：走部署路径，候选来自 Pass-1 预触发。
    """
    if mode == 'offline':
        return engine.process_segment(audio, sr, candidate=None)
    engine.reset_stream()
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    lead = np.zeros(int(sr * STREAM_LEAD_MS / 1000.0), dtype=np.float32)
    tail = np.zeros(int(sr * STREAM_TAIL_MS / 1000.0), dtype=np.float32)
    stream = np.concatenate([lead, x, tail])
    results = []
    for i in range(0, stream.size, STREAM_BLOCK):
        results.extend(engine.process_stream_block(stream[i:i + STREAM_BLOCK], sr))
    return results[0] if results else None


class _EmptyResult:
    """切不出语音段时的占位结果：这一条按「没识别出来」计。"""

    scores: Dict[str, float] = {}
    acoustic_scores: Dict[str, float] = {}
    gate_accepted = False
    executed = False
    command = ''
    action = 'reject'
    zone = 'red'
    rule = 'no_segment'
    consistency = 'no_candidate'
    debounced = False


def evaluate(rows: List[dict], cfg, gate_enabled: bool, retry_mode: str,
             mode: str = 'offline') -> dict:
    engine = build_engine(cfg, gate_enabled)

    n = len(rows)
    argmax_ok = 0
    single_gate_ok = 0
    single_gate_exec = 0
    session_ok = 0
    clarify_n = 0
    reject_n = 0
    danger_n = 0
    danger_ok_n = 0
    debounced_n = 0
    recovered_n = 0        # argmax 错、对话救回来的
    lost_n = 0             # argmax 对、对话弄丢的
    by_command: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'n': 0, 'argmax_ok': 0, 'dialog_ok': 0, 'clarify': 0,
                 'gate_accept': 0})
    per_class_rows = []

    for i, row in enumerate(rows):
        truth = row['command']
        audio, sr = load_wav(row['_path'])
        # 离线回放里相邻语句间隔为零，去抖窗口会把「同命令重复」全判成抑制，
        # 那是仿真的时间轴问题不是识别问题，所以这里按语句重置。
        engine.debouncer.reset()
        res = run_segment(engine, audio, sr, mode) or _EmptyResult()

        # 基线口径必须用**无偏**后验：stream 模式下 scores 是带关键词偏置的，
        # 拿它算 argmax 就等于把偏置算进基线里，臂与臂之间不可比。
        scores = res.acoustic_scores or res.scores
        top1 = max(scores, key=lambda c: scores[c]) if scores else ''
        ok_a = bool(top1 == truth)
        ok_b = bool(res.gate_accepted and top1 == truth)
        argmax_ok += int(ok_a)
        single_gate_ok += int(ok_b)
        single_gate_exec += int(res.gate_accepted)

        executed = res.executed
        final_command = res.command if executed else ''
        action_first = res.action
        clarified = action_first == 'clarify'
        clarify_n += int(clarified)
        reject_n += int(action_first == 'reject')
        debounced_n += int(res.debounced)

        retry_id = ''
        if clarified:
            retry = pick_retry(rows, i, retry_mode)
            if retry is not None:
                retry_id = retry['utterance_id']
                engine.debouncer.reset()
                r_audio, r_sr = load_wav(retry['_path'])
                res2 = run_segment(engine, r_audio, r_sr, mode) or _EmptyResult()
                executed = res2.executed
                final_command = res2.command if executed else ''

        ok_c = bool(executed and final_command == truth)
        session_ok += int(ok_c)
        recovered_n += int(ok_c and not ok_a)
        lost_n += int(ok_a and not ok_c)
        if executed and final_command != truth and is_dangerous(final_command):
            danger_n += 1
        if ok_c and is_dangerous(truth):
            danger_ok_n += 1

        bucket = by_command[truth]
        bucket['n'] += 1
        bucket['argmax_ok'] += int(ok_a)
        bucket['dialog_ok'] += int(ok_c)
        bucket['clarify'] += int(clarified)
        bucket['gate_accept'] += int(res.gate_accepted)

        per_class_rows.append({
            'utterance_id': row['utterance_id'],
            'command': truth,
            'expression_group_id': row.get('expression_group_id', ''),
            'argmax': top1,
            'argmax_ok': ok_a,
            'gate_accepted': bool(res.gate_accepted),
            'gate_rule': res.rule,
            'action_first': action_first,
            'zone': res.zone,
            'consistency': res.consistency,
            'pass1_command': getattr(res, 'pass1_command', ''),
            'clarify': bool(clarified),
            'retry_utterance_id': retry_id,
            'debounced': bool(res.debounced),
            'executed': bool(executed),
            'final_command': final_command,
            'session_ok': ok_c,
        })

    pair_names = ('CURTAIN_OPEN', 'CURTAIN_CLOSE')
    pair_rows = [r for r in per_class_rows if r['command'] in pair_names]
    pair_report = {
        'commands': list(pair_names),
        'n': len(pair_rows),
        'argmax_accuracy': rate(sum(r['argmax_ok'] for r in pair_rows), len(pair_rows)),
        'single_gate_accuracy': rate(
            sum(r['argmax_ok'] and r['gate_accepted'] for r in pair_rows),
            len(pair_rows)),
        'dialog_session_success': rate(sum(r['session_ok'] for r in pair_rows),
                                       len(pair_rows)),
        'clarification_rate': rate(sum(r['clarify'] for r in pair_rows), len(pair_rows)),
    }
    if pair_report['n']:
        pair_report['gain_vs_single_gate'] = round(
            pair_report['dialog_session_success'] - pair_report['single_gate_accuracy'], 4)
        pair_report['gain_vs_argmax'] = round(
            pair_report['dialog_session_success'] - pair_report['argmax_accuracy'], 4)

    report = {
        'n': n,
        'gate_enabled': bool(gate_enabled),
        'mode': mode,
        'retry_mode': retry_mode,
        'argmax_accuracy': rate(argmax_ok, n),
        'single_gate_coverage': rate(single_gate_exec, n),
        'single_gate_accuracy': rate(single_gate_ok, n),
        'single_gate_precision': rate(single_gate_ok, single_gate_exec),
        'dialog_session_success': rate(session_ok, n),
        'clarification_rate': rate(clarify_n, n),
        'reject_rate': rate(reject_n, n),
        'debounced': debounced_n,
        'dangerous_misaction_rate': rate(danger_n, n),
        'dangerous_correct_rate': rate(danger_ok_n, n),
        'recovered_by_dialog': recovered_n,
        'lost_by_dialog': lost_n,
        'session_success_gain_vs_single_gate': (
            round(rate(session_ok, n) - rate(single_gate_ok, n), 4) if n else None),
        'session_success_gain_vs_argmax': (round(rate(session_ok, n) - rate(argmax_ok, n), 4)
                                           if n else None),
        'by_command': {k: {**v,
                           'argmax_accuracy': rate(v['argmax_ok'], v['n']),
                           'gate_acceptance': rate(v['gate_accept'], v['n']),
                           'dialog_success': rate(v['dialog_ok'], v['n']),
                           'clarification_rate': rate(v['clarify'], v['n'])}
                       for k, v in sorted(by_command.items())},
        'confusable_pair': pair_report,
        'note': '重说样本取自同一命令的另一个表达组（--retry-mode same 为同组悲观口径）；'
                '合成语料，非真人性能。argmax 口径不含门限，与 dialog 口径不可直接比；'
                '可比的是 single_gate（同样带门限、没有对话补救）与 dialog。'
                'mode=offline 表示没有 Pass-1 候选（无关键词偏置、无两遍一致性），'
                'mode=stream 才是部署路径。',
    }
    return report, per_class_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--subset', default='test')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--config', default=None)
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'dialog_eval.json'))
    parser.add_argument('--no-gate', action='store_true',
                        help='不启用双信号门限（对照口径）')
    parser.add_argument('--retry-mode', choices=('group', 'same'), default='group',
                        help='重说样本取自另一个表达组（group，默认）还是同组兄弟（same）')
    parser.add_argument('--mode', choices=('offline', 'stream'), default='offline',
                        help='offline：没有 Pass-1 候选；stream：部署路径'
                             '（预触发 + 关键词偏置 + 两遍一致性）')
    parser.add_argument('--stream-manifest', default=None,
                        help='流式回放用的**原始**音频清单；默认由 --manifest '
                             '去掉 _vad 后缀推导。裁剪语料含 pre-roll，'
                             '再进 VAD 会叠加，模型输入分布与线上不一致')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    manifest = Path(args.manifest)
    if args.mode == 'stream':
        manifest = (Path(args.stream_manifest) if args.stream_manifest
                    else stream_manifest_path(manifest))
    rows = load_manifest(manifest, args.subset, args.limit)
    if not rows:
        sys.exit('manifest 中没有可用样本')
    report, detail = evaluate(rows, cfg, not args.no_gate, args.retry_mode, args.mode)
    report['manifest'] = str(manifest)
    report['subset'] = args.subset
    report['model'] = str(cfg.model_path)
    report['rows'] = detail
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding='utf-8')
    printable = {k: v for k, v in report.items() if k != 'rows'}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print('已写入:', args.out)


if __name__ == '__main__':
    main()

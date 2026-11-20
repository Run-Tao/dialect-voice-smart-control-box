# -*- coding: utf-8 -*-
"""流式两遍管线测试：Pass-1 预触发、快速端点、两遍一致性、去抖、ACK 闭环。

这里全部用**后验固定的假预测器**：测的是管线的接缝（谁在什么时刻把什么
交给谁），不是模型的准确率。模型准确率由 ``scripts/evaluate_pipeline.py``
在真实语料上单独测，两件事混在一起会让失败原因无法定位。

假预测器必须**如实复刻**真模型的两个约定，否则这些测试就是在测自己造的
假象：

1. 关键词偏置加在 logits 上，再按温度走 softmax。所以
   ``temperature = 1.0`` 时，``p_biased ∝ p * exp(bias)``（只作用于候选类）。
2. ``predict_detailed`` 的 ``extras['seed_probs']`` 是**逐种子**概率
   （[S, C]），不是均值。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.features_v3 import FeatureSpec
from runtime.streaming import Pass1Detector
from src.realtime import DialectEngine
from src.vad import EnergyVad

FS = 16000

# 与部署模型一致的 6 类。2 类替身测不出「证据衰减」这类行为：归一化后
# 最大类概率恒 ≥ 0.5，Pass-1 的证据 EMA 永远落不到重武装阈值以下；
# 6 类下最大类可以低到 1/6，衰减路径才测得到。
SIX = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE',
       'ALL_OFF', 'STATUS_ALL']


# ----------------------------------------------------------------------
# 替身
# ----------------------------------------------------------------------
@dataclass
class _StubPrediction:
    command: str
    confidence: float
    scores: Dict[str, float]
    margin: float = 0.0


class _StubPredictor:
    """后验固定的假预测器。"""

    def __init__(self, classes, probs: Dict[str, float], seed_probs=None):
        self.classes_ = list(classes)
        self.probs = {c: float(v) for c, v in probs.items()}
        self.seed_probs = seed_probs
        self.feature_spec = FeatureSpec()
        self.temperature = 1.0
        self.threshold = 0.0

    def _probs(self, bias=None, bias_class=None) -> np.ndarray:
        p = np.asarray([self.probs.get(c, 0.0) for c in self.classes_],
                       dtype=np.float64)
        p = p / p.sum()
        if bias and bias_class and bias_class in self.classes_:
            idx = self.classes_.index(bias_class)
            logp = np.log(np.clip(p, 1e-300, None))
            logp = logp + float(bias) * (np.arange(p.size) == idx)
            logp = logp - logp.max()
            e = np.exp(logp)
            p = e / e.sum()
        return p

    def _pred(self, p: np.ndarray) -> _StubPrediction:
        order = np.argsort(-p)
        best = int(order[0])
        second = float(p[int(order[1])]) if len(order) > 1 else 0.0
        return _StubPrediction(
            command=self.classes_[best],
            confidence=float(p[best]),
            scores={c: float(p[i]) for i, c in enumerate(self.classes_)},
            margin=float(p[best] - second))

    def predict_features(self, seq, bias=None, bias_class=None) -> _StubPrediction:
        return self._pred(self._probs(bias, bias_class))

    def predict_detailed(self, seq, bias=None, bias_class=None):
        extras: Dict[str, object] = {'embedding': np.zeros(8, dtype=np.float64)}
        if self.seed_probs is not None:
            extras['seed_probs'] = np.asarray(self.seed_probs, dtype=np.float64)
        return self._pred(self._probs(bias, bias_class)), extras


class _StubExecutor:
    """记录下发的执行器；``async_ack`` 可切，用来测 ACK 闭环。"""

    def __init__(self, async_ack: bool = False, ack_ok: bool = True):
        self.async_ack = bool(async_ack)
        self.ack_ok = bool(ack_ok)
        self.dispatched: List[str] = []
        self.waited = 0

    def dispatch(self, command: str) -> Dict[str, object]:
        self.dispatched.append(command)
        return {'command': command, 'ok': True, 'detail': 'sent'}

    def execute(self, command: str) -> Dict[str, object]:
        """单遍路径用的是 execute（= 下发 + 等回执）。"""
        info = dict(self.dispatch(command))
        if self.async_ack:
            info.update(self.wait_ack())
        return info

    def wait_ack(self, timeout=None) -> Dict[str, object]:
        self.waited += 1
        return {'ok': self.ack_ok, 'detail': 'ack'}


class _StubSpeaker:
    """只记录播了哪些资产。"""

    def __init__(self):
        self.spoken: List[str] = []

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


def _speech_like(n_samples: int, seed: int = 7, amp: float = 0.25) -> np.ndarray:
    """一段能量起伏的类语音信号（能同时骗过 VAD 与 Pass-1 的语音帧判定）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / float(FS)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    f0 = 180.0 + 40.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / float(FS)
    x = amp * env * np.sin(phase) + 0.05 * rng.standard_normal(n_samples)
    return x.astype(np.float32)


def _silence(n_samples: int) -> np.ndarray:
    return np.zeros(int(n_samples), dtype=np.float32)


def _build_engine(predictor, **kwargs) -> DialectEngine:
    """造一个两遍引擎：单阈值放行（不接双信号门限），播报同步。"""
    kwargs.setdefault('threshold', 0.0)
    kwargs.setdefault('fast_silence_ms', 120.0)
    kwargs.setdefault('broadcast_async', False)
    kwargs.setdefault('bias', 0.3)
    engine = DialectEngine(predictor,
                            vad=EnergyVad(fs=FS),
                            **kwargs)
    engine.executor = _StubExecutor()
    engine.speaker = _StubSpeaker()
    return engine


# ----------------------------------------------------------------------
# Pass-1：预触发
# ----------------------------------------------------------------------
def test_pass1_fires_after_two_consistent_windows():
    """预触发要连续两个窗证据一致才发，避免单窗抖动就抢跑。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.9, 'LIGHT_OFF': 0.1})
    det = Pass1Detector(pred, fs=FS, window_ms=1000, hop_ms=100,
                        pretrigger_threshold=0.5, min_hits=2)
    audio = _speech_like(int(FS * 1.5))
    events = []
    for i in range(0, audio.size, 1024):
        ev = det.feed(audio[i:i + 1024])
        if ev is not None:
            events.append(ev)
    assert len(events) == 1
    assert events[0].command == 'LIGHT_ON'
    assert events[0].evidence >= 0.5
    assert events[0].latency_ms >= 0.0


def test_pass1_does_not_fire_without_enough_speech():
    """窗里没人说话就不打分（不满足语音帧占比下限）。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.9, 'LIGHT_OFF': 0.1})
    det = Pass1Detector(pred, fs=FS, window_ms=1000, hop_ms=100,
                        pretrigger_threshold=0.5, min_hits=2)
    audio = _silence(int(FS * 1.5))
    fired = False
    for i in range(0, audio.size, 1024):
        if det.feed(audio[i:i + 1024]) is not None:
            fired = True
    assert not fired


def test_pass1_hysteresis_prevents_immediate_refire():
    """证据没掉回阈值的 60% 之前不重新武装，一句话只触发一次。"""
    pred = _StubPredictor(SIX, {'LIGHT_ON': 0.9, 'LIGHT_OFF': 0.02,
                                 'CURTAIN_OPEN': 0.02, 'CURTAIN_CLOSE': 0.02,
                                 'ALL_OFF': 0.02, 'STATUS_ALL': 0.02})
    det = Pass1Detector(pred, fs=FS, window_ms=1000, hop_ms=100,
                        pretrigger_threshold=0.5, min_hits=2)
    audio = _speech_like(int(FS * 1.5))
    count = 0
    for i in range(0, audio.size, 1024):
        if det.feed(audio[i:i + 1024]) is not None:
            count += 1
    assert count == 1

    # 证据还在高位时继续喂，不该再触发
    more = _speech_like(int(FS * 0.5), seed=11)
    for i in range(0, more.size, 1024):
        assert det.feed(more[i:i + 1024]) is None

    # 证据掉到阈值 60% 以下后重新武装
    pred.probs = {c: 1.0 / len(SIX) for c in SIX}
    quiet = _speech_like(int(FS * 0.6), seed=13)
    for i in range(0, quiet.size, 1024):
        det.feed(quiet[i:i + 1024])
    assert det._fired is False


# ----------------------------------------------------------------------
# 快速端点：预触发把静音挂起压短
# ----------------------------------------------------------------------
def test_fast_endpoint_shortens_hangover_after_pretrigger():
    """预触发命中后静音挂起从默认 ~390 ms 压到 fast_silence_ms（120 ms）。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.9, 'LIGHT_OFF': 0.1})
    engine = _build_engine(pred)
    default_hangover = engine._hangover_ms()
    assert default_hangover > 300.0

    seen: Dict[str, object] = {}

    def on_pretrigger(event):
        seen['command'] = event.command
        seen['min_silence_frames'] = int(engine.vad.min_silence_frames)

    engine.on_pretrigger = on_pretrigger

    audio = np.concatenate([_speech_like(int(FS * 1.6)),
                            _silence(int(FS * 0.9))])
    results = []
    for i in range(0, audio.size, 1024):
        results.extend(engine.process_stream_block(audio[i:i + 1024], FS))

    assert seen, '预触发没有命中'
    assert seen['command'] == 'LIGHT_ON'
    assert seen['min_silence_frames'] == 4          # 120 ms / 30 ms 每帧
    assert len(results) == 1
    assert results[0].hangover_ms <= 121.0
    assert results[0].hangover_ms < default_hangover
    assert results[0].pass1_command == 'LIGHT_ON'
    # 一句结束后静音挂起必须恢复，否则下一句会被上一句的快速端点带跑。
    assert engine.vad.min_silence_frames == engine._default_min_silence_frames


def test_hangover_stays_default_without_pretrigger():
    """没命中预触发就不压挂起：延迟预算靠的是预触发，不是无条件提速。"""
    # 6 类均匀分布 -> 最大类只有 1/6，Pass-1 证据永远够不到 0.5 阈值。
    pred = _StubPredictor(SIX, {c: 1.0 / len(SIX) for c in SIX})
    engine = _build_engine(pred)
    audio = np.concatenate([_speech_like(int(FS * 1.6)),
                            _silence(int(FS * 1.2))])
    results = []
    for i in range(0, audio.size, 1024):
        results.extend(engine.process_stream_block(audio[i:i + 1024], FS))
    assert len(results) == 1
    assert results[0].pass1_command == ''
    assert results[0].hangover_ms >= 300.0


def test_soft_candidate_needs_one_window_pretrigger_needs_two():
    """软候选一个窗就成立，严格预触发要连续两窗。

    两者门槛必须分开：预触发是给用户看的反馈，不能抖；快速端点是内部的
    时序决策，只等一个窗。绑在一起时短命令（语音段 < 窗长 + 一跳）结构上
    等不到预触发，静音挂起压在默认 390 ms，「说完 -> 动作」就进不了 300 ms。
    """
    pred = _StubPredictor(SIX, {'LIGHT_ON': 0.9, 'LIGHT_OFF': 0.02,
                                 'CURTAIN_OPEN': 0.02, 'CURTAIN_CLOSE': 0.02,
                                 'ALL_OFF': 0.02, 'STATUS_ALL': 0.02})
    # hop_ms=1000：整段音频只够打一个窗，把「一窗」和「两窗」区分开。
    det = Pass1Detector(pred, fs=FS, window_ms=1000, hop_ms=1000,
                        pretrigger_threshold=0.5, min_hits=2, soft_threshold=0.5)
    audio = _speech_like(int(FS * 1.2))
    events = []
    for i in range(0, audio.size, 1024):
        ev = det.feed(audio[i:i + 1024])
        if ev is not None:
            events.append(ev)
    assert events == [], '一个窗不该触发严格预触发'
    assert det.soft_candidate is not None
    assert det.soft_candidate[0] == 'LIGHT_ON'
    assert det.soft_candidate[1] >= 0.5


def test_soft_candidate_absent_when_window_is_weak():
    """单窗后验不够高就没有软候选：不能凭猜的候选去压挂起。"""
    pred = _StubPredictor(SIX, {c: 1.0 / len(SIX) for c in SIX})
    det = Pass1Detector(pred, fs=FS, window_ms=1000, hop_ms=1000,
                        pretrigger_threshold=0.5, min_hits=2, soft_threshold=0.5)
    audio = _speech_like(int(FS * 1.2))
    for i in range(0, audio.size, 1024):
        det.feed(audio[i:i + 1024])
    assert det.soft_candidate is None


def test_fast_endpoint_engages_without_strict_pretrigger():
    """只有软候选（没有严格预触发）时，快速端点照样把挂起压短。"""
    pred = _StubPredictor(SIX, {'LIGHT_ON': 0.9, 'LIGHT_OFF': 0.02,
                                 'CURTAIN_OPEN': 0.02, 'CURTAIN_CLOSE': 0.02,
                                 'ALL_OFF': 0.02, 'STATUS_ALL': 0.02})
    # hop_ms=1000 -> 只会打一个窗 -> 严格预触发（要两窗）不会成立。
    engine = _build_engine(pred, window_ms=1000, hop_ms=1000)
    default_hangover = engine._hangover_ms()
    assert default_hangover > 300.0

    audio = np.concatenate([_speech_like(int(FS * 1.2)),
                            _silence(int(FS * 0.9))])
    results = []
    hangovers = []
    for i in range(0, audio.size, 1024):
        results.extend(engine.process_stream_block(audio[i:i + 1024], FS))
        hangovers.append(int(engine.vad.min_silence_frames))

    assert len(results) == 1
    res = results[0]
    assert res.pass1_command == '', '严格预触发不该命中'
    assert min(hangovers) == 4, '软候选命中后挂起应被压到 fast_silence_ms'
    assert res.hangover_ms <= 121.0
    assert res.hangover_ms < default_hangover


def test_fast_endpoint_stays_off_without_soft_candidate():
    """没有软候选就不压挂起：弱窗不能换来一个提前切段的错误。"""
    pred = _StubPredictor(SIX, {c: 1.0 / len(SIX) for c in SIX})
    engine = _build_engine(pred, window_ms=1000, hop_ms=1000)
    audio = np.concatenate([_speech_like(int(FS * 1.2)),
                            _silence(int(FS * 1.2))])
    results = []
    hangovers = []
    for i in range(0, audio.size, 1024):
        results.extend(engine.process_stream_block(audio[i:i + 1024], FS))
        hangovers.append(int(engine.vad.min_silence_frames))

    assert len(results) == 1
    res = results[0]
    assert res.pass1_command == ''
    assert min(hangovers) == engine._default_min_silence_frames
    assert res.hangover_ms >= 300.0


def test_action_latency_is_hangover_plus_cpu():
    """动作延迟 = 音频时间（挂起）+ CPU 时间（复核/门限/决策/下发）。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.9, 'LIGHT_OFF': 0.1})
    engine = _build_engine(pred)
    audio = np.concatenate([_speech_like(int(FS * 1.6)),
                            _silence(int(FS * 0.9))])
    results = []
    for i in range(0, audio.size, 1024):
        results.extend(engine.process_stream_block(audio[i:i + 1024], FS))
    res = results[0]
    assert res.executed
    assert res.hangover_ms > 0
    assert res.latency_ms > 0
    assert res.action_latency_ms >= 0
    total = res.hangover_ms + res.latency_ms + res.action_latency_ms
    assert total < 300.0, f'动作延迟 {total:.1f} ms 超出 300 ms 预算'


# ----------------------------------------------------------------------
# 两遍一致性
# ----------------------------------------------------------------------
def _segment(engine) -> np.ndarray:
    return _speech_like(int(FS * 0.8))


def test_consistent_candidate_executes():
    """候选与 Pass-2 top-1 一致 -> 绿区执行。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.40, 'LIGHT_OFF': 0.60})
    engine = _build_engine(pred)
    res = engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    assert res.consistency == 'consistent'
    assert res.action == 'execute'
    assert res.executed
    assert engine.executor.dispatched == ['LIGHT_OFF']


def test_inconsistent_but_decisive_is_not_downgraded():
    """Pass-1 猜错但 Pass-2 整句很自信（margin 大）时不降级。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.85, 'LIGHT_OFF': 0.15})
    engine = _build_engine(pred)
    res = engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    assert res.consistency == 'inconsistent'
    assert res.action == 'execute'
    assert res.executed
    assert engine.executor.dispatched == ['LIGHT_ON']


def test_inconsistent_and_unsure_is_downgraded():
    """不一致且 margin 小 -> 强制降级，不得执行。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.60, 'LIGHT_OFF': 0.40})
    engine = _build_engine(pred)
    res = engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    assert res.consistency == 'inconsistent'
    assert res.action != 'execute'
    assert not res.executed
    assert res.reason == 'pass2_inconsistent'
    assert engine.executor.dispatched == []


def test_inconsistent_unsure_on_confusable_pair_asks_for_clarification():
    """不一致 + 易混对 + 拿不准 -> 澄清（而不是执行或直接拒识）。"""
    pred = _StubPredictor(['CURTAIN_OPEN', 'CURTAIN_CLOSE'],
                          {'CURTAIN_OPEN': 0.55, 'CURTAIN_CLOSE': 0.45})
    engine = _build_engine(pred, bias=0.0)
    res = engine.process_segment(_segment(engine), FS,
                                 candidate='CURTAIN_CLOSE')
    assert res.consistency == 'inconsistent'
    assert res.action == 'clarify'
    assert res.zone == 'yellow'
    assert not res.executed
    assert res.clarify['candidates'] == ['CURTAIN_OPEN', 'CURTAIN_CLOSE']
    assert 'CLARIFY' in engine.speaker.spoken


def test_seed_disagreement_opens_yellow_zone():
    """各种子 argmax 不一致 -> 判为高风险 -> 走澄清。"""
    pred = _StubPredictor(['CURTAIN_OPEN', 'CURTAIN_CLOSE'],
                          {'CURTAIN_OPEN': 0.60, 'CURTAIN_CLOSE': 0.40},
                          seed_probs=[[0.9, 0.1], [0.1, 0.9]])
    engine = _build_engine(pred)
    res = engine.process_segment(_segment(engine), FS)
    assert res.action == 'clarify'
    assert res.zone == 'yellow'
    assert res.signals.get('seed_agree') == 0.0


def test_no_candidate_is_not_treated_as_inconsistent():
    """没跑 Pass-1 就没候选，不做一致性判断，按单遍处理。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.9, 'LIGHT_OFF': 0.1})
    engine = _build_engine(pred)
    res = engine.process_segment(_segment(engine), FS)
    assert res.consistency == 'no_candidate'
    assert res.action == 'execute'
    assert res.executed


# ----------------------------------------------------------------------
# 去抖
# ----------------------------------------------------------------------
def test_debounce_suppresses_repeated_execution():
    """同一命令在去抖窗口内只执行一次；识别照常，抑制的是执行。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.40, 'LIGHT_OFF': 0.60})
    engine = _build_engine(pred)
    first = engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    second = engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    assert first.executed and not first.debounced
    assert not second.executed and second.debounced
    assert second.reason == 'debounced'
    assert second.action == 'execute'      # 决策仍然是执行，只是被去抖拦下
    assert engine.executor.dispatched == ['LIGHT_OFF']


def test_debounce_does_not_block_a_different_command():
    """去抖按命令分别计数：另一条命令不该被上一条压住。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.40, 'LIGHT_OFF': 0.60})
    engine = _build_engine(pred)
    engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    pred.probs = {'LIGHT_ON': 0.90, 'LIGHT_OFF': 0.10}
    res = engine.process_segment(_segment(engine), FS, candidate='LIGHT_ON')
    assert res.executed
    assert engine.executor.dispatched == ['LIGHT_OFF', 'LIGHT_ON']


# ----------------------------------------------------------------------
# ACK 闭环与播报
# ----------------------------------------------------------------------
def test_ack_closed_loop_dispatch_then_wait_then_confirm():
    """动作先下发，播报随后；回执到了才播「已完成」。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.40, 'LIGHT_OFF': 0.60})
    engine = _build_engine(pred)
    engine.executor = _StubExecutor(async_ack=True)
    res = engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    assert res.executed
    assert engine.executor.dispatched == ['LIGHT_OFF']
    assert engine.executor.waited == 1
    assert 'ACK_DONE' in engine.speaker.spoken
    assert res.spoken


def test_failed_ack_speaks_error_instead_of_done():
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.40, 'LIGHT_OFF': 0.60})
    engine = _build_engine(pred)
    engine.executor = _StubExecutor(async_ack=True, ack_ok=False)
    res = engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    assert 'ACK_FAIL' in engine.speaker.spoken
    assert 'ACK_DONE' not in engine.speaker.spoken
    assert res.spoken


def test_two_pass_path_does_not_broadcast_the_command_again():
    """两遍路径里执行器已经控了设备，播报只能是确认语，不能再念一遍命令。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.40, 'LIGHT_OFF': 0.60})
    engine = _build_engine(pred)
    res = engine.process_segment(_segment(engine), FS, candidate='LIGHT_OFF')
    assert 'LIGHT_OFF' not in engine.speaker.spoken
    assert 'ACK_DONE' in engine.speaker.spoken


def test_single_pass_path_still_broadcasts_the_command():
    """单遍路径语义不同：那里的普通话播报**就是**执行本身，必须保留。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.40, 'LIGHT_OFF': 0.60})
    engine = _build_engine(pred)
    res = engine.process_utterance(_segment(engine), FS)
    assert res.executed
    assert 'LIGHT_OFF' in engine.speaker.spoken


def test_rejected_utterance_speaks_reject():
    """门限拒识 -> 红区 -> 播报可用命令。"""
    pred = _StubPredictor(['LIGHT_ON', 'LIGHT_OFF'],
                          {'LIGHT_ON': 0.40, 'LIGHT_OFF': 0.60})
    engine = _build_engine(pred, threshold=1.01)
    res = engine.process_segment(_segment(engine), FS)
    assert res.action == 'reject'
    assert res.zone == 'red'
    assert not res.executed
    assert 'REJECT_UNKNOWN' in engine.speaker.spoken

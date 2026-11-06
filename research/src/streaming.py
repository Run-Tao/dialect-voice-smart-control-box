# -*- coding: utf-8 -*-
"""Pass-1 流式候选检测：环形缓冲 + 滑窗打分 + 证据累积 + 预触发。

对应 v5 方案设计 §6.1 的「机制 1：两遍识别」的前一半：
Pass-1 用 1 s 窗、100 ms 步进的滑窗反复打分同一段正在进行的语音，
累积**逐类证据**，一旦某个命令的证据稳定越过阈值就发出**预触发**事件。
预触发只做反馈（OLED 亮起 / 提示音），**不执行动作**——动作必须等
Pass-2 整句复核通过（见 ``verify.py``）。

为什么不用能量阈值直接预触发
----------------------------
语音开始的那一刻能量就已经很高，但「高能量」不等于「是命令」；
本项目的真实域拒识实测显示，真实非命令语音的能量与命令语音高度重叠
（见 docs/v5方案设计.md §9.1）。所以预触发条件是**分类器证据**而不是能量，
能量只用来判断「窗里有没有人在说话」。

模块自包含：只依赖 numpy 与 ``features_v3``，不依赖 ``src/vad.py``，
因此可以整体拷到端侧（runtime/ 目录的要求）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features_v3 import extract_sequence

__all__ = ['RingBuffer', 'Pass1Event', 'Pass1Detector']


class RingBuffer:
    """定长环形缓冲：写入新样本、按时序读出最近 N 个样本。

    用 ``numpy`` 的预分配数组而不是 ``deque``，是为了避免每 10 ms
    一次的拼接开销——Pass-1 每秒要跑 100 次窗口读取。
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('capacity 必须 >= 1')
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def write(self, x) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        n = min(x.size, self.capacity)
        if n < x.size:
            x = x[-n:]
        idx = (self._pos + np.arange(n)) % self.capacity
        self._buf[idx] = x
        self._pos = int((self._pos + n) % self.capacity)
        self._size = int(min(self._size + n, self.capacity))

    def read(self) -> np.ndarray:
        """按时序返回当前缓冲的全部样本（长度可能小于 capacity）。"""
        if self._size == 0:
            return np.zeros(0, dtype=np.float32)
        end = self._pos
        start = (end - self._size) % self.capacity
        if start < end:
            return self._buf[start:end].copy()
        return np.concatenate([self._buf[start:], self._buf[:end]])

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._buf[:] = 0.0
        self._pos = 0
        self._size = 0


@dataclass
class Pass1Event:
    """一次预触发事件。"""

    command: str
    score: float                 # 该命令当前窗内的后验
    evidence: float              # 累积证据（跨窗 EMA）
    sample_pos: int              # 事件触发时已写入的总样本数
    latency_ms: float = 0.0      # 从本块音频进来到事件产生的处理耗时

    def as_dict(self) -> dict:
        return {
            'command': self.command,
            'score': round(float(self.score), 4),
            'evidence': round(float(self.evidence), 4),
            'sample_pos': int(self.sample_pos),
            'latency_ms': round(float(self.latency_ms), 1),
        }


class _SpeechTracker:
    """窗内「有没有人在说话」的判据：帧能量 + 自适应噪声底。

    只服务于预触发的**前置条件**（没人说话就别打分），
    不承担切段职责——切段仍由 ``src/vad.py`` 的 ``EnergyVad`` 负责。
    """

    def __init__(self, fs: int = 16000, frame_ms: float = 20.0,
                 margin_db: float = 9.0, alpha: float = 0.05):
        self.frame_len = max(1, int(fs * frame_ms / 1000.0))
        self.margin_db = float(margin_db)
        self.alpha = float(alpha)
        self.noise_db = -55.0
        self._tail: List[np.ndarray] = []

    def push(self, x: np.ndarray) -> None:
        self._tail.append(np.asarray(x, dtype=np.float32).reshape(-1))
        if sum(a.size for a in self._tail) > 4 * self.frame_len:
            self._tail = self._tail[-2:]

    def speech_ratio(self) -> float:
        """返回最近若干帧里「被判为语音」的比例；样本不足时返回 0。"""
        if not self._tail:
            return 0.0
        x = np.concatenate(self._tail)
        if x.size < self.frame_len:
            return 0.0
        n = 1 + (x.size - self.frame_len) // self.frame_len
        frames = np.stack([x[i * self.frame_len:(i + 1) * self.frame_len]
                           for i in range(n)])
        power = np.mean(frames.astype(np.float64) ** 2, axis=1)
        db = 10.0 * np.log10(power + 1e-12)
        voiced = db > (self.noise_db + self.margin_db)
        # 只有判为静音的帧才更新噪声底，否则会把语音学成环境噪声。
        for v, e in zip(voiced, db):
            if not v:
                self.noise_db = (1.0 - self.alpha) * self.noise_db + self.alpha * e
        return float(voiced.mean())


class Pass1Detector:
    """滑窗打分 + 逐类证据累积 + 预触发。

    参数
    ----
    predictor : 有 ``predict_features(seq)`` 接口的预测器（``V3Predictor``）
    window_ms : 滑窗长度（默认 1000 ms）
    hop_ms    : 步进（默认 100 ms）
    pretrigger_threshold : 证据阈值，越过即触发
    evidence_alpha : 证据 EMA 的更新系数（越大越看重当前窗）
    min_hits  : 连续多少个窗的证据都指向同一命令才触发（防止单窗抖动）
    min_speech_ratio : 窗内语音帧占比下限，低于此值不参与打分
    soft_threshold : 单窗后验阈值，越过即认为「用户正在说命令」。
        与 ``pretrigger_threshold`` 分开：预触发是**给用户看的反馈**，
        要求连续两窗一致；快速端点是**内部的时序决策**，只等一个窗。
        两者绑在一起时，短命令（语音段 < 窗长 + 一跳）永远等不到预触发，
        静音挂起压在默认 390 ms，「说完 -> 动作」就进不了 300 ms 预算。
    """

    def __init__(self, predictor, fs: int = 16000, window_ms: float = 1000.0,
                 hop_ms: float = 100.0, pretrigger_threshold: float = 0.5,
                 evidence_alpha: float = 0.5, min_hits: int = 2,
                 min_speech_ratio: float = 0.25,
                 soft_threshold: Optional[float] = None):
        self.predictor = predictor
        self.fs = int(fs)
        self.window_samples = max(1, int(self.fs * window_ms / 1000.0))
        self.hop_samples = max(1, int(self.fs * hop_ms / 1000.0))
        self.pretrigger_threshold = float(pretrigger_threshold)
        self.evidence_alpha = float(evidence_alpha)
        self.min_hits = max(1, int(min_hits))
        self.min_speech_ratio = float(min_speech_ratio)
        self.soft_threshold = float(
            pretrigger_threshold if soft_threshold is None else soft_threshold)
        self._buf = RingBuffer(self.window_samples)
        self._speech = _SpeechTracker(fs=self.fs)
        self._pending = 0
        self._total = 0
        self._evidence: Dict[str, float] = {}
        self._hits: Dict[str, int] = {}
        self._fired = False
        # 软候选：最近一个「单窗后验越过 soft_threshold」的窗。
        # 只在 reset() 里清空——后续窗口分数再低也不撤销，
        # 因为「用户说过一句像命令的话」这个判断在一条语音内不该翻来覆去。
        self._soft_class: str = ''
        self._soft_score: float = 0.0

    # ------------------------------------------------------------------
    @property
    def classes_(self) -> List[str]:
        return list(self.predictor.classes_)

    @property
    def evidence(self) -> Dict[str, float]:
        return dict(self._evidence)

    @property
    def soft_candidate(self) -> Optional[Tuple[str, float]]:
        """最近一个「单窗后验 ≥ ``soft_threshold``」的窗，返回 (类, 后验)。

        没有这样的窗时返回 None。用于**快速端点**：它只要求「用户此刻在说
        像命令的话」，不要求连续两窗一致，所以短命令也能命中。
        """
        if not self._soft_class:
            return None
        return (self._soft_class, self._soft_score)

    def reset(self) -> None:
        self._buf.clear()
        self._speech = _SpeechTracker(fs=self.fs)
        self._pending = 0
        self._total = 0
        self._evidence.clear()
        self._hits.clear()
        self._fired = False
        self._soft_class = ''
        self._soft_score = 0.0

    # ------------------------------------------------------------------
    def feed(self, audio) -> Optional[Pass1Event]:
        """送入一块音频；有新的窗口需要打分时返回预触发事件，否则返回 None。

        事件是**边沿**触发：一次语音只触发一次（``_fired`` 闩锁），
        证据回落到阈值的 60% 以下才重新武装，避免一句话连发多次。
        """
        import time

        t0 = time.perf_counter()
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return None
        self._buf.write(x)
        self._speech.push(x)
        self._total += int(x.size)
        self._pending += int(x.size)

        if self._pending < self.hop_samples:
            return None
        self._pending = 0

        window = self._buf.read()
        if window.size < self.window_samples:
            return None
        if self._speech.speech_ratio() < self.min_speech_ratio:
            return None

        spec = getattr(self.predictor, 'feature_spec', None)
        seq = extract_sequence(window, self.fs, spec)
        pred = self.predictor.predict_features(seq)
        scores = {c: float(pred.scores.get(c, 0.0)) for c in self.classes_}

        # 软候选：这一窗的后验本身就够高就算「用户在说命令」。
        # 与下面的严格预触发分开——预触发要连续两窗一致（给用户看的反馈
        # 不能抖），快速端点只要一个窗（延迟预算不能等两窗）。
        top_class = max(scores, key=lambda c: scores[c])
        top_score = float(scores[top_class])
        if top_score >= self.soft_threshold:
            self._soft_class = top_class
            self._soft_score = top_score

        alpha = self.evidence_alpha
        for c, s in scores.items():
            prev = self._evidence.get(c, 0.0)
            self._evidence[c] = alpha * s + (1.0 - alpha) * prev

        candidate = max(self._evidence, key=lambda c: self._evidence[c])
        self._hits[candidate] = self._hits.get(candidate, 0) + 1
        for c in list(self._hits):
            if c != candidate:
                self._hits[c] = 0

        ev = self._evidence[candidate]
        if self._fired:
            # 迟滞：证据回落到阈值的 60% 才重新武装
            if ev < self.pretrigger_threshold * 0.6:
                self._fired = False
            return None

        if ev >= self.pretrigger_threshold and self._hits[candidate] >= self.min_hits:
            self._fired = True
            return Pass1Event(
                command=candidate,
                score=scores.get(candidate, 0.0),
                evidence=ev,
                sample_pos=self._total,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        return None

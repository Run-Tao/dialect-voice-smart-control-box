# -*- coding: utf-8 -*-
"""Pass-2 整句复核：关键词偏置 + 集成方差 + top-2 margin + 一致性判定。

对应 v5 方案设计 §6.1 的「机制 1：两遍识别」的后一半：
Pass-1（``streaming.py``）在流式过程中给出候选命令，
Pass-2 对**整句**做一次完整推理，并把 Pass-1 的候选当作关键词偏置
加到该类的 logit 上（U2-KWS 式 keyword bias 的简化版）。

偏置为什么不能太大
------------------
偏置的作用是**在整句推理被上下文带偏时把候选拉回来**，不是替候选拍板。
所以偏置幅度必须小到「真正的强烈冲突仍然冲突」：
实测（测试集 483 条）里窗帘开/关的误判是**自信的错**——
错判样本的 top-1 后验中位数约 0.90、top-2 只有 0.05 上下，
这种时候任何小幅偏置都翻不动，正好被一致性规则抓成「不一致 → 降级」。

偏置在哪一层生效（这一条决定门限标定能不能用）
------------------------------------------------
偏置加在**每个种子的 logits** 上，随后除以温度走 softmax
（见 ``v3_model._forward``），所以 softmax 输入空间里的有效偏置是
``bias / temperature``（部署模型 temperature=0.5，0.3 的偏置等效 0.6）。

本模块把结果分成两份，**用途不能互换**：

- ``probs``：**带偏置**的后验。决策层看这一份——它才是 Pass-2 真正的输出。
- ``acoustic_probs``：**无偏**后验。门限（``selective.SelectiveGate``）看这一份。
  门限是在无偏后验上标定的工件，拿带偏置的后验去比阈值等于换了口径；
  偏置能把错误候选的后验抬高，用它过门限等于给拒识开后门。

``acoustic_probs`` 不是重跑一次网络得到的，而是**解析还原**的：
加在单一 logit 上的偏置可逆——

    p_biased = softmax(v + b·e_c)  =>  v ∝ log p_biased − b·e_c
    => p_unbiased = softmax(log p_biased − b·e_c)

这是恒等变形，不是近似。逐种子概率同理还原，所以集成分歧信号
仍然是「无偏口径下各种子到底分不分歧」，不会被偏置人为抹平成一致。

一致性规则
----------
- 一致：Pass-1 候选 == Pass-2（偏置后）的 top-1 → 进决策层。
- 不一致：偏置之后仍然不是候选 → **强制降级**（黄区或红区），不得执行。
- 无候选（没跑 Pass-1）：不做一致性判断，按单遍处理。

模块自包含：只依赖 numpy 与 ``features_v3``。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .features_v3 import extract_sequence

__all__ = ['Pass2Result', 'Pass2Verifier', 'CONSISTENT', 'INCONSISTENT',
           'NO_CANDIDATE', 'unbias', 'seed_stats']

CONSISTENT = 'consistent'
INCONSISTENT = 'inconsistent'
NO_CANDIDATE = 'no_candidate'


@dataclass
class Pass2Result:
    """Pass-2 复核结果。

    ``probs`` 带偏置（决策层用），``acoustic_probs`` 无偏（门限用）。
    """
    command: str
    probs: np.ndarray
    classes: List[str]
    top1: str
    top2: str
    margin: float
    seed_std: float            # 各种子后验的标准差均值（越大越不一致）
    seed_agree: bool           # 各种子 argmax 是否一致
    candidate: Optional[str]   # Pass-1 候选（若有）
    biased: bool               # 是否真的施加了关键词偏置
    consistency: str
    runner_up_prob: float = 0.0
    acoustic_probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    embedding: Optional[np.ndarray] = None

    def as_dict(self) -> dict:
        return {
            'command': self.command,
            'top1': self.top1,
            'top2': self.top2,
            'margin': round(float(self.margin), 4),
            'seed_std': round(float(self.seed_std), 6),
            'seed_agree': bool(self.seed_agree),
            'candidate': self.candidate,
            'biased': bool(self.biased),
            'consistency': self.consistency,
            'runner_up_prob': round(float(self.runner_up_prob), 6),
        }


def _normalize(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError('后验非法')
    return arr / total


def _softmax1(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def unbias(prob, bias: float, index: int) -> np.ndarray:
    """去掉加在单一 logit 上的关键词偏置，返回无偏后验。

    见模块 docstring 的恒等式。``index < 0`` 或 ``bias == 0`` 时原样返回。
    """
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    if index < 0 or float(bias) == 0.0:
        return _normalize(p)
    logp = np.log(np.clip(p, 1e-300, None))
    logp = logp - float(bias) * (np.arange(logp.size) == int(index))
    return _softmax1(logp)


def seed_stats(seed_probs) -> tuple:
    """由逐种子概率算集成方差与一致性。

    ``seed_probs`` 形状为 [S, C] 或 [S*V, C]（V = TTA 变体数），
    为 None 时返回「无分歧」。
    返回 ``(seed_std, seed_agree)``：各类概率在种子间的标准差均值、
    以及各种子 argmax 是否一致。
    """
    if seed_probs is None:
        return 0.0, True
    sp = np.asarray(seed_probs, dtype=np.float64)
    if sp.ndim != 2 or sp.shape[0] < 2:
        return 0.0, True
    std = float(sp.std(axis=0).mean())
    agree = bool((sp.argmax(axis=1) == sp.argmax(axis=1)[0]).all())
    return std, agree


def _supports_bias(fn) -> bool:
    """预测器是否支持 ``bias`` / ``bias_class`` 关键字（用签名判断，不靠捕获异常）。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return 'bias' in params and 'bias_class' in params


class Pass2Verifier:
    """整句复核器。

    参数
    ----
    predictor : ``V3Predictor``（需支持 ``predict_detailed`` 才能给出集成方差）
    bias : 关键词偏置幅度（加在 logits 上）。默认 0.3：部署模型温度 0.5，
        折算到 softmax 输入空间是 0.6，足以在近似平局时把候选拉回，
        不足以盖过强烈冲突。
    """
    def __init__(self, predictor, bias: float = 0.3):
        self.predictor = predictor
        self.bias = float(bias)

    # ------------------------------------------------------------------
    @property
    def temperature(self) -> float:
        return float(getattr(self.predictor, 'temperature', 1.0) or 1.0)

    def verify(self, audio, sr: int = 16000,
               candidate: Optional[str] = None) -> Pass2Result:
        """对整句音频做复核；``candidate`` 为 Pass-1 候选（可为 None）。"""
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            raise ValueError('音频为空')
        spec = getattr(self.predictor, 'feature_spec', None)
        seq = extract_sequence(x, int(sr), spec)
        classes = list(self.predictor.classes_)

        usable = bool(candidate) and candidate in classes and self.bias != 0.0
        index = classes.index(candidate) if usable else -1

        extras = None
        applied = False
        if hasattr(self.predictor, 'predict_detailed'):
            fn = self.predictor.predict_detailed
            if _supports_bias(fn):
                pred, extras = fn(seq, bias=self.bias if usable else None,
                                  bias_class=candidate if usable else None)
                applied = usable
            else:
                pred, extras = fn(seq)
        else:
            fn = self.predictor.predict_features
            if _supports_bias(fn):
                pred = fn(seq, bias=self.bias if usable else None,
                          bias_class=candidate if usable else None)
                applied = usable
            else:
                pred = fn(seq)

        probs = _normalize([pred.scores.get(c, 0.0) for c in classes])

        # softmax 输入空间里的有效偏置：logits 上加 bias，再除以温度。
        eff_bias = (self.bias / max(self.temperature, 1e-6)) if applied else 0.0

        if applied:
            acoustic = unbias(probs, eff_bias, index)
        else:
            acoustic = probs

        raw_seed = (extras or {}).get('seed_probs') if extras else None
        if raw_seed is not None:
            rows = np.asarray(raw_seed, dtype=np.float64)
            if applied:
                rows = np.stack([unbias(row, eff_bias, index) for row in rows])
            seed_std, seed_agree = seed_stats(rows)
        else:
            seed_std, seed_agree = seed_stats(None)

        order = np.argsort(-probs)
        top1 = classes[int(order[0])]
        top2 = classes[int(order[1])] if len(order) > 1 else ''
        margin = float(probs[int(order[0])] - (probs[int(order[1])]
                                               if len(order) > 1 else 0.0))
        runner_up = float(probs[int(order[1])]) if len(order) > 1 else 0.0

        if not candidate:
            consistency = NO_CANDIDATE
        elif top1 == candidate:
            consistency = CONSISTENT
        else:
            consistency = INCONSISTENT

        return Pass2Result(
            command=top1, probs=probs, classes=classes, top1=top1, top2=top2,
            margin=margin, seed_std=seed_std, seed_agree=seed_agree,
            candidate=candidate, biased=applied, consistency=consistency,
            runner_up_prob=runner_up, acoustic_probs=acoustic,
            embedding=(extras or {}).get('embedding') if extras else None,
        )

# -*- coding: utf-8 -*-
"""决策层：状态先验融合 + 期望代价三态决策 + 易混对表。

对应 v5 方案设计 §6.3（机制 3）与 §6.7（澄清策略）。

三态
----
- ``execute`` 绿区：声学证据明确（门限已过、top-2 竞争者不构成混淆）→ 直接执行。
- ``clarify`` 黄区：top-2 命中**已知易混对**且该对的竞争者不可忽略 → 问一句让用户重说。
- ``reject``  红区：门限拒识（选择性分类判定为「不像命令」）→ 播报可用命令。

先验
----
``P(c | audio, state) ∝ P(audio | c) · P(c | state)^τ``，``τ`` 为先验温度
（默认 0.3，弱先验）。**先验只在黄区生效**，且有三重护栏：

1. 先验只能在**声学 top-2 内部**重排，不能凭空造出一个类别；
2. 声学 margin 足够大（默认 0.15）时不让先验翻盘；
3. 黄区执行还要过期望代价这一关，先验推不上去的概率就执行不了。

护栏 1–3 合起来就是「先验错误不得执行」这条反例的实现依据。

期望代价
--------
- ``E_execute(c) = C_act_wrong(c) · (1 − q_c · ρ)``
- ``E_clarify   = C_clarify + (1 − p_retry) · C_reject``
- ``E_reject    = C_reject``

``C_act_wrong`` 按命令风险分级（灯 0.3×、窗帘 0.6×、全部关闭 3.0×、查询 0.1×，
以基础值 10.0 为基准）；``ρ``（``yellow_reliability``，默认 0.5）是**黄区后验的
可信系数**：黄区是模型已知不可信的区域（种子分歧 + 竞争者非忽略），
所以不直接采信后验。

实测依据（测试集 483 条，见 docs/v5方案设计.md §9.4）
--------------------------------------------------
- 易混对（窗帘开/关）的误判是**自信的错**：错判样本 top-1 后验中位数约 0.90，
  margin 中位数 0.90，**靠先验翻不动**，只能靠澄清。
- 单纯用 margin 或先验都无法把易混对拉出来；真正有判别力的是
  **集成分歧**（各种子 argmax 不一致 / seed_std）与**竞争者概率**。
- 状态先验在测试集上只改动 2–4/483 条（τ=0.3），净效果在噪声量级；
  把 τ 提到 1.0 反而净变差。因此默认 τ 保持弱先验，并且只在黄区生效。
- 两遍一致性**不单独构成降级理由**：Pass-1 的 1 s 滑窗看的是半句话，
  实测「全部關晒」的滑窗常常给 LIGHT_OFF，而 Pass-2 的整句后验
  margin 高达 0.998（maxprob 0.999）。一刀切降级会把**整句证据明确**
  的正确结果扔掉（测试集上 7/30 条 ALL_OFF 因此被拒识）。
  现在的一致性规则是「两遍不一致 **且** Pass-2 的 margin <
  consistency_margin」才降级；margin 够大时分歧只说明 Pass-1 窗口不完整。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ['DecisionCosts', 'Decision', 'decide', 'state_weight',
           'state_weights', 'fuse_prior', 'CONFUSABLE_PAIRS', 'RISK_MULTIPLIER']

# 已知易混对：占测试集全部误判 32.8% 的最大单一误判流向（docs/v5方案设计.md §1.3）
CONFUSABLE_PAIRS: Tuple[Tuple[str, str], ...] = (('CURTAIN_OPEN', 'CURTAIN_CLOSE'),)

# 命令风险分级（相对基准代价的倍数）
RISK_MULTIPLIER: Dict[str, float] = {
    'LIGHT_ON': 0.3,
    'LIGHT_OFF': 0.3,
    'CURTAIN_OPEN': 0.6,
    'CURTAIN_CLOSE': 0.6,
    'ALL_OFF': 3.0,
    'STATUS_ALL': 0.1,
}


@dataclass(frozen=True)
class DecisionCosts:
    """期望代价参数（见模块 docstring 的公式）。

    ``reject`` 默认 4.0 而不是方案 §8.3 草稿里的 1.0：拒识意味着用户要把
    **整句**重说一遍，比「二选一澄清」贵。按草稿的 1.0 算，
    ``E_clarify = 1.5 + 0.1×1.0 = 1.6 > E_reject = 1.0``，
    澄清永远被拒识压过去，§6.7 的澄清机制根本触发不了。
    """

    act_wrong: float = 10.0        # 误动作的基础代价
    reject: float = 4.0            # 拒识代价（用户要整句重说，比澄清贵）
    clarify: float = 1.5           # 澄清代价
    retry_success: float = 0.9     # 用户重说后成功的概率
    yellow_reliability: float = 0.5  # 黄区后验可信系数

    def cost_of(self, command: str) -> float:
        """按命令风险分级后的误动作代价。"""
        return float(self.act_wrong) * float(RISK_MULTIPLIER.get(command, 1.0))

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, float]]) -> 'DecisionCosts':
        """从配置字典构造；只接受已知字段，缺省用类默认值。"""
        values = dict(values or {})
        known = {'act_wrong', 'reject', 'clarify', 'retry_success',
                 'yellow_reliability'}
        return cls(**{k: float(v) for k, v in values.items() if k in known})


@dataclass
class Decision:
    """三态决策结果。"""

    action: str                     # execute / clarify / reject
    command: str                   # 选中的命令（reject 时为空串）
    zone: str                      # green / yellow / red
    reason: str
    probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prior_probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    margin: float = 0.0
    candidates: List[str] = field(default_factory=list)
    expected_costs: Dict[str, float] = field(default_factory=dict)
    prior_applied: bool = False
    signals: Dict[str, float] = field(default_factory=dict)

    @property
    def executed(self) -> bool:
        return self.action == 'execute'

    @property
    def needs_clarify(self) -> bool:
        return self.action == 'clarify'

    def as_dict(self) -> dict:
        return {
            'action': self.action,
            'command': self.command,
            'zone': self.zone,
            'reason': self.reason,
            'margin': round(float(self.margin), 4),
            'candidates': list(self.candidates),
            'expected_costs': {k: round(float(v), 4)
                               for k, v in self.expected_costs.items()},
            'prior_applied': bool(self.prior_applied),
            'signals': {k: round(float(v), 6) for k, v in self.signals.items()},
        }


# ---------------------------------------------------------------------------
# 状态先验
# ---------------------------------------------------------------------------

def state_weight(command: str, state, redundancy_weight: float = 0.25) -> float:
    """命令在当前状态下的合理程度。

    1.0 = 说得通；``redundancy_weight`` = 冗余（当前状态已经满足这个动作，
    用户再说一遍的概率较低）。按**器件分组**给权重，不做 6 类全局归一化——
    全局归一会把组间信息稀释掉（1/5.6 vs 1/5.6），先验退化成常数，
    这是实现时实测到的，所以这里保留分组尺度。
    """
    if state is None:
        return 1.0
    r = float(redundancy_weight)
    light = getattr(state, 'light', 'off')
    curtain = getattr(state, 'curtain', 'closed')
    if command == 'LIGHT_ON':
        return r if light == 'on' else 1.0
    if command == 'LIGHT_OFF':
        return r if light == 'off' else 1.0
    if command == 'CURTAIN_OPEN':
        return r if curtain == 'open' else 1.0
    if command == 'CURTAIN_CLOSE':
        return r if curtain == 'closed' else 1.0
    if command == 'ALL_OFF':
        return r if (light == 'off' and curtain == 'closed') else 1.0
    return 1.0  # STATUS_ALL：查询永远说得通


def state_weights(classes: Sequence[str], state,
                  redundancy_weight: float = 0.25) -> np.ndarray:
    return np.asarray([state_weight(c, state, redundancy_weight)
                        for c in classes], dtype=np.float64)


def fuse_prior(prob: np.ndarray, weights: np.ndarray,
               temperature: float = 0.3) -> np.ndarray:
    """``P(c|audio,state) ∝ P(audio|c) · w(c)^τ``，返回归一化结果。"""
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if p.shape != w.shape:
        raise ValueError('prob 与 weights 长度不一致')
    total = p.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError('prob 非法')
    p = p / total
    tau = float(temperature)
    if tau <= 0:
        return p
    w = np.clip(w, 1e-6, None)
    q = p * np.power(w, tau)
    s = q.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError('先验融合结果非法')
    return q / s


# ---------------------------------------------------------------------------
# 三态决策
# ---------------------------------------------------------------------------

def _pair_hit(top1: str, top2: str,
              confusable_pairs: Sequence[Sequence[str]]) -> bool:
    for pair in confusable_pairs:
        members = set(str(x) for x in pair)
        if members == {top1, top2}:
            return True
    return False


def decide(prob, classes: Sequence[str], *,
           state=None,
           gate_accepted: bool = True,
           gate_reason: str = '',
           costs: Optional[DecisionCosts] = None,
           confusable_pairs: Optional[Sequence[Sequence[str]]] = None,
           competitor_floor: float = 0.005,
           prior_temperature: float = 0.3,
           redundancy_weight: float = 0.25,
           prior_override_margin: float = 0.15,
           seed_agree: Optional[bool] = None,
           seed_std: Optional[float] = None,
           consistency: Optional[str] = None,
           consistency_margin: float = 0.15,
           clarify_allowed: bool = True) -> Decision:
    """返回三态决策。

    参数
    ----
    prob : 各类后验（会被归一化）
    classes : 类别名，顺序与 ``prob`` 一致
    state : ``DeviceState``（可为 None = 不使用先验）
    gate_accepted : 选择性分类门限是否放行；False 直接进红区
    consistency : 两遍一致性（``consistent`` / ``inconsistent`` / None）。
        ``inconsistent`` 时**还要看 Pass-2 自己够不够自信**：
        margin < ``consistency_margin`` 才强制降级（只允许澄清或拒识，
        **不得执行**）；margin 够大说明整句证据没有分歧，分歧只来自
        Pass-1 的滑窗看的是半句话，这时降级反而会扔掉正确结果。
    clarify_allowed : 是否允许澄清（会话内最多一次，由 ``dialog.py`` 控制）
    """
    costs = costs or DecisionCosts()
    pairs = list(confusable_pairs if confusable_pairs is not None
                 else CONFUSABLE_PAIRS)
    classes = list(classes)
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    if p.shape[0] != len(classes):
        raise ValueError('prob 与 classes 长度不一致')
    total = p.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError('prob 非法')
    p = p / total

    order = np.argsort(-p)
    top1 = classes[int(order[0])]
    top2 = classes[int(order[1])] if len(order) > 1 else ''
    margin = float(p[int(order[0])] - (p[int(order[1])]
                                       if len(order) > 1 else 0.0))
    runner_up = float(p[int(order[1])]) if len(order) > 1 else 0.0

    signals = {
        'maxprob': float(p[int(order[0])]),
        'runner_up': runner_up,
        'margin': margin,
        'competitor_floor': float(competitor_floor),
    }
    if seed_agree is not None:
        signals['seed_agree'] = 1.0 if seed_agree else 0.0
    if seed_std is not None:
        signals['seed_std'] = float(seed_std)

    # ---- 红区：门限拒识 -------------------------------------------------
    if not gate_accepted:
        return Decision(action='reject', command='', zone='red',
                        reason=gate_reason or 'gate_rejected',
                        probs=p, margin=margin, candidates=[top1, top2],
                        expected_costs={'reject': float(costs.reject)},
                        signals=signals)

    # ---- 黄区判定 -------------------------------------------------------
    pair_hit = _pair_hit(top1, top2, pairs)
    risky = bool(runner_up >= float(competitor_floor))
    if seed_agree is False:
        risky = True
    if seed_std is not None and float(seed_std) >= 0.02:
        risky = True
    yellow = pair_hit and risky

    degraded = (consistency == 'inconsistent'
                and margin < float(consistency_margin))
    signals['consistency_degraded'] = 1.0 if degraded else 0.0

    if not yellow:
        # 绿区：先验不生效（方案 §6.3：只在黄区生效），直接执行声学 top-1。
        cost = costs.cost_of(top1)
        e_exec = cost * (1.0 - float(p[int(order[0])]))
        if degraded:
            # 两遍不一致：强制降级，不执行。
            e_clarify = (float(costs.clarify)
                         + (1.0 - float(costs.retry_success)) * float(costs.reject))
            if clarify_allowed and pair_hit:
                return Decision(action='clarify', command=top1, zone='yellow',
                                reason='pass2_inconsistent',
                                probs=p, margin=margin, candidates=[top1, top2],
                                expected_costs={'execute': e_exec, 'clarify': e_clarify},
                                signals=signals)
            return Decision(action='reject', command='', zone='red',
                            reason='pass2_inconsistent',
                            probs=p, margin=margin, candidates=[top1, top2],
                            expected_costs={'execute': e_exec, 'reject': float(costs.reject)},
                            signals=signals)
        return Decision(action='execute', command=top1, zone='green',
                        reason='green', probs=p, margin=margin,
                        candidates=[top1, top2],
                        expected_costs={'execute': e_exec,
                                        'reject': float(costs.reject)},
                        signals=signals)

    # ---- 黄区：先验融合（只在黄区生效）---------------------------------
    w = state_weights(classes, state, redundancy_weight)
    q = fuse_prior(p, w, prior_temperature)
    qorder = np.argsort(-q)
    q_top1 = classes[int(qorder[0])]

    # 护栏 1：先验只能在声学 top-2 内部重排
    cand_idx = [int(order[0])] + ([int(order[1])] if len(order) > 1 else [])
    prior_applied = False
    if q_top1 in (top1, top2):
        # 护栏 2：声学 margin 足够大时不让先验翻盘
        if q_top1 == top1 or margin >= float(prior_override_margin):
            chosen = top1
        else:
            chosen = q_top1
            prior_applied = True
    else:
        chosen = top1

    c_idx = classes.index(chosen)
    q_chosen = float(q[c_idx])
    cost = costs.cost_of(chosen)
    reliability = float(costs.yellow_reliability)
    e_exec = cost * (1.0 - q_chosen * reliability)
    e_clarify = (float(costs.clarify)
                 + (1.0 - float(costs.retry_success)) * float(costs.reject))
    e_reject = float(costs.reject)

    options = {'execute': e_exec}
    if pair_hit and clarify_allowed:
        options['clarify'] = e_clarify
    options['reject'] = e_reject

    if degraded:
        # 两遍不一致：执行这一项直接去掉，只剩澄清或拒识。
        options.pop('execute', None)

    best = min(options, key=lambda k: options[k])
    if degraded and best == 'execute':  # 理论上到不了，兜底
        best = 'clarify' if 'clarify' in options else 'reject'

    if best == 'execute':
        action, command, zone = 'execute', chosen, 'green'
        reason = 'yellow_prior_resolved' if prior_applied else 'yellow_execute'
    elif best == 'clarify':
        action, command, zone = 'clarify', chosen, 'yellow'
        # 降级原因是「两遍不一致」时要说清楚，别把它记成普通易混对，
        # 否则日志里看不出这一条是被一致性规则拦下来的。
        if degraded:
            reason = 'pass2_inconsistent'
        else:
            reason = 'confusable_pair' if pair_hit else 'ambiguous'
    else:
        action, command, zone = 'reject', '', 'red'
        reason = 'reject_cost' if not degraded else 'pass2_inconsistent'

    return Decision(action=action, command=command, zone=zone, reason=reason,
                    probs=p, prior_probs=q, margin=margin,
                    candidates=[top1, top2], expected_costs=options,
                    prior_applied=prior_applied, signals=signals)

# -*- coding: utf-8 -*-
"""三态决策层测试：期望代价、状态先验护栏、两遍一致性降级。

重点是两条反例测试（方案 §6.3 要求「必须有反例测试（先验错误时系统
不得执行错误动作）」）：

- ``test_wrong_prior_cannot_flip_confident_acoustics``：先验方向与声学相反
  且声学 margin 足够大时，先验**不得**翻盘，执行的是声学 top-1；
- ``test_wrong_prior_cannot_push_execution_past_cost_test``：先验确实把
  候选翻过来了，但期望代价这一关不允许执行，转澄清。

另外固化一条已知的方案偏离：``cost_reject`` 必须大于 ``cost_clarify``
的量级，否则澄清机制永远触发不了（见 DecisionCosts 的 docstring）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.decision import (CONFUSABLE_PAIRS, DecisionCosts, decide,
                              fuse_prior, state_weight, state_weights)
from runtime.state import DeviceState

CLASSES = ['ALL_OFF', 'CURTAIN_CLOSE', 'CURTAIN_OPEN',
           'LIGHT_OFF', 'LIGHT_ON', 'STATUS_ALL']


def probs(**kwargs):
    """按类名构造后验向量；未提到的类补 0。"""
    v = np.zeros(len(CLASSES), dtype=np.float64)
    for name, value in kwargs.items():
        v[CLASSES.index(name)] = value
    total = v.sum()
    assert total > 0
    return v / total


# ----------------------------------------------------------------------
# 三态本身
# ----------------------------------------------------------------------
def test_green_zone_executes():
    d = decide(probs(LIGHT_ON=0.95, LIGHT_OFF=0.05), CLASSES,
               gate_accepted=True)
    assert d.action == 'execute'
    assert d.command == 'LIGHT_ON'
    assert d.zone == 'green'


def test_gate_rejection_goes_red():
    d = decide(probs(LIGHT_ON=0.95, LIGHT_OFF=0.05), CLASSES,
               gate_accepted=False, gate_reason='maxprob_below')
    assert d.action == 'reject'
    assert d.zone == 'red'
    assert d.reason == 'maxprob_below'


def test_confusable_pair_enters_yellow():
    d = decide(probs(CURTAIN_OPEN=0.55, CURTAIN_CLOSE=0.45), CLASSES,
               gate_accepted=True)
    assert d.zone == 'yellow'
    assert d.candidates[:2] == ['CURTAIN_OPEN', 'CURTAIN_CLOSE']


def test_clarify_is_cheaper_than_reject_and_cheaper_than_wrong_action():
    """黄区的三态由期望代价决定，不是拍脑袋。"""
    costs = DecisionCosts()
    d = decide(probs(CURTAIN_OPEN=0.55, CURTAIN_CLOSE=0.45), CLASSES,
               gate_accepted=True, costs=costs)
    e = d.expected_costs
    assert e['clarify'] < e['reject']
    assert e['clarify'] < e['execute']
    assert d.action == 'clarify'


def test_reject_cost_of_one_makes_clarify_unreachable():
    """方案 §8.3 草稿的 cost_reject=1.0 会让澄清永远触发不了。

    这是被实测推翻并改成 4.0 的原因，固化成测试防止有人改回去。
    """
    costs = DecisionCosts(reject=1.0)
    d = decide(probs(CURTAIN_OPEN=0.55, CURTAIN_CLOSE=0.45), CLASSES,
               gate_accepted=True, costs=costs)
    assert d.expected_costs['clarify'] > d.expected_costs['reject']
    assert d.action == 'reject'


def test_yellow_zone_uses_risk_multiplier_in_execution_cost():
    """黄区的执行代价按命令风险分级（窗帘 0.6×），不是固定值。"""
    d = decide(probs(CURTAIN_OPEN=0.55, CURTAIN_CLOSE=0.45), CLASSES,
               gate_accepted=True)
    assert d.zone == 'yellow'
    # cost_of(CURTAIN_OPEN) = 10 × 0.6 = 6，再按 (1 − q·ρ) 折算
    assert d.expected_costs['execute'] < 6.0
    assert d.expected_costs['execute'] > 4.0


def test_yellow_zone_only_for_known_confusable_pairs():
    """黄区只在**已知易混对**上触发（方案 §6.7），不是所有低 margin 都进。"""
    d = decide(probs(ALL_OFF=0.55, LIGHT_OFF=0.45), CLASSES, gate_accepted=True)
    assert d.zone == 'green'
    assert d.action == 'execute'


# ----------------------------------------------------------------------
# 状态先验与护栏
# ----------------------------------------------------------------------
def test_state_weight_marks_redundant_actions():
    state = DeviceState(light='on', curtain='closed')
    assert state_weight('LIGHT_ON', state) < 1.0     # 灯已经开着，再说一次是冗余
    assert state_weight('LIGHT_OFF', state) == 1.0
    assert state_weight('CURTAIN_CLOSE', state) < 1.0
    assert state_weight('CURTAIN_OPEN', state) == 1.0
    assert state_weight('STATUS_ALL', state) == 1.0  # 查询永远说得通


def test_fuse_prior_keeps_probability_normalized():
    p = probs(CURTAIN_CLOSE=0.5, CURTAIN_OPEN=0.5)
    w = state_weights(CLASSES, DeviceState(curtain='closed'), 0.25)
    q = fuse_prior(p, w, 0.3)
    assert abs(q.sum() - 1.0) < 1e-12
    assert q[CLASSES.index('CURTAIN_OPEN')] > q[CLASSES.index('CURTAIN_CLOSE')]


def test_wrong_prior_cannot_flip_confident_acoustics():
    """反例：窗帘已经关着（先验偏向 OPEN），但用户说的是 CLOSE。

    声学 margin 足够大时先验**不得**翻盘，否则会执行错误动作。
    """
    state = DeviceState(curtain='closed')   # 先验：CLOSE 冗余，OPEN 说得通
    d = decide(probs(CURTAIN_CLOSE=0.60, CURTAIN_OPEN=0.30), CLASSES,
               state=state, gate_accepted=True)
    assert d.prior_applied is False
    assert d.command == 'CURTAIN_CLOSE'


def test_wrong_prior_cannot_push_execution_past_cost_test():
    """反例：声学 margin 很小，先验把候选翻过来了。

    但翻过来的候选可信度不够，期望代价这一关不允许执行，转澄清。
    """
    state = DeviceState(curtain='closed')
    d = decide(probs(CURTAIN_CLOSE=0.30, CURTAIN_OPEN=0.25), CLASSES,
               state=state, gate_accepted=True)
    assert d.prior_applied is True
    assert d.command == 'CURTAIN_OPEN'
    assert d.action != 'execute'


def test_prior_cannot_invent_a_class_outside_top2():
    """护栏 1：先验只能在声学 top-2 内部重排。"""
    state = DeviceState(light='off', curtain='closed')
    p = probs(LIGHT_ON=0.5, LIGHT_OFF=0.3, CURTAIN_OPEN=0.2)
    d = decide(p, CLASSES, state=state, gate_accepted=True)
    assert d.command in ('LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN')
    assert d.command in d.candidates[:2] or d.command == d.candidates[0]


# ----------------------------------------------------------------------
# 两遍一致性
# ----------------------------------------------------------------------
def test_inconsistent_with_small_margin_is_degraded():
    """两遍不一致且 Pass-2 自己也不自信 -> 不得执行。"""
    d = decide(probs(CURTAIN_OPEN=0.30, CURTAIN_CLOSE=0.28), CLASSES,
               gate_accepted=True, consistency='inconsistent')
    assert d.action != 'execute'
    assert d.reason == 'pass2_inconsistent'


def test_inconsistent_with_large_margin_is_not_degraded():
    """两遍不一致但 Pass-2 的整句证据明确 -> 不降级。

    Pass-1 的 1 s 滑窗看的是半句话，实测常把「全部關晒」判成 LIGHT_OFF，
    而 Pass-2 的整句后验 margin 能到 0.998；一刀切降级会扔掉正确结果。
    """
    d = decide(probs(LIGHT_ON=0.999, LIGHT_OFF=0.001), CLASSES,
               gate_accepted=True, consistency='inconsistent')
    assert d.action == 'execute'
    assert d.command == 'LIGHT_ON'
    assert d.signals['consistency_degraded'] == 0.0


def test_no_candidate_does_not_degrade():
    d = decide(probs(LIGHT_ON=0.999, LIGHT_OFF=0.001), CLASSES,
               gate_accepted=True, consistency='no_candidate')
    assert d.action == 'execute'


def test_confusable_pairs_constant_is_the_curtain_pair():
    assert CONFUSABLE_PAIRS == (('CURTAIN_OPEN', 'CURTAIN_CLOSE'),)

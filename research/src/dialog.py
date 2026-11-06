# -*- coding: utf-8 -*-
"""对话层：去抖 + 澄清状态机。

对应 v5 方案设计 §6.5（去抖）与 §6.7（澄清策略）。

去抖
----
同一命令在 ``window_ms``（默认 2000 ms）内只放行一次，防止流式连发或
用户重复说导致重复执行。抑制的是**执行**，不是识别。

澄清
----
黄区且 top-2 命中已知易混对时，向用户播一句二选一的普通话问句，
请用户重说一次；**最多澄清一次**，仍不确定就拒识并播报可用命令。
不追问第二次是刻意的：追问超过一次会让用户放弃，改成一次性把候选摆出来。

模块自包含：只依赖 numpy（其实不依赖 numpy）与标准库。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

__all__ = ['Debouncer', 'ClarifyPrompt', 'ClarifyDialog', 'COMMAND_ZH']

# 命令的普通话短说法（与 src/commands.py 的 zh 字段一致，
# tests/test_dialog.py 会断言两处相同，防止各改一处）。
COMMAND_ZH: Dict[str, str] = {
    'LIGHT_ON': '打开灯',
    'LIGHT_OFF': '关灯',
    'CURTAIN_OPEN': '打开窗帘',
    'CURTAIN_CLOSE': '关上窗帘',
    'ALL_OFF': '全部关闭',
    'STATUS_ALL': '查询设备状态',
}


class Debouncer:
    """同一命令在窗口内抑制重复执行。"""

    def __init__(self, window_ms: int = 2000,
                 clock: Optional[Callable[[], float]] = None):
        self.window_ms = int(window_ms)
        self._clock = clock or time.monotonic
        self._last: Dict[str, float] = {}

    def allow(self, command: str, now: Optional[float] = None) -> bool:
        """返回是否允许执行 ``command``；被抑制时返回 False。"""
        t = self._clock() if now is None else float(now)
        last = self._last.get(command)
        if last is not None and (t - last) * 1000.0 < self.window_ms:
            return False
        self._last[command] = t
        return True

    def suppress_ms(self, command: str, now: Optional[float] = None) -> float:
        """距离上次同命令执行过去了多少毫秒（没执行过返回 -1）。"""
        last = self._last.get(command)
        if last is None:
            return -1.0
        t = self._clock() if now is None else float(now)
        return (t - last) * 1000.0

    def reset(self) -> None:
        self._last.clear()


@dataclass
class ClarifyPrompt:
    """一次澄清请求。"""

    text: str
    candidates: List[str]
    asset: str          # 预录音频文件名（相对普通话音频目录）
    attempt: int        # 第几次澄清（从 1 开始）

    def as_dict(self) -> dict:
        return {
            'text': self.text,
            'candidates': list(self.candidates),
            'asset': self.asset,
            'attempt': int(self.attempt),
        }


def clarify_asset(candidates: Sequence[str]) -> str:
    """澄清问句的音频文件名（按字典序，保证同一对候选总是同一个文件）。"""
    pair = sorted(str(c) for c in candidates)
    if not pair:
        return 'CLARIFY_RETRY.wav'
    if len(pair) == 1:
        return f'CLARIFY_{pair[0]}.wav'
    return f'CLARIFY_{pair[0]}_{pair[1]}.wav'


class ClarifyDialog:
    """澄清状态机。

    状态：
    - ``used == 0``：还没澄清过，可以发起；
    - ``awaiting_retry``：刚发起过澄清，**用户正在重说**——这一次不许再澄清，
      否则会变成「问-答-再问-再答」的死循环；
    - ``used >= max_clarify``：本轮澄清次数用完，调用方应走拒识。

    ``used`` 在一次澄清**有结果之后**清零：澄清成功或失败都算这一轮结束，
    下一条独立的歧义语句仍然可以再澄清一次。限制的是「同一次澄清不许追问」，
    不是「整个会话只能澄清一次」。
    """

    def __init__(self, max_clarify: int = 1, texts: Optional[Dict[str, str]] = None):
        self.max_clarify = max(0, int(max_clarify))
        self.texts = dict(texts or COMMAND_ZH)
        self.used = 0
        self.pending: Optional[ClarifyPrompt] = None
        self.awaiting_retry = False

    # ------------------------------------------------------------------
    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_clarify

    def reset(self) -> None:
        self.used = 0
        self.pending = None
        self.awaiting_retry = False

    # ------------------------------------------------------------------
    def request(self, candidates: Sequence[str]) -> Optional[ClarifyPrompt]:
        """发起一次澄清；已用完次数或候选不足时返回 None。"""
        cands = [str(c) for c in candidates if c]
        if self.awaiting_retry or self.exhausted or len(cands) < 2:
            return None
        self.used += 1
        names = [self.texts.get(c, c) for c in cands]
        text = f'你是要{names[0]}，还是{names[1]}？'
        prompt = ClarifyPrompt(text=text, candidates=cands,
                                asset=clarify_asset(cands), attempt=self.used)
        self.pending = prompt
        self.awaiting_retry = True
        return prompt

    def on_retry(self, success: bool) -> None:
        """用户重说后的结果；``success`` 为 True 表示这次听懂了。

        无论成功与否，这一轮澄清都算结束、额度清零，下一条独立语句可以再澄清。
        """
        self.pending = None
        self.awaiting_retry = False
        self.used = 0

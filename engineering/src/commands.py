# -*- coding: utf-8 -*-
"""命令表：模型类别 -> 中文指令文本 -> 执行动作。

命令集与 runtime 模型的 classes_ 保持一致（6 类）。建议书原先写的 8 类
（开风扇/开空调/调温度等）在本模型上没有训练语料，已在
docs/技术路线修订说明.md 中记录该调整与后续扩展路径。
"""

from typing import Optional

LIGHT = 'light'
CURTAIN = 'curtain'

COMMANDS = {
    'LIGHT_ON': {
        'zh': '打开灯',
        'actuator': LIGHT,
        'action': 'on',
        'device': '灯',
    },
    'LIGHT_OFF': {
        'zh': '关灯',
        'actuator': LIGHT,
        'action': 'off',
        'device': '灯',
    },
    'CURTAIN_OPEN': {
        'zh': '打开窗帘',
        'actuator': CURTAIN,
        'action': 'open',
        'device': '窗帘',
    },
    'CURTAIN_CLOSE': {
        'zh': '关上窗帘',
        'actuator': CURTAIN,
        'action': 'close',
        'device': '窗帘',
    },
    'ALL_OFF': {
        'zh': '全部关闭',
        'actuator': 'all',
        'action': 'off',
        'device': '所有设备',
    },
    'STATUS_ALL': {
        'zh': '查询设备状态',
        'actuator': 'none',
        'action': 'query',
        'device': '所有设备',
    },
}

# 设备状态，供 STATUS_ALL 与日志使用。
INITIAL_STATE = {LIGHT: 'off', CURTAIN: 'closed'}

# 执行后 actuator 应处的状态值。必须与 INITIAL_STATE、runtime.state.DeviceState、
# runtime.decision.state_weight 的取值口径一致：灯 on/off、窗帘 open/closed。
# 注意 action（命令语义，如 CURTAIN_CLOSE='close'）与状态值（'closed'）不同——
# 直接把 action 写进 state 会产生 'close'，与上述三处期望的 'closed' 失配，
# 导致状态先验冗余判断与 ALL_OFF 的「窗帘已关」判断失效（2026-11-19 审查发现）。
_STATE_VALUE = {
    'LIGHT_ON': 'on',
    'LIGHT_OFF': 'off',
    'CURTAIN_OPEN': 'open',
    'CURTAIN_CLOSE': 'closed',
}


def resulting_state(command: str) -> Optional[str]:
    """命令执行后 actuator 的状态值；query/all 类返回 None。"""
    return _STATE_VALUE.get(command)


def zh_text(command: str) -> str:
    """命令类别对应的普通话指令文本。未知命令返回原样，便于排查。"""
    entry = COMMANDS.get(command)
    return entry['zh'] if entry else command


def spoken_text(command: str, wake_word: str = '') -> str:
    """播放给智能音箱听的完整文本，可选前置唤醒词。"""
    text = zh_text(command)
    return f'{wake_word}{text}' if wake_word else text

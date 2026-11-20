# -*- coding: utf-8 -*-
"""命令表：模型类别 -> 中文指令文本 -> 执行动作。

命令集与 runtime 模型的 classes_ 保持一致（6 类）。建议书原先写的 8 类
（开风扇/开空调/调温度等）在本模型上没有训练语料，已在
docs/技术路线修订说明.md 中记录该调整与后续扩展路径。
"""

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


def zh_text(command: str) -> str:
    """命令类别对应的普通话指令文本。未知命令返回原样，便于排查。"""
    entry = COMMANDS.get(command)
    return entry['zh'] if entry else command


def spoken_text(command: str, wake_word: str = '') -> str:
    """播放给智能音箱听的完整文本，可选前置唤醒词。"""
    text = zh_text(command)
    return f'{wake_word}{text}' if wake_word else text

"""方言文本归一化：繁→简、粤语口语词映射、冗余词清理、数字归一。"""
from __future__ import annotations

import re

from opencc import OpenCC

from .textutils import cn2num

_CC = OpenCC("t2s")

# 粤语口语 → 普通话书面词（按长度降序替换，避免子串误伤）
CANTONESE_LEXICON = {
    "冷气机": "空调",
    "冷气": "空调",
    "冷汽": "空调",
    "熄咗": "关了",
    "熄": "关",
    "拉埋": "拉上",
    "较": "调",
    "细声": "小声",
    "冻啲": "冷一点",
    "冻滴": "冷一点",
    "冻一点": "冷一点",
    "暖啲": "热一点",
    "暖滴": "热一点",
    "暖一点": "热一点",
    "凉啲": "冷一点",
    "凉滴": "冷一点",
    "凉一点": "冷一点",
    "热啲": "热一点",
    "光啲": "亮一点",
    "光滴": "亮一点",
    "暗啲": "暗一点",
    "快啲": "快一点",
    "慢啲": "慢一点",
    "大声啲": "大声一点",
    "小声啲": "小声一点",
    "咗": "了",
    "嘅": "的",
    "喺": "在",
    "D": "啲",
    "滴": "啲",
    "啲": "一点",
    "展灯": "盏灯",
    "盏灯": "灯",
    "插苏": "插座",
    "冲凉房": "浴室",
    "煮饭房": "厨房",
    "睡房": "卧室",
}

# 冗余礼貌/填充词（直接删除）
FILLERS = ["唔该", "唔该晒", "麻烦", "帮我", "请", "一下", "唔好意思"]

_LEX_SORTED = sorted(CANTONESE_LEXICON.items(), key=lambda kv: -len(kv[0]))
_NUM_UNIT = re.compile(r"([零一二两三四五六七八九十百千廿卅\d]+)(度|档|个台)")


def normalize(text: str) -> str:
    text = _CC.convert(text.strip())
    # 定点迭代：短规则（D/滴→啲）会在本轮后期产出长规则（冻啲→冷一点）的输入，
    # 单遍长度降序会漏映射；D/滴→啲 同遍内先于 啲→一点 被消费，且不产出 D/滴/展，故必收敛
    for _ in range(4):
        before = text
        for src, dst in _LEX_SORTED:
            text = text.replace(src, dst)
        if text == before:
            break
    for w in sorted(FILLERS, key=len, reverse=True):
        text = text.replace(w, "")
    text = re.sub(r"[，。！？、,.!?\s]+", "", text)

    def _num_sub(m: re.Match) -> str:
        val = cn2num(m.group(1))
        return (str(val) if val is not None else m.group(1)) + m.group(2)

    text = _NUM_UNIT.sub(_num_sub, text)
    return text
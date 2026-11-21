"""中文数字转换与口音模糊拼音工具。"""
from __future__ import annotations

import re

from pypinyin import lazy_pinyin

CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNITS = {"十": 10, "百": 100, "千": 1000}
CN_SHORT = {"廿": 20, "卅": 30}  # 粤语常用合文数字

_CN_NUM_PATTERN = re.compile(r"[零一二两三四五六七八九十百千廿卅\d]+")


def cn2num(text: str) -> int | None:
    """把中文数字串转为整数：二十六→26、廿五→25、两→2、十→10。"""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    for ch, val in CN_SHORT.items():
        if ch in text:
            rest = text.replace(ch, "")
            tail = cn2num(rest) if rest else 0
            if tail is None:
                return None
            return val + tail
    total, current = 0, 0
    seen = False
    for ch in text:
        if ch in CN_DIGITS:
            current = CN_DIGITS[ch]
            seen = True
        elif ch in CN_UNITS:
            unit = CN_UNITS[ch]
            if current == 0:
                current = 1  # 「十」=10、「百」=100
            total += current * unit
            current = 0
            seen = True
        else:
            return None
    if not seen:
        return None
    return total + current


def extract_numbers_with_unit(text: str, unit: str = "度") -> list[int]:
    """提取「X度」形式的数值，支持中文与阿拉伯数字。"""
    out = []
    for m in re.finditer(_CN_NUM_PATTERN.pattern + unit, text):
        val = cn2num(m.group(0)[: -len(unit)])
        if val is not None:
            out.append(val)
    return out


# 口音混淆等价类：把拼音音节映射到「模糊域」，吸收方言口音错字
_INITIAL_MAP = [("zh", "z"), ("ch", "c"), ("sh", "s"), ("n", "l"), ("f", "h")]
_FINAL_MAP = [("iang", "ian"), ("uang", "uan"), ("ang", "an"), ("eng", "en"), ("ing", "in")]


def fuzzy_syllable(py: str) -> str:
    for a, b in _INITIAL_MAP:
        if py.startswith(a):
            py = b + py[len(a):]
            break
    for a, b in _FINAL_MAP:
        if py.endswith(a):
            py = py[: -len(a)] + b
            break
    return py


def fuzzy_pinyin(text: str) -> str:
    """文本→空格分隔的模糊拼音串（仅保留汉字，非汉字原样保留）。"""
    parts = []
    for ch in text:
        if "一" <= ch <= "鿿":
            parts.append(fuzzy_syllable(lazy_pinyin(ch)[0]))
        elif not ch.isspace():
            parts.append(ch)
    return " ".join(parts)
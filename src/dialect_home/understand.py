"""封闭域指令理解：原文精确 + 模糊拼音双通道，房间/设备/动作/数值槽位。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from .normalize import normalize
from .textutils import extract_numbers_with_unit, fuzzy_pinyin

# 封闭域控制语素锚：闲聊/购买/损坏/疑问句不含任何控制语素，直接拒识，
# 防止模糊拼音把「看/买/坏」等非控制动词误配到 开/关 单字核心。
CONTROL_MORPHEMES = "开关熄调较拉换转设降升暗快慢光冻热大细亮返着上小冷下"

# 否定/抱怨/疑问/条件/意愿/完成体陈述类言语行为守卫：按语法标记类拒识，
# 而非逐句黑名单。否定副词 咪/米/别 带词边界（避免误伤 妈咪/玉米/特别）。
NON_CONTROL_GUARD = re.compile(
    r"唔够|唔冷|唔冻|唔转|唔着|唔开|唔好|唔想|唔使|唔需要|不想|不要|"
    r"太[光暗热冻冷大细快慢声]|好(?=[光暗冻热冷大细快慢声])|"
    r"(?<![妈阿])咪(?=[" + CONTROL_MORPHEMES + r"])|"
    r"(?<![妈厘玉])米(?=[" + CONTROL_MORPHEMES + r"])|"
    r"(?<!特)别(?=[" + CONTROL_MORPHEMES + r"])|"
    r"坏|卡住|漏水|系咪|几钱|点算|点解|为什么|为何|睇|查|好看|"
    r"边个|谁|咩|吗|了未|[" + CONTROL_MORPHEMES + r"]未|如果|假如|一旦|就(?=[" + CONTROL_MORPHEMES + r"])|"
    r"(我|你|佢|他|她)[^了]{0,3}了")


@dataclass
class Command:
    device: str
    action: str
    room: str | None
    room_name: str
    slots: dict = field(default_factory=dict)
    confidence: float = 0.0
    matched_text: str = ""

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "action": self.action,
            "room": self.room,
            "room_name": self.room_name,
            "slots": self.slots,
            "confidence": round(self.confidence, 3),
        }


class CommandLibrary:
    def __init__(self, path: str | Path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.rooms: dict = {}
        for key, spec in raw["rooms"].items():
            aliases = sorted({normalize(a) for a in spec["aliases"]} | {normalize(spec["name"])},
                             key=len, reverse=True)
            self.rooms[key] = {"name": spec["name"], "aliases": aliases}
        self.devices: dict = {}
        for key, spec in raw["devices"].items():
            aliases = sorted({normalize(a) for a in spec["aliases"]} | {normalize(spec["name"])},
                             key=len, reverse=True)
            actions = {}
            for act, aspec in spec["actions"].items():
                a_aliases = sorted({normalize(a) for a in aspec["aliases"]}, key=len, reverse=True)
                actions[act] = {
                    "aliases": a_aliases,
                    "feedback": aspec.get("feedback", "已执行"),
                    "slot": aspec.get("slot"),
                }
            self.devices[key] = {"name": spec["name"], "aliases": aliases, "actions": actions}


def _best_alias_match(text: str, aliases: list[str], threshold: float) -> tuple[str | None, float]:
    """返回 (命中别名, 分数)。原文子串=100；否则模糊拼音/字面 partial ratio。"""
    text_py = fuzzy_pinyin(text)
    best, best_score = None, 0.0
    for alias in aliases:
        if alias and alias in text:
            return alias, 100.0
        score = max(
            fuzz.partial_ratio(fuzzy_pinyin(alias), text_py),
            fuzz.partial_ratio(alias, text),
        )
        if score > best_score:
            best, best_score = alias, score
    if best is not None and best_score >= threshold:
        return best, float(best_score)
    return None, 0.0


class Understander:
    def __init__(self, library: CommandLibrary, fuzzy_threshold: float = 78,
                 entity_threshold: float = 85, default_room: str = "living_room"):
        self.lib = library
        self.fuzzy_threshold = fuzzy_threshold
        self.entity_threshold = entity_threshold
        self.default_room = default_room

    def parse(self, text: str) -> Command | None:
        text = normalize(text)
        if not text:
            return None
        # 锚检查在剔除设备/房间别名后的骨架上进行：否则「空调」自带的「调」会绕过锚
        anchor_text = text
        for spec in self.lib.devices.values():
            for alias in spec["aliases"]:
                anchor_text = anchor_text.replace(alias, "")
        for spec in self.lib.rooms.values():
            for alias in spec["aliases"]:
                anchor_text = anchor_text.replace(alias, "")
        if not any(ch in anchor_text for ch in CONTROL_MORPHEMES):
            return None
        if NON_CONTROL_GUARD.search(text):
            return None

        room_key, room_alias, room_score = self._find_room(text)
        dev_key, _, dev_score = self._find_device(text)
        inf_key, inf_score = self._infer_device_from_action(text)
        # 精确证据优先：动作别名精确命中反推的设备，胜过设备名的模糊拼音命中
        if inf_key is not None and dev_score < 100:
            dev_key, dev_score = inf_key, inf_score
        if dev_key is None:
            return None
        device = self.lib.devices[dev_key]

        skeleton = text
        for alias in device["aliases"]:
            skeleton = skeleton.replace(alias, "")
        if room_alias:
            skeleton = skeleton.replace(room_alias, "")

        action, score, slots = self._find_action(dev_key, device, text, skeleton)
        if action is None:
            return None
        if action in ("temp_up", "temp_down") and "delta" not in slots:
            nums = extract_numbers_with_unit(text, "度")
            slots["delta"] = nums[0] if nums else 1

        if room_key is None:
            room_key = self.default_room
        room_name = self.lib.rooms[room_key]["name"]
        return Command(
            device=dev_key, action=action, room=room_key, room_name=room_name,
            slots=slots, confidence=min(score, dev_score) / 100.0, matched_text=text,
        )

    def _find_room(self, text: str):
        best = (None, None, 0.0)
        for key, spec in self.lib.rooms.items():
            alias, score = _best_alias_match(text, spec["aliases"], self.entity_threshold)
            if alias and score > best[2]:
                best = (key, alias, score)
        return best

    def _find_device(self, text: str):
        best = (None, None, 0.0)
        for key, spec in self.lib.devices.items():
            alias, score = _best_alias_match(text, spec["aliases"], self.entity_threshold)
            if alias and score > best[2]:
                best = (key, alias, score)
        return best

    def _infer_device_from_action(self, text: str):
        """「转台」「大声啲」等未提设备名的说法：用动作别名反推设备。"""
        best_key, best_len = None, 0
        for key, spec in self.lib.devices.items():
            for act_spec in spec["actions"].values():
                for alias in act_spec["aliases"]:
                    if alias in text and len(alias) > best_len:
                        best_key, best_len = key, len(alias)
        return (best_key, 90.0) if best_key else (None, 0.0)

    def _find_action(self, dev_key: str, device: dict, text: str, skeleton: str):
        # 数值槽位规则通道（空调温度）：归一化后「较→调」「两度→2度」
        if dev_key == "ac":
            nums = extract_numbers_with_unit(skeleton, "度") or extract_numbers_with_unit(text, "度")
            if re.search(r"调到|设定|降到|降至", skeleton):
                if nums:
                    return "set_temp", 100.0, {"temperature": nums[0]}
                return None, 0.0, {}
            if re.search(r"调低|降低", skeleton):
                return "temp_down", 100.0, {"delta": nums[0] if nums else 1}
            if re.search(r"调高|升高", skeleton):
                return "temp_up", 100.0, {"delta": nums[0] if nums else 1}
            # 「调一点/调D」无方向词：升降歧义，拒识而非武断选方向
            if "调一点" in skeleton:
                return None, 0.0, {}

        skeleton_py = fuzzy_pinyin(skeleton)
        # 精确子串命中优先于模糊命中（模糊拼音存在 光≈关 等鼻音合并同分碰撞）
        best_exact = (None, 0)
        best_fuzzy = (None, 0.0)
        for act, spec in device["actions"].items():
            for alias in spec["aliases"]:
                if alias in text:
                    if len(alias) > best_exact[1]:
                        best_exact = (act, len(alias))
                    continue
                core = alias
                for d_alias in device["aliases"]:
                    core = core.replace(d_alias, "")
                # 单字核心（开/关/熄）的拼音模糊仅在字面出现于骨架时启用：
                # 鼻音合并使 光≈关，否则「灯太光」会凭 core 拼音满分误判 turn_off
                core_py = (fuzz.partial_ratio(fuzzy_pinyin(core), skeleton_py)
                           if core and (len(core) > 1 or core in skeleton) else 0)
                score = max(
                    fuzz.partial_ratio(fuzzy_pinyin(alias), fuzzy_pinyin(text)),
                    core_py,
                    fuzz.partial_ratio(core, skeleton) if core else 0,
                )
                if score > best_fuzzy[1]:
                    best_fuzzy = (act, score)
        if best_exact[0] is not None:
            action, score = best_exact[0], 100.0
        elif best_fuzzy[0] is not None and best_fuzzy[1] >= self.fuzzy_threshold:
            action, score = best_fuzzy[0], float(best_fuzzy[1])
        else:
            return None, 0.0, {}
        # 槽位校验对精确/模糊两通道统一生效：set_temp 缺温度不得产出空槽指令
        if device["actions"][action]["slot"] == "temperature":
            nums = extract_numbers_with_unit(text, "度")
            if not nums:
                return None, 0.0, {}
            return action, score, {"temperature": nums[0]}
        return action, score, {}
"""虚拟家居模拟器：无硬件即可完整演示与自动化测试。"""
from __future__ import annotations

import copy

from ..understand import Command
from .base import ExecResult, Executor

DEFAULT_STATE = {
    "light": {"on": False, "brightness": 60},
    "ac": {"on": False, "temp": 26, "mode": "cool"},
    "tv": {"on": False, "channel": 1, "volume": 20},
    "curtain": {"open": False},
    "fan": {"on": False, "speed": 1},
    "socket": {"on": False},
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class SimulatorExecutor(Executor):
    def __init__(self, rooms: list[str] | None = None):
        self.rooms = rooms or ["living_room", "bedroom", "kitchen", "bathroom", "study"]
        self.state = {r: copy.deepcopy(DEFAULT_STATE) for r in self.rooms}

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def reset(self) -> None:
        self.state = {r: copy.deepcopy(DEFAULT_STATE) for r in self.rooms}

    def execute(self, cmd: Command, action_spec: dict) -> ExecResult:
        room = self.state.setdefault(cmd.room, copy.deepcopy(DEFAULT_STATE))
        dev = room.setdefault(cmd.device, copy.deepcopy(DEFAULT_STATE.get(cmd.device, {})))
        act, slots = cmd.action, cmd.slots

        if act in ("turn_on", "open"):
            if cmd.device == "curtain":
                dev["open"] = True
            else:
                dev["on"] = True
        elif act == "turn_off":
            if cmd.device == "curtain":
                dev["open"] = False
            else:
                dev["on"] = False
        elif act == "close":
            dev["open"] = False
        elif act == "brighter":
            dev["on"] = True
            dev["brightness"] = _clamp(dev.get("brightness", 60) + 10, 10, 100)
        elif act == "dimmer":
            dev["brightness"] = _clamp(dev.get("brightness", 60) - 10, 10, 100)
        elif act == "temp_up":
            dev["on"] = True
            dev["temp"] = _clamp(dev.get("temp", 26) + slots.get("delta", 1), 16, 30)
        elif act == "temp_down":
            dev["on"] = True
            dev["temp"] = _clamp(dev.get("temp", 26) - slots.get("delta", 1), 16, 30)
        elif act == "set_temp":
            if "temperature" not in slots:
                return ExecResult(False, "请说明要设定的温度，例如「空调调到26度」")
            dev["on"] = True
            dev["temp"] = _clamp(slots["temperature"], 16, 30)
        elif act == "channel_up":
            dev["on"] = True
            dev["channel"] = dev.get("channel", 1) % 99 + 1
        elif act == "channel_down":
            dev["on"] = True
            dev["channel"] = (dev.get("channel", 1) - 2) % 99 + 1
        elif act == "volume_up":
            dev["volume"] = _clamp(dev.get("volume", 20) + 5, 0, 100)
        elif act == "volume_down":
            dev["volume"] = _clamp(dev.get("volume", 20) - 5, 0, 100)
        elif act == "speed_up":
            dev["on"] = True
            dev["speed"] = _clamp(dev.get("speed", 1) + 1, 1, 3)
        elif act == "speed_down":
            dev["speed"] = _clamp(dev.get("speed", 1) - 1, 1, 3)
        else:
            return ExecResult(False, f"模拟器不支持动作: {act}")

        msg = action_spec.get("feedback", "已执行").format(
            room=cmd.room_name, delta=slots.get("delta", 1), **dev)
        return ExecResult(True, msg, self.snapshot())
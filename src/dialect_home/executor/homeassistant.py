"""Home Assistant 执行适配：REST 服务调用。"""
from __future__ import annotations

import requests

from ..understand import Command
from .base import ExecResult, Executor

# (device, action) → (domain, service, 额外数据)
_SERVICE_MAP = {
    ("light", "turn_on"): ("light", "turn_on", {}),
    ("light", "turn_off"): ("light", "turn_off", {}),
    ("light", "brighter"): ("light", "turn_on", {"brightness_pct": 90}),
    ("light", "dimmer"): ("light", "turn_on", {"brightness_pct": 30}),
    ("ac", "turn_on"): ("climate", "turn_on", {}),
    ("ac", "turn_off"): ("climate", "turn_off", {}),
    ("ac", "set_temp"): ("climate", "set_temperature", {}),
    ("tv", "turn_on"): ("media_player", "turn_on", {}),
    ("tv", "turn_off"): ("media_player", "turn_off", {}),
    ("tv", "volume_up"): ("media_player", "volume_up", {}),
    ("tv", "volume_down"): ("media_player", "volume_down", {}),
    ("curtain", "open"): ("cover", "open_cover", {}),
    ("curtain", "close"): ("cover", "close_cover", {}),
    ("fan", "turn_on"): ("fan", "turn_on", {}),
    ("fan", "turn_off"): ("fan", "turn_off", {}),
    ("socket", "turn_on"): ("switch", "turn_on", {}),
    ("socket", "turn_off"): ("switch", "turn_off", {}),
}


class HomeAssistantExecutor(Executor):
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"}

    def execute(self, cmd: Command, action_spec: dict) -> ExecResult:
        mapping = _SERVICE_MAP.get((cmd.device, cmd.action))
        if mapping is None:
            return ExecResult(False, f"Home Assistant 未映射: {cmd.device}.{cmd.action}")
        domain, service, extra = mapping
        entity_id = f"{domain}.{cmd.room}_{cmd.device}"
        data = {"entity_id": entity_id, **extra}
        if cmd.action == "set_temp":
            data["temperature"] = cmd.slots.get("temperature", 26)
        try:
            resp = requests.post(f"{self.url}/api/services/{domain}/{service}",
                                 json=data, headers=self.headers, timeout=5)
            ok = resp.status_code == 200
            return ExecResult(ok, f"已调用 HA 服务 {domain}.{service} → {entity_id}"
                              if ok else f"HA 返回 {resp.status_code}")
        except requests.RequestException as exc:
            return ExecResult(False, f"Home Assistant 连接失败: {exc}")
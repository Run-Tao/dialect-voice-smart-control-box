"""云端适配层：直接复用 PC 版方言智能家居引擎（SenseVoice ASR + 归一化 + 规则 NLU + 执行器）。

不重建任何识别模型：dialect_home.pipeline.Pipeline 以可编辑安装或 sys.path 方式引入，
本模块只负责把引擎输出映射为端云契约 JSON，并生成粤语 reply_text。
云端不做本机 TTS 播报（Tts(enabled=False)），语音反馈由终端侧自行决定。
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ACTION_MAP = {"turn_on": "on", "turn_off": "off"}

DIALECT_MAP = {"yue": "cantonese", "zh": "mandarin", "en": "english", "ja": "japanese"}

REPLY_FALLBACK = "唔好意思，我听唔清要控制边个设备，唔该再讲一次"


class CloudEngine:
    def __init__(self, engine_root: str | Path):
        self.engine_root = Path(engine_root).resolve()
        self._lock = threading.Lock()
        self._pipeline = None

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def _ensure(self):
        with self._lock:
            if self._pipeline is not None:
                return
            src = str(self.engine_root / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            from dialect_home.pipeline import Pipeline, load_settings
            from dialect_home.tts import Tts

            settings = load_settings(self.engine_root / "config" / "settings.yaml")
            settings.setdefault("tts", {})["enabled"] = False
            self._pipeline = Pipeline(settings=settings, tts=Tts(enabled=False))

    def _reply(self, cmd: dict) -> str:
        room = cmd["room_name"]
        device = cmd["device"]
        dev = "冷气" if device == "ac" else self._pipeline.lib.devices[device]["name"]
        slots = cmd.get("slots") or {}
        delta = slots.get("delta", 1)
        temp = slots.get("temperature", 26)
        action = cmd["action"]
        templates = {
            "turn_on": "好嘅，已经帮你开咗{room}{dev}",
            "turn_off": "好嘅，已经帮你熄咗{room}{dev}",
            "brighter": "{room}{dev}已经较光啲",
            "dimmer": "{room}{dev}已经较暗啲",
            "temp_up": "{room}冷气已经调高{delta}度",
            "temp_down": "{room}冷气已经调低{delta}度",
            "set_temp": "{room}冷气已经设定到{temp}度",
            "open": "好嘅，已经帮你打开{room}{dev}",
            "close": "好嘅，已经帮你关咗{room}{dev}",
            "speed_up": "{room}{dev}风速已经较快啲",
            "speed_down": "{room}{dev}风速已经较慢啲",
            "channel_up": "{room}{dev}已经转咗台",
            "channel_down": "{room}{dev}已经返上一个台",
            "volume_up": "{room}{dev}已经大声啲",
            "volume_down": "{room}{dev}已经细声啲",
        }
        tpl = templates.get(action)
        if tpl is None:
            return "好嘅，已经帮你执行咗"
        return tpl.format(room=room, dev=dev, delta=delta, temp=temp)

    def understand(self, wav_path: str | Path) -> dict:
        """WAV → 契约 dict（sample_id 由调用方回填）。"""
        self._ensure()
        res = self._pipeline.process_wav(wav_path)
        cmd = res.command
        if cmd:
            slots = {
                "location": cmd["room"],
                "device": cmd["device"],
                "action": ACTION_MAP.get(cmd["action"], cmd["action"]),
            }
            slots.update(cmd.get("slots") or {})
            intent = "device_control"
            confidence = float(cmd["confidence"])
            reply = self._reply(cmd)
        else:
            slots = {}
            intent = "unknown"
            confidence = 0.0
            reply = REPLY_FALLBACK
        return {
            "sample_id": "",
            "dialect": DIALECT_MAP.get(res.language, res.language or "unknown"),
            "raw_text": res.raw_text,
            "normalized_text": res.normalized_text,
            "intent": intent,
            "slots": slots,
            "confidence": round(confidence, 3),
            "reply_text": reply,
        }
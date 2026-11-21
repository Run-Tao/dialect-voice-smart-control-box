"""MQTT 执行适配：把结构化命令发布到智能家居总线。"""
from __future__ import annotations

import json

from ..understand import Command
from .base import ExecResult, Executor


class MqttExecutor(Executor):
    def __init__(self, host: str = "127.0.0.1", port: int = 1883,
                 topic_prefix: str = "dialecthome/cmd"):
        import paho.mqtt.client as mqtt

        self.topic_prefix = topic_prefix
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()

    def execute(self, cmd: Command, action_spec: dict) -> ExecResult:
        topic = f"{self.topic_prefix}/{cmd.room}/{cmd.device}"
        payload = json.dumps({"action": cmd.action, **cmd.slots}, ensure_ascii=False)
        info = self.client.publish(topic, payload, qos=1)
        info.wait_for_publish(timeout=5)
        return ExecResult(info.is_published(), f"已发送 MQTT 指令 → {topic}: {payload}")
from .base import ExecResult, Executor
from .simulator import SimulatorExecutor


def build_executor(cfg: dict) -> Executor:
    kind = cfg.get("type", "simulator")
    if kind == "simulator":
        return SimulatorExecutor()
    if kind == "mqtt":
        from .mqtt import MqttExecutor
        return MqttExecutor(**cfg.get("mqtt", {}))
    if kind == "homeassistant":
        from .homeassistant import HomeAssistantExecutor
        return HomeAssistantExecutor(**cfg.get("homeassistant", {}))
    raise ValueError(f"未知执行器类型: {kind}")


__all__ = ["ExecResult", "Executor", "SimulatorExecutor", "build_executor"]
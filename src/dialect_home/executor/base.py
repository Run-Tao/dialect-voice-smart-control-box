from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..understand import Command


@dataclass
class ExecResult:
    ok: bool
    message: str
    state: dict | None = None


class Executor(ABC):
    @abstractmethod
    def execute(self, cmd: Command, action_spec: dict) -> ExecResult:
        ...

    def snapshot(self) -> dict | None:
        return None
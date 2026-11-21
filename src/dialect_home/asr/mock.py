"""Mock 后端：读取 wav 旁车 .txt 转写，供零模型依赖的单元测试。"""
from __future__ import annotations

import time
from pathlib import Path

from ..audio import AudioClip
from .base import AsrBackend, AsrResult


class MockBackend(AsrBackend):
    name = "mock"

    def __init__(self):
        self.last_path: Path | None = None

    def transcribe(self, clip: AudioClip, language: str = "auto") -> AsrResult:
        t0 = time.perf_counter()
        text = ""
        if self.last_path is not None:
            sidecar = self.last_path.with_suffix(".txt")
            if sidecar.exists():
                text = sidecar.read_text(encoding="utf-8").strip()
        ms = (time.perf_counter() - t0) * 1000
        return AsrResult(text=text, language=language, backend=self.name, ms=ms)
"""faster-whisper 备用后端：CPU int8，用于交叉验证与降级。"""
from __future__ import annotations

import time
from pathlib import Path

from ..audio import AudioClip
from .base import AsrBackend, AsrResult


class WhisperBackend(AsrBackend):
    name = "whisper"

    def __init__(self, model_size: str = "small", models_dir: str | Path = "models",
                 num_threads: int = 0):
        import os

        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model_size, device="cpu", compute_type="int8",
            cpu_threads=num_threads if num_threads > 0 else (os.cpu_count() or 4),
            download_root=str(Path(models_dir) / "whisper"),
        )

    def transcribe(self, clip: AudioClip, language: str = "auto") -> AsrResult:
        t0 = time.perf_counter()
        lang = None if language == "auto" else language
        segments, info = self.model.transcribe(clip.samples, language=lang, vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()
        ms = (time.perf_counter() - t0) * 1000
        return AsrResult(text=text, language=info.language, backend=self.name, ms=ms)
"""SenseVoice-Small 后端：原生粤语，非自回归，CPU 实时。"""
from __future__ import annotations

import os
import time
from pathlib import Path

from ..audio import AudioClip
from .base import AsrBackend, AsrResult

_MODEL_ID = "iic/SenseVoiceSmall"
_VAD_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


class SenseVoiceBackend(AsrBackend):
    name = "sensevoice"

    def __init__(self, models_dir: str | Path = "models", device: str = "cpu",
                 num_threads: int = 0):
        models_dir = Path(models_dir).resolve()
        models_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MODELSCOPE_CACHE", str(models_dir))
        import torch
        # 0 = 自适应逻辑核数（本机 16，树莓派 4B 为 4）
        torch.set_num_threads(num_threads if num_threads > 0 else (os.cpu_count() or 4))
        from funasr import AutoModel

        self.model = AutoModel(
            model=_MODEL_ID,
            vad_model=_VAD_ID,
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            cache_dir=str(models_dir),
            disable_update=True,
        )

    def transcribe(self, clip: AudioClip, language: str = "auto") -> AsrResult:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        t0 = time.perf_counter()
        res = self.model.generate(
            input=clip.samples,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
        )
        ms = (time.perf_counter() - t0) * 1000
        raw = res[0]["text"] if res else ""
        text = rich_transcription_postprocess(raw)
        lang = language
        if "<|yue|>" in raw:
            lang = "yue"
        elif "<|zh|>" in raw:
            lang = "zh"
        return AsrResult(text=text, language=lang, backend=self.name, ms=ms)
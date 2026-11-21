from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..audio import AudioClip


@dataclass
class AsrResult:
    text: str
    language: str
    backend: str
    ms: float


class AsrBackend(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, clip: AudioClip, language: str = "auto") -> AsrResult:
        ...
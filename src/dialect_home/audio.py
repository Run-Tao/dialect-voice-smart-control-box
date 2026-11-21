from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


@dataclass
class AudioClip:
    samples: np.ndarray  # float32, mono, [-1, 1]
    sr: int = TARGET_SR

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr


def to_mono_16k(samples: np.ndarray, sr: int) -> AudioClip:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != TARGET_SR:
        from math import gcd

        g = gcd(int(sr), TARGET_SR)
        samples = resample_poly(samples, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return AudioClip(samples=samples, sr=TARGET_SR)


def load_wav(path: str | Path) -> AudioClip:
    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_16k(samples, sr)


def save_wav(path: str | Path, clip: AudioClip) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), clip.samples, clip.sr, subtype="PCM_16")


def record(seconds: float, sr: int = TARGET_SR) -> AudioClip:
    import sounddevice as sd

    frames = int(seconds * sr)
    data = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return to_mono_16k(data[:, 0], sr)
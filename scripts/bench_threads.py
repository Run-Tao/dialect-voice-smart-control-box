"""线程数基准：同一进程内对比 4/8/16 推理线程的 ASR 延迟（中位数）。"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dialect_home.asr.sensevoice import SenseVoiceBackend  # noqa: E402
from dialect_home.audio import load_wav  # noqa: E402

CLIPS = ["yue01", "yue07", "yue13", "zh03", "zh05", "zh11"]


def main() -> None:
    backend = SenseVoiceBackend(num_threads=16)
    clips = [load_wav(ROOT / "data" / "corpus" / f"{c}.wav") for c in CLIPS]
    results = {}
    for threads in [4, 8, 16]:
        torch.set_num_threads(threads)
        backend.transcribe(clips[0])  # warmup
        times = [backend.transcribe(c).ms for _ in range(2) for c in clips]
        results[threads] = {
            "median_ms": round(statistics.median(times), 1),
            "mean_ms": round(statistics.mean(times), 1),
            "min_ms": round(min(times), 1),
            "max_ms": round(max(times), 1),
        }
        print(f"threads={threads}: {results[threads]}")
    out = ROOT / "data" / "results" / "thread_benchmark.json"
    out.write_text(__import__("json").dumps(results, indent=2), encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
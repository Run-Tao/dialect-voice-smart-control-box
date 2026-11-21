from .base import AsrBackend, AsrResult


def build_asr(cfg: dict, models_dir: str = "models") -> AsrBackend:
    backend = cfg.get("backend", "sensevoice")
    threads = cfg.get("num_threads", 0)
    if backend == "sensevoice":
        from .sensevoice import SenseVoiceBackend
        return SenseVoiceBackend(models_dir=models_dir, num_threads=threads)
    if backend == "whisper":
        from .whisper_backend import WhisperBackend
        return WhisperBackend(model_size=cfg.get("whisper_model", "small"),
                              models_dir=models_dir, num_threads=threads)
    if backend == "mock":
        from .mock import MockBackend
        return MockBackend()
    raise ValueError(f"未知 ASR 后端: {backend}")


__all__ = ["AsrBackend", "AsrResult", "build_asr"]
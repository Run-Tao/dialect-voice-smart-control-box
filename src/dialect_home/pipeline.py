"""流水线编排：音频 → ASR → 归一化 → 理解 → 执行 → 反馈 + 指标。"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .asr import AsrBackend, build_asr
from .audio import AudioClip, load_wav
from .executor import Executor, build_executor
from .normalize import normalize
from .tts import Tts
from .understand import Command, CommandLibrary, Understander

ROOT = Path(__file__).resolve().parents[2]


def load_settings(path: str | Path | None = None) -> dict:
    import yaml

    path = Path(path) if path else ROOT / "config" / "settings.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class PipelineResult:
    raw_text: str = ""
    language: str = ""
    backend: str = ""
    normalized_text: str = ""
    command: dict | None = None
    ok: bool = False
    feedback: str = ""
    timings_ms: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    def __init__(self, settings: dict | None = None, asr: AsrBackend | None = None,
                 executor: Executor | None = None, tts: Tts | None = None):
        self.settings = settings or load_settings()
        models_dir = ROOT / self.settings["asr"].get("models_dir", "models")
        self.asr = asr or build_asr(self.settings["asr"], models_dir=str(models_dir))
        self.executor = executor or build_executor(self.settings["executor"])
        ucfg = self.settings["understand"]
        self.lib = CommandLibrary(ROOT / "config" / "commands.yaml")
        self.understander = Understander(
            self.lib,
            fuzzy_threshold=ucfg.get("fuzzy_threshold", 78),
            entity_threshold=ucfg.get("entity_threshold", 85),
            default_room=ucfg.get("default_room", "living_room"),
        )
        tcfg = self.settings.get("tts", {})
        self.tts = tts or Tts(enabled=tcfg.get("enabled", True),
                              engine=tcfg.get("engine", "pyttsx3"),
                              edge_voice=tcfg.get("edge_voice", "zh-CN-XiaoxiaoNeural"))
        self.log_path = ROOT / self.settings.get("logging", {}).get(
            "interactions_file", "data/results/interactions.jsonl")

    def process_audio(self, clip: AudioClip, language: str | None = None) -> PipelineResult:
        result = PipelineResult()
        lang = language or self.settings["asr"].get("language", "auto")
        asr_res = self.asr.transcribe(clip, language=lang)
        result.raw_text = asr_res.text
        result.language = asr_res.language
        result.backend = asr_res.backend
        result.timings_ms["asr"] = round(asr_res.ms, 1)
        self._finish(result)
        return result

    def process_wav(self, path: str | Path, language: str | None = None) -> PipelineResult:
        path = Path(path)
        if self.asr.name == "mock":
            self.asr.last_path = path
        t0 = time.perf_counter()
        clip = load_wav(path)
        result = self.process_audio(clip, language=language)
        result.timings_ms["load"] = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def process_text(self, text: str) -> PipelineResult:
        result = PipelineResult(raw_text=text, backend="text-input")
        self._finish(result)
        return result

    def _finish(self, result: PipelineResult) -> None:
        t0 = time.perf_counter()
        result.normalized_text = normalize(result.raw_text)
        cmd: Command | None = self.understander.parse(result.raw_text)
        result.timings_ms["nlu"] = round((time.perf_counter() - t0) * 1000, 1)
        if cmd is None:
            result.error = "未识别到可执行的家居指令"
            result.feedback = "不好意思，我没听清要控制哪个设备，请再说一次"
            self.tts.speak(result.feedback)
            self._log(result)
            return
        result.command = cmd.to_dict()
        action_spec = self.lib.devices[cmd.device]["actions"][cmd.action]
        t0 = time.perf_counter()
        exec_res = self.executor.execute(cmd, action_spec)
        result.timings_ms["exec"] = round((time.perf_counter() - t0) * 1000, 1)
        result.ok = exec_res.ok
        result.feedback = exec_res.message
        result.timings_ms["total"] = round(sum(result.timings_ms.values()), 1)
        self.tts.speak(result.feedback)
        self._log(result)

    def _log(self, result: PipelineResult) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **result.to_dict()}
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
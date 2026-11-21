"""零模型依赖的流水线测试：mock ASR 读 wav 旁车 .txt。"""
import numpy as np

from dialect_home.asr.mock import MockBackend
from dialect_home.audio import AudioClip, save_wav
from dialect_home.executor.simulator import SimulatorExecutor
from dialect_home.pipeline import Pipeline, load_settings
from dialect_home.tts import Tts


def _settings(tmp_path):
    s = load_settings()
    s["asr"]["backend"] = "mock"
    s["tts"]["enabled"] = False
    s["logging"]["interactions_file"] = str(tmp_path / "interactions.jsonl")
    return s


def test_pipeline_cantonese_wav(tmp_path):
    wav = tmp_path / "cmd.wav"
    save_wav(wav, AudioClip(samples=np.zeros(16000, dtype=np.float32)))
    wav.with_suffix(".txt").write_text("唔该帮我熄咗客厅盏灯", encoding="utf-8")

    pipe = Pipeline(_settings(tmp_path), asr=MockBackend(),
                    executor=SimulatorExecutor(), tts=Tts(enabled=False))
    pipe.executor.state["living_room"]["light"]["on"] = True
    res = pipe.process_wav(wav)
    assert res.raw_text == "唔该帮我熄咗客厅盏灯"
    assert res.command["device"] == "light"
    assert res.command["action"] == "turn_off"
    assert res.ok
    assert pipe.executor.state["living_room"]["light"]["on"] is False
    assert (tmp_path / "interactions.jsonl").exists()


def test_pipeline_text_debug(tmp_path):
    pipe = Pipeline(_settings(tmp_path), asr=MockBackend(),
                    executor=SimulatorExecutor(), tts=Tts(enabled=False))
    res = pipe.process_text("冷气较低两度")
    assert res.ok
    assert res.command["slots"]["delta"] == 2
    assert pipe.executor.state["living_room"]["ac"]["temp"] == 24


def test_pipeline_unknown(tmp_path):
    pipe = Pipeline(_settings(tmp_path), asr=MockBackend(),
                    executor=SimulatorExecutor(), tts=Tts(enabled=False))
    res = pipe.process_text("今日天气真系好好")
    assert not res.ok
    assert res.command is None
    assert res.error
# -*- coding: utf-8 -*-
"""命令表、执行器与实时管线测试。"""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.commands import COMMANDS, INITIAL_STATE, spoken_text, zh_text
from src.control import LoggingExecutor, MockEsp32Executor
from src.realtime import DEFAULT_MODEL, DEFAULT_THRESHOLDS, DialectEngine
from src.vad import EnergyVad

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'
THRESHOLDS = ROOT / 'runtime' / 'models' / 'v3_thresholds_vad.json'

pytestmark = pytest.mark.skipif(not MODEL.exists(),
                                reason='runtime 模型未同步，先运行 scripts/sync_runtime.py')


def test_deployed_defaults_exist():
    """部署默认路径必须指向真实存在的工件，否则线上会静默加载失败。"""
    assert DEFAULT_MODEL.exists(), DEFAULT_MODEL
    assert DEFAULT_THRESHOLDS.exists(), DEFAULT_THRESHOLDS


def test_command_table_matches_model_classes():
    engine = DialectEngine.load(str(MODEL), str(THRESHOLDS))
    assert set(engine.classes_) == set(COMMANDS)


def test_command_table_fields():
    for name, entry in COMMANDS.items():
        assert entry['zh'], name
        assert entry['actuator'] in {'light', 'curtain', 'all', 'none'}, name
        assert entry['action'] in {'on', 'off', 'open', 'close', 'query'}, name


def test_spoken_text_with_wake_word():
    assert spoken_text('LIGHT_ON') == '打开灯'
    assert spoken_text('LIGHT_ON', '小爱同学') == '小爱同学打开灯'
    assert zh_text('NOT_A_COMMAND') == 'NOT_A_COMMAND'


def test_logging_executor_updates_state():
    ex = LoggingExecutor()
    ex.execute('LIGHT_ON')
    assert ex.state['light'] == 'on'
    ex.execute('ALL_OFF')
    assert ex.state == dict(INITIAL_STATE)


def test_mock_esp32_executor_roundtrip():
    ex = MockEsp32Executor()
    ex.execute('CURTAIN_OPEN')
    assert ex.controller.state['curtain'] == 'open'
    ex.execute('STATUS_ALL')
    assert 'CURTAIN=open' in ex.controller.handle_line('CMD,999,STATUS_ALL')


def test_engine_rejects_low_confidence():
    engine = DialectEngine.load(str(MODEL), str(THRESHOLDS), threshold=1.01)
    rng = np.random.default_rng(0)
    audio = (0.05 * rng.standard_normal(16000)).astype(np.float32)
    result = engine.process_utterance(audio, 16000)
    assert not result.accepted
    assert result.reason.startswith('below_threshold')


def test_engine_reports_empty_audio():
    engine = DialectEngine.load(str(MODEL), str(THRESHOLDS))
    result = engine.process_utterance(np.zeros(0, dtype=np.float32), 16000)
    assert result.reason == 'empty_audio'
    assert not result.accepted


def test_engine_rejects_wrong_sample_rate():
    engine = DialectEngine.load(str(MODEL), str(THRESHOLDS))
    result = engine.process_utterance(np.zeros(8000, dtype=np.float32), 8000)
    assert 'sample_rate_mismatch' in result.reason


def test_engine_picks_a_class_and_confidence():
    engine = DialectEngine.load(str(MODEL), str(THRESHOLDS))
    t = np.arange(16000) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    result = engine.process_utterance(audio, 16000)
    assert result.command in COMMANDS
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms > 0


def test_vad_feeds_engine():
    engine = DialectEngine.load(str(MODEL), str(THRESHOLDS))
    vad = EnergyVad(fs=16000)
    t = np.arange(8000) / 16000
    tone = (0.25 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    audio = np.concatenate([np.zeros(8000), tone, np.zeros(8000)])
    utterances = vad.feed(audio)
    assert utterances
    result = engine.process_utterance(utterances[0], 16000)
    assert result.command in COMMANDS

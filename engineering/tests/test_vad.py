# -*- coding: utf-8 -*-
"""VAD 端点检测测试。"""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vad import EnergyVad, frame_energy_db, trim_silence

FS = 16000


def make_utterance(speech_sec=0.6, silence_sec=0.8, tone=220.0, amp=0.3):
    t = np.arange(int(FS * speech_sec)) / FS
    speech = amp * np.sin(2 * np.pi * tone * t)
    quiet = np.zeros(int(FS * silence_sec))
    return np.concatenate([quiet, speech, quiet]).astype(np.float32)


def test_frame_energy_db_silence_is_low():
    energy = frame_energy_db(np.zeros(FS), 512, 512)
    assert energy.size > 0
    assert energy.max() < -100.0


def test_vad_extracts_one_utterance():
    vad = EnergyVad(fs=FS, initial_noise_db=-55.0)
    utterances = vad.feed(make_utterance())
    assert len(utterances) == 1
    seg = utterances[0]
    assert 0.3 < seg.size / FS < 1.2
    # 起点应落在语音开始附近，而不是从文件开头就带静音
    leading = np.flatnonzero(np.abs(seg) > 0.01)
    assert leading.size > 0
    assert leading[0] / FS < 0.35


def test_vad_ignores_pure_silence():
    vad = EnergyVad(fs=FS, initial_noise_db=-55.0)
    assert vad.feed(np.zeros(FS, dtype=np.float32)) == []


def test_vad_splits_two_utterances():
    vad = EnergyVad(fs=FS, initial_noise_db=-55.0)
    signal = np.concatenate([make_utterance(), make_utterance()])
    utterances = vad.feed(signal)
    assert len(utterances) == 2


def test_vad_flush_returns_pending():
    vad = EnergyVad(fs=FS, initial_noise_db=-55.0)
    t = np.arange(int(FS * 0.5)) / FS
    speech = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    vad.feed(speech)
    assert vad.in_speech
    assert len(vad.flush()) == 1


def test_trim_silence_removes_edges():
    audio = make_utterance()
    trimmed, lead, trail = trim_silence(audio, fs=FS)
    assert trimmed.size < audio.size
    assert lead > 0.5
    assert trail > 0.5

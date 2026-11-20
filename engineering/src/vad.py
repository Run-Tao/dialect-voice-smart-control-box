# -*- coding: utf-8 -*-
"""基于帧能量的端点检测（VAD）。

建议书的技术路线里「方言语音 -> VAD -> MFCC -> ...」是第一步，但整个仓库此前
没有实现：识别入口用的是固定长度滑窗，既不会跳过静音，也会把半句话送进模型。
本模块补上这一环。

做法：帧能量转 dB，自适应噪声底（最小值统计，慢速更新），判决门限取
噪声底 + energy_margin_db。语音段结束后用静音挂起（hangover）判定结束，
并保留 pre-roll，避免吃掉爆破音起始。
"""
from __future__ import annotations

from collections import deque
from typing import List

import numpy as np


def frame_energy_db(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """把信号切成帧，返回每帧平均能量的 dB 值。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size))
    n_frames = 1 + (x.size - frame_len) // hop_len
    frames = np.stack([x[i * hop_len:i * hop_len + frame_len] for i in range(n_frames)])
    power = np.mean(frames * frames, axis=1)
    return 10.0 * np.log10(power + 1e-12)


class EnergyVad:
    """流式能量 VAD。`feed` 逐块送入音频，返回切出的完整语音段。"""

    def __init__(self, fs: int = 16000, frame_ms: float = 30.0,
                 energy_margin_db: float = 9.0, min_speech_ms: float = 200.0,
                 min_silence_ms: float = 400.0, pre_roll_ms: float = 300.0,
                 max_utterance_ms: float = 4000.0,
                 noise_update_alpha: float = 0.02,
                 initial_noise_db: float = -55.0):
        self.fs = int(fs)
        self.frame_len = max(1, int(self.fs * frame_ms / 1000.0))
        self.hop_len = self.frame_len
        self.energy_margin_db = float(energy_margin_db)
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.min_silence_frames = max(1, int(min_silence_ms / frame_ms))
        self.pre_roll_frames = max(0, int(pre_roll_ms / frame_ms))
        self.max_utterance_frames = max(1, int(max_utterance_ms / frame_ms))
        self.noise_update_alpha = float(noise_update_alpha)
        self.noise_db = float(initial_noise_db)
        self.initial_noise_db = float(initial_noise_db)

        self._buffer = np.zeros(0, dtype=np.float64)
        self._pre_roll = deque(maxlen=max(1, self.pre_roll_frames))
        self._speech_frames: List[np.ndarray] = []
        self._trailing_silent = 0
        self._silence_run = 0
        self._in_speech = False
        # 流位置记账：延迟基准要量「用户说完 -> 动作下发」的**音频时间**，
        # 不能靠「静音挂起多少毫秒」反推——那会把切段误差算到延迟头上。
        # _stream_pos 只增不减，reset() 不清零，这样它和上游的累计样本数同坐标系。
        self._stream_pos = 0
        self._pos_base = 0
        self.speech_start_sample = -1
        self.speech_end_sample = -1
        self.cut_sample = -1

    # ------------------------------------------------------------------
    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def noise_floor_db(self) -> float:
        return self.noise_db

    def reset_noise_floor(self) -> None:
        """把噪声底复位到初值。

        连续监听时噪声底应当自适应，不该复位；但**离线逐条回放**时每条
        语音是独立的流，复位才与「语料裁剪时每条用一个新 VAD」的条件一致
        （否则前一条的噪声底会带到下一条，切出的段与训练语料对不上）。
        """
        self.noise_db = self.initial_noise_db

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float64)
        self._pre_roll.clear()
        self._speech_frames.clear()
        self._trailing_silent = 0
        self._silence_run = 0
        self._in_speech = False
        # 缓冲清空了，但流的绝对位置不变：新缓冲的起点就是当前流位置。
        self._pos_base = self._stream_pos

    # ------------------------------------------------------------------
    def feed(self, audio: np.ndarray) -> List[np.ndarray]:
        """送入一块音频，返回本次切出的语音段列表。"""
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        self._stream_pos += x.size
        self._buffer = np.concatenate([self._buffer, x])
        if self._buffer.size < self.frame_len:
            return []

        n_frames = 1 + (self._buffer.size - self.frame_len) // self.hop_len
        take = n_frames * self.hop_len
        block = self._buffer[:take + self.frame_len - self.hop_len]
        self._buffer = self._buffer[take:]

        energies = frame_energy_db(block, self.frame_len, self.hop_len)
        utterances: List[np.ndarray] = []
        for i in range(n_frames):
            frame = block[i * self.hop_len:i * self.hop_len + self.frame_len]
            frame_pos = self._pos_base + i * self.hop_len
            utterance = self._process_frame(frame, float(energies[i]), frame_pos)
            if utterance is not None:
                utterances.append(utterance)
        self._pos_base += take
        return utterances

    def flush(self) -> List[np.ndarray]:
        """流结束时把正在进行的语音段吐出。"""
        if self._in_speech and self._speech_frames:
            if self._trailing_silent:
                self._speech_frames = self._speech_frames[:-self._trailing_silent]
            frames = list(self._pre_roll) + list(self._speech_frames)
            self._pre_roll.clear()
            self._speech_frames = []
            self._trailing_silent = 0
            self._silence_run = 0
            self._in_speech = False
            self.cut_sample = self._stream_pos
            if len(frames) >= self.min_speech_frames:
                return [np.concatenate(frames).astype(np.float32)]
        self.reset()
        return []

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, energy_db: float, frame_pos: int = -1):
        threshold = self.noise_db + self.energy_margin_db
        is_speech = energy_db > threshold

        if is_speech:
            # 只有判定为语音的帧才更新噪声底，避免把语音当成环境噪声。
            self._silence_run = 0
        else:
            self.noise_db = ((1.0 - self.noise_update_alpha) * self.noise_db
                             + self.noise_update_alpha * energy_db)

        if not self._in_speech:
            self._pre_roll.append(frame)
            if is_speech:
                self._in_speech = True
                self._speech_frames = [frame]
                self._silence_run = 0
                if frame_pos >= 0:
                    self.speech_start_sample = frame_pos
                    self.speech_end_sample = frame_pos + self.frame_len
                return None
            return None

        self._speech_frames.append(frame)

        if not is_speech:
            self._silence_run += 1
            self._trailing_silent += 1
            if self._silence_run >= self.min_silence_frames:
                if frame_pos >= 0:
                    self.cut_sample = frame_pos + self.frame_len
                return self._close_utterance()
            return None

        self._trailing_silent = 0
        if frame_pos >= 0:
            self.speech_end_sample = frame_pos + self.frame_len
        if len(self._speech_frames) >= self.max_utterance_frames:
            if frame_pos >= 0:
                self.cut_sample = frame_pos + self.frame_len
            return self._close_utterance()
        return None

    def _close_utterance(self):
        # 挂起期的静音帧只用于判定结束，不送进模型，否则尾部静音会拉低特征质量。
        if self._trailing_silent:
            self._speech_frames = self._speech_frames[:-self._trailing_silent]
        frames = list(self._pre_roll) + list(self._speech_frames)
        self._pre_roll.clear()
        self._speech_frames = []
        self._trailing_silent = 0
        self._silence_run = 0
        self._in_speech = False
        if len(frames) < self.min_speech_frames:
            return None
        return np.concatenate(frames).astype(np.float32)


def trim_silence(audio: np.ndarray, fs: int = 16000,
                 margin_db: float = 12.0, pad_ms: float = 50.0):
    """裁掉首尾静音，返回裁剪后的音频与首尾静音时长（秒）。

    用于离线批量评估：让待测音频与 VAD 实时输出尽量一致。
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0:
        return audio.astype(np.float32), 0.0, 0.0
    frame_len = max(1, int(fs * 0.02))
    energies = frame_energy_db(audio, frame_len, frame_len)
    if energies.size == 0:
        return audio.astype(np.float32), 0.0, 0.0
    peak = float(np.max(energies))
    threshold = peak - margin_db
    voiced = np.nonzero(energies > threshold)[0]
    if voiced.size == 0:
        return audio.astype(np.float32), 0.0, float(audio.size / fs)
    pad = int(fs * pad_ms / 1000.0)
    start = max(0, int(voiced[0] * frame_len) - pad)
    end = min(audio.size, int((voiced[-1] + 1) * frame_len) + pad)
    lead = start / fs
    trail = (audio.size - end) / fs
    return audio[start:end].astype(np.float32), lead, trail

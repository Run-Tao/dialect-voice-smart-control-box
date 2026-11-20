# -*- coding: utf-8 -*-
"""纯numpy MFCC特征提取。"""
import numpy as np


def pre_emphasis(x, coeff=0.97):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    return np.append(x[0], x[1:] - coeff * x[:-1])


def frame_signal(x, frame_len, hop_len):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("音频为空")
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // hop_len
    frames = np.stack([x[i * hop_len:i * hop_len + frame_len] for i in range(n_frames)])
    return frames


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, sample_rate, fmin=50.0, fmax=None):
    if fmax is None:
        fmax = sample_rate / 2.0
    if not (0 <= fmin < fmax <= sample_rate / 2.0):
        raise ValueError("fmin/fmax不合法")
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    bins = np.maximum.accumulate(bins)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        center = min(center, n_fft // 2)
        right = min(right, n_fft // 2)
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def dct_matrix(n_out, n_in):
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    mat = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    mat[0, :] *= 1.0 / np.sqrt(2.0)
    return mat * np.sqrt(2.0 / n_in)


def extract_mfcc(audio, sample_rate, n_fft=512, hop_length=160,
                 n_mfcc=13, n_mels=26, fmin=50.0, fmax=None,
                 preemphasis=0.97, use_cmvn=True):
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate必须为正数")
    if n_mfcc > n_mels:
        raise ValueError("n_mfcc不能大于n_mels")
    x = pre_emphasis(audio, preemphasis)
    frames = frame_signal(x, n_fft, hop_length)
    window = np.hamming(n_fft)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    fb = mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)
    mel = np.log(np.maximum(np.dot(spec, fb.T), 1e-10))
    mfcc = np.dot(mel, dct_matrix(n_mfcc, n_mels).T)
    if use_cmvn:
        mfcc = (mfcc - np.mean(mfcc, axis=0, keepdims=True)) / (
            np.std(mfcc, axis=0, keepdims=True) + 1e-8
        )
    validate_feature_matrix(mfcc)
    return mfcc.astype(np.float32)


def validate_feature_matrix(x, name='features'):
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name}必须是二维矩阵，当前shape={arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name}不能有空维度，当前shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}包含NaN或Inf")
    return True

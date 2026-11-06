# -*- coding: utf-8 -*-
"""纯 numpy 的 MFCC/log-Mel 特征，便于先跑通。"""
import numpy as np


def pre_emphasis(x, coeff=0.97):
    x = np.asarray(x, dtype=np.float64)
    return np.append(x[0], x[1:] - coeff * x[:-1])


def frame_signal(x, frame_len=512, hop_len=160):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    frames = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        frames.append(x[start:start + frame_len])
    return np.asarray(frames)


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, fs, fmin=50, fmax=None):
    if fmax is None:
        fmax = fs / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def log_mel_spectrogram(x, fs=16000, n_fft=512, hop_length=160, n_mels=26):
    x = pre_emphasis(x)
    frames = frame_signal(x, n_fft, hop_length)
    window = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    fb = mel_filterbank(n_mels, n_fft, fs)
    mel = np.dot(spec, fb.T)
    mel = np.maximum(mel, 1e-10)
    return np.log(mel)


def dct_matrix(n_out, n_in):
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    return np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))


def mfcc(x, fs=16000, n_fft=512, hop_length=160, n_mels=26, n_mfcc=13):
    lm = log_mel_spectrogram(x, fs, n_fft, hop_length, n_mels)
    dct = dct_matrix(n_mfcc, n_mels)
    return np.dot(lm, dct.T)


def sequence_feature(x, fs=16000, n_fft=512, hop_length=160,
                     n_mels=26, n_mfcc=13, use_mfcc=True):
    if use_mfcc:
        feat = mfcc(x, fs, n_fft, hop_length, n_mels, n_mfcc)
    else:
        feat = log_mel_spectrogram(x, fs, n_fft, hop_length, n_mels)
    feat = (feat - np.mean(feat, axis=0, keepdims=True)) / (np.std(feat, axis=0, keepdims=True) + 1e-8)
    return feat.astype(np.float32)

# -*- coding: utf-8 -*-
"""第三阶段声学前端（纯 numpy，可配置）。

模块定位
--------
本项目二期的 ``src/features.py`` 只提供固定配置的 MFCC13 + 整句 CMVN。
第三阶段的探针实验表明，**动态特征（Δ/ΔΔ）+ 判别式分类器** 是当前最有效的
杠杆；同时语速/音高条件数据需要音高、能量、log-mel 等可配置通道。本模块
因此提供一个可配置、可哈希、可序列化的前端，并且：

1. 不修改 ``src/features.py``（二期代码保持原样），只复用其中的
   ``pre_emphasis / frame_signal / mel_filterbank / dct_matrix``，保证
   ``include_delta=False`` 时与旧 ``extract_mfcc`` **数值等价**。
2. 全部计算确定性（无随机数）、纯 numpy，输入输出无 NaN/Inf。
3. 特征块顺序固定，便于嵌入函数与下游模型稳定 import：

   ``[mfcc][Δmfcc][ΔΔmfcc][logmel][Δlogmel][ΔΔlogmel][pitch][Δpitch][ΔΔpitch][energy][Δenergy][ΔΔenergy]``
   （各块按 spec 开关出现；缺失块直接省略，不补零）

对齐约定
--------
所有帧级通道（MFCC / log-mel / 音高 / 能量）都使用**同一个**帧网格：
``frame_signal(audio, n_fft, hop)``，即帧长 32 ms、帧移 10 ms（16 kHz，
n_fft=512, hop=160）。短音频（不足一帧）按旧逻辑在尾部补零到一帧。

音高（F0）
----------
自相关法，逐帧估计：

* 帧内先去均值，用 FFT 计算自相关（无偏归一化除以零延迟能量）；
* 在 lag ∈ [sr/f0_max, sr/f0_min] = [40, 266]（60–400 Hz）内取归一化
  自相关峰值，峰值 ≥ ``f0_voicing_threshold`` 且帧 RMS ≥ ``rms_threshold``
  判为浊音；
* 浊音帧写入 ``log(f0 / f0_ref)``（``f0_ref=100 Hz``，量级接近 0），
  清音/静音帧写入 ``0.0``，另有一维 ``voiced`` 标志（0/1）；
* 因此音高块基础维度为 2；若打开 Δ/ΔΔ，则对 ``[logf0, voiced]`` 两维
  再做差分（各 2 维）。

能量
----
帧 log-RMS：``log(sqrt(mean(frame^2)) + 1e-8)``，基础 1 维；打开 Δ/ΔΔ 后 3 维。

CMVN
----
``use_cmvn=True`` 时对 **MFCC 块** 和 **log-mel 块** 分别沿时间轴做
``(x-mean)/(std+1e-8)``（与二期 ``extract_mfcc`` 的定义一致）。
音高/能量不做 CMVN（保持绝对量级，由下游分类器标准化）。

依赖：numpy（>=1.20）。不依赖 scipy/sklearn。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .features import dct_matrix, frame_signal, mel_filterbank, pre_emphasis

__all__ = [
    'FeatureSpec', 'extract_sequence', 'output_dim', 'delta_features',
    'stats_embedding', 'fixed_length_embedding', 'DEFAULT_STATS',
    'mel_filterbank_v3', 'pitch_contour',
]

# 统计嵌入默认统计量（与 Lead 探针一致，便于直接对比）
DEFAULT_STATS: Tuple[str, ...] = ('mean', 'std', 'min', 'max', 'p10', 'p90')

_SUPPORTED_STATS = frozenset(DEFAULT_STATS)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """可配置声学前端配置。

    frozen dataclass：可直接哈希（可作 dict key / functools.lru_cache 参数），
    并支持 ``to_dict()`` / ``from_dict()`` 做 JSON 序列化（用于实验记录与
    模型文件中的特征配置）。
    """

    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 160
    n_mels: int = 26
    n_mfcc: int = 13
    fmin: float = 50.0
    fmax: Optional[float] = None          # None -> sample_rate / 2
    preemphasis: float = 0.97
    use_cmvn: bool = True
    include_delta: bool = False
    include_delta2: bool = False
    delta_width: int = 2
    include_pitch: bool = False
    include_energy: bool = False
    include_logmel: bool = False
    include_mfcc: bool = True
    logmel_cmvn: bool = True
    f0_min: float = 60.0
    f0_max: float = 400.0
    f0_voicing_threshold: float = 0.30
    f0_rms_threshold: float = 1e-3
    f0_ref_hz: float = 100.0

    # ---- 校验 / 序列化 -------------------------------------------------
    def __post_init__(self):
        if int(self.sample_rate) <= 0:
            raise ValueError('sample_rate 必须为正数')
        if int(self.n_fft) < 16:
            raise ValueError('n_fft 太小')
        if int(self.hop) <= 0:
            raise ValueError('hop 必须为正数')
        if int(self.n_mels) <= 0:
            raise ValueError('n_mels 必须为正数')
        if int(self.n_mfcc) <= 0:
            raise ValueError('n_mfcc 必须为正数')
        if int(self.n_mfcc) > int(self.n_mels):
            raise ValueError('n_mfcc 不能大于 n_mels')
        if not (0.0 <= float(self.fmin)):
            raise ValueError('fmin 必须非负')
        if self.fmax is not None and not (float(self.fmin) < float(self.fmax)):
            raise ValueError('fmax 必须大于 fmin')
        if not (0.0 <= float(self.preemphasis) < 1.0):
            raise ValueError('preemphasis 必须在 [0,1) 内')
        if int(self.delta_width) < 1:
            raise ValueError('delta_width 必须 >= 1')
        if not (0.0 <= float(self.f0_voicing_threshold) <= 1.0):
            raise ValueError('f0_voicing_threshold 必须在 [0,1] 内')
        if not (0.0 < float(self.f0_min) < float(self.f0_max)):
            raise ValueError('需要 0 < f0_min < f0_max')
        if not bool(self.include_mfcc or self.include_logmel
                    or self.include_pitch or self.include_energy):
            raise ValueError('至少需要打开一种特征块')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSpec':
        """从 dict 恢复配置；忽略未知字段（向前兼容旧记录）。"""
        if isinstance(d, FeatureSpec):
            return d
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in valid})

    def fingerprint(self) -> str:
        """稳定短哈希，用于特征缓存文件名。"""
        import hashlib

        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------

def _cmvn(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)


def delta_features(seq: np.ndarray, width: int = 2) -> np.ndarray:
    """标准回归式差分（与二期探针一致，edge padding，O(T·D)）。"""
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2:
        raise ValueError('seq 必须是二维 [T,D]')
    if seq.shape[0] == 0:
        raise ValueError('seq 不能为空')
    width = int(width)
    if width < 1:
        raise ValueError('width 必须 >= 1')
    T = seq.shape[0]
    pad = np.pad(seq, ((width, width), (0, 0)), mode='edge')
    num = np.zeros_like(seq)
    den = 2.0 * sum(k * k for k in range(1, width + 1))
    for k in range(1, width + 1):
        num += k * (pad[width + k:width + k + T] - pad[width - k:width - k + T])
    return num / den


def mel_filterbank_v3(n_mels: int, n_fft: int, sample_rate: int,
                      fmin: float = 50.0, fmax: Optional[float] = None):
    """薄封装：复用二期 mel_filterbank，fmax=None 时取 sr/2。"""
    return mel_filterbank(n_mels, n_fft, sample_rate, fmin=fmin, fmax=fmax)


def pitch_contour(audio: np.ndarray, sr: int, n_fft: int = 512,
                  hop: int = 160, f0_min: float = 60.0,
                  f0_max: float = 400.0, voicing_threshold: float = 0.30,
                  rms_threshold: float = 1e-3,
                  f0_ref_hz: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """逐帧自相关 F0。

    返回 ``(f0_log, voiced)``，两者均为 float64 [T]：
    ``f0_log[t] = log(f0/ref)``（浊音）或 0.0（清音/静音）；
    ``voiced[t] ∈ {0.0, 1.0}``。T 与 MFCC 帧数严格一致。
    """
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError('音频为空')
    if sr <= 0:
        raise ValueError('sample_rate 必须为正数')
    lag_min = max(2, int(np.ceil(sr / float(f0_max))))
    lag_max = int(np.floor(sr / float(f0_min)))
    frames = frame_signal(x, n_fft, hop)
    if lag_max >= n_fft:
        raise ValueError('f0_min 过低：lag_max 超过帧长 n_fft')
    frames = frames - frames.mean(axis=1, keepdims=True)
    n_fft_auto = 1 << int(np.ceil(np.log2(2 * n_fft)))
    spec = np.fft.rfft(frames, n=n_fft_auto, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=n_fft_auto, axis=1)[:, :n_fft]
    ac0 = ac[:, 0].copy()
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-16)
    r = ac[:, lag_min:lag_max + 1] / (ac0[:, None] + 1e-12)
    best = np.argmax(r, axis=1)
    peak = r[np.arange(r.shape[0]), best]
    f0 = sr / (lag_min + best).astype(np.float64)
    voiced = (peak >= float(voicing_threshold)) & (rms >= float(rms_threshold))
    f0_log = np.zeros(frames.shape[0], dtype=np.float64)
    f0_log[voiced] = np.log(f0[voiced] / float(f0_ref_hz))
    return f0_log, voiced.astype(np.float64)


def _block_with_deltas(block: np.ndarray, spec: FeatureSpec) -> list:
    """按 spec 生成 [block, Δblock, ΔΔblock] 列表。"""
    out = [block]
    if spec.include_delta:
        d1 = delta_features(block, spec.delta_width)
        out.append(d1)
        if spec.include_delta2:
            out.append(delta_features(d1, spec.delta_width))
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def extract_sequence(audio, sr: Optional[int] = None,
                     spec: Optional[FeatureSpec] = None) -> np.ndarray:
    """提取帧级特征序列。

    参数
    ----
    audio : array-like，单声道波形（[-1,1] 量级，float 或 int16 均可）
    sr : 采样率；None 时使用 ``spec.sample_rate``；与 spec 不一致时报错
    spec : FeatureSpec；None 时使用默认配置（等价于旧 extract_mfcc 默认参数）

    返回
    ----
    np.float32 [T, D]，无 NaN/Inf；D 可用 ``output_dim(spec)`` 计算。
    """
    if spec is None:
        spec = FeatureSpec()
    if not isinstance(spec, FeatureSpec):
        raise TypeError('spec 必须是 FeatureSpec')
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError('音频为空')
    if sr is None:
        sr = int(spec.sample_rate)
    sr = int(sr)
    if sr != int(spec.sample_rate):
        raise ValueError(
            f'采样率不匹配：音频 {sr} Hz，FeatureSpec.sample_rate={spec.sample_rate} Hz')

    n_fft, hop = int(spec.n_fft), int(spec.hop)
    x_pre = pre_emphasis(x, spec.preemphasis)
    frames = frame_signal(x_pre, n_fft, hop)
    window = np.hamming(n_fft)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    fb = mel_filterbank_v3(spec.n_mels, n_fft, sr, spec.fmin, spec.fmax)
    mel = np.log(np.maximum(np.dot(power, fb.T), 1e-10))

    parts = []
    if spec.include_mfcc:
        mfcc = np.dot(mel, dct_matrix(int(spec.n_mfcc), int(spec.n_mels)).T)
        if spec.use_cmvn:
            mfcc = _cmvn(mfcc)
        parts.extend(_block_with_deltas(mfcc, spec))
    if spec.include_logmel:
        logmel = _cmvn(mel) if (spec.use_cmvn and spec.logmel_cmvn) else mel
        parts.extend(_block_with_deltas(logmel, spec))
    if spec.include_pitch:
        f0_log, voiced = pitch_contour(
            x, sr, n_fft=n_fft, hop=hop, f0_min=spec.f0_min, f0_max=spec.f0_max,
            voicing_threshold=spec.f0_voicing_threshold,
            rms_threshold=spec.f0_rms_threshold, f0_ref_hz=spec.f0_ref_hz)
        parts.extend(_block_with_deltas(np.stack([f0_log, voiced], axis=1), spec))
    if spec.include_energy:
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-16)
        log_rms = np.log(rms + 1e-8)[:, None]
        parts.extend(_block_with_deltas(log_rms, spec))

    feat = np.concatenate(parts, axis=1).astype(np.float32)
    if feat.ndim != 2 or feat.shape[0] < 1 or feat.shape[1] < 1:
        raise ValueError(f'特征形状非法: {feat.shape}')
    if not np.all(np.isfinite(feat)):
        raise ValueError('特征包含 NaN/Inf')
    return feat


def output_dim(spec: Optional[FeatureSpec] = None) -> int:
    """返回 ``extract_sequence`` 的特征维度 D（不读音频）。"""
    if spec is None:
        spec = FeatureSpec()
    mult = 1 + (1 if spec.include_delta else 0) + (
        1 if (spec.include_delta and spec.include_delta2) else 0)
    d = 0
    if spec.include_mfcc:
        d += int(spec.n_mfcc) * mult
    if spec.include_logmel:
        d += int(spec.n_mels) * mult
    if spec.include_pitch:
        d += 2 * mult
    if spec.include_energy:
        d += 1 * mult
    return int(d)


# ---------------------------------------------------------------------------
# 嵌入：整句统计量 / 固定帧数展平
# ---------------------------------------------------------------------------

def _trim_sequence(seq: np.ndarray, trim: Optional[Sequence[float]]) -> np.ndarray:
    """按相对时间比例裁剪 [a, b) 区间，用于只取语音起始段落。"""
    if trim is None:
        return seq
    a, b = float(trim[0]), float(trim[1])
    if not (0.0 <= a < b <= 1.0):
        raise ValueError('trim 必须满足 0 <= a < b <= 1')
    T = seq.shape[0]
    i0 = int(np.floor(T * a))
    i1 = int(np.ceil(T * b))
    i0 = min(max(i0, 0), T - 1)
    i1 = min(max(i1, i0 + 1), T)
    return seq[i0:i1]


def stats_embedding(seq: np.ndarray, stats: Sequence[str] = DEFAULT_STATS,
                    trim: Optional[Sequence[float]] = None) -> np.ndarray:
    """整句统计嵌入：沿时间轴拼接 ``mean/std/min/max/p10/p90``。

    返回 float32 [len(stats) * D]；对任意 T >= 1 都产生有限值。
    统计量顺序严格按传入顺序，便于实验记录复现。
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2:
        raise ValueError('seq 必须是二维 [T,D]')
    if seq.shape[0] == 0:
        raise ValueError('seq 不能为空')
    stats = tuple(stats)
    bad = [s for s in stats if s not in _SUPPORTED_STATS]
    if bad:
        raise ValueError(f'不支持的统计量: {bad}，可选 {sorted(_SUPPORTED_STATS)}')
    if not stats:
        raise ValueError('stats 不能为空')
    seq = _trim_sequence(seq, trim)
    out = []
    for s in stats:
        if s == 'mean':
            out.append(seq.mean(axis=0))
        elif s == 'std':
            out.append(seq.std(axis=0))
        elif s == 'min':
            out.append(seq.min(axis=0))
        elif s == 'max':
            out.append(seq.max(axis=0))
        elif s == 'p10':
            out.append(np.percentile(seq, 10, axis=0))
        elif s == 'p90':
            out.append(np.percentile(seq, 90, axis=0))
    vec = np.concatenate(out).astype(np.float32)
    if not np.all(np.isfinite(vec)):
        raise ValueError('统计嵌入包含 NaN/Inf')
    return vec


def fixed_length_embedding(seq: np.ndarray, n_frames: int = 40,
                           trim: Optional[Sequence[float]] = None) -> np.ndarray:
    """固定帧数嵌入：沿时间轴线性插值到 ``n_frames`` 帧后展平。

    返回 float32 [n_frames * D]。T=1 时重复该帧；T 任意时用
    ``np.interp`` 的线性插值（确定性）。
    """
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2:
        raise ValueError('seq 必须是二维 [T,D]')
    if seq.shape[0] == 0:
        raise ValueError('seq 不能为空')
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError('n_frames 必须 >= 1')
    seq = _trim_sequence(seq, trim)
    T = seq.shape[0]
    if T == n_frames:
        fixed = seq
    elif T == 1:
        fixed = np.repeat(seq, n_frames, axis=0)
    else:
        src = np.linspace(0.0, T - 1.0, T)
        dst = np.linspace(0.0, T - 1.0, n_frames)
        fixed = np.stack([np.interp(dst, src, seq[:, j]) for j in range(seq.shape[1])],
                         axis=1)
    vec = fixed.reshape(-1).astype(np.float32)
    if not np.all(np.isfinite(vec)):
        raise ValueError('固定长度嵌入包含 NaN/Inf')
    return vec

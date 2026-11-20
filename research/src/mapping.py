# -*- coding: utf-8 -*-
"""逐元素OLS/Ridge映射，以及SVD候选实现。"""
from dataclasses import dataclass
import numpy as np
from .alignment import fixed_length_sequence, dtw_align_pair


def _validate_pairs(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError("X和Y必须是[N,T,D]三维数组")
    if X.shape != Y.shape:
        raise ValueError(f"X和Y形状不一致: {X.shape} vs {Y.shape}")
    if X.shape[0] < 2:
        raise ValueError("至少需要2个配对样本")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("X或Y包含NaN/Inf")
    return X, Y


class _ElementwiseMappingBase:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = bool(fit_intercept)
        self.omega_ = None
        self.epsilon_ = None
        self.name_ = 'base'

    def _fit_element(self, x, y):
        raise NotImplementedError

    def fit(self, X, Y):
        X, Y = _validate_pairs(X, Y)
        n, t, d = X.shape
        omega = np.zeros((t, d), dtype=np.float64)
        epsilon = np.zeros((t, d), dtype=np.float64)
        for i in range(t):
            for j in range(d):
                omega[i, j], epsilon[i, j] = self._fit_element(X[:, i, j], Y[:, i, j])
        self.omega_ = omega
        self.epsilon_ = epsilon
        return self

    def transform(self, X):
        if self.omega_ is None:
            raise ValueError("映射尚未fit")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("transform输入必须是[T,D]")
        if X.shape != self.omega_.shape:
            raise ValueError(f"输入形状不匹配: {X.shape} vs {self.omega_.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("transform输入包含NaN/Inf")
        return X * self.omega_ + self.epsilon_

    def to_dict(self):
        if self.omega_ is None:
            raise ValueError("映射尚未fit")
        return {
            'name': self.name_,
            'fit_intercept': self.fit_intercept,
            'omega': self.omega_.tolist(),
            'epsilon': self.epsilon_.tolist(),
        }


class OLSMapping(_ElementwiseMappingBase):
    """原论文候选实现：逐元素一元线性回归 y = omega*x + epsilon。"""
    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)
        self.name_ = 'OLS'

    def _fit_element(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.fit_intercept:
            xm = x.mean()
            ym = y.mean()
            xc = x - xm
            sxx = float(np.sum(xc * xc))
            if sxx < 1e-12:
                raise ValueError("OLS：某个元素在所有训练对上方差为0，无法估计斜率")
            slope = float(np.sum(xc * (y - ym)) / sxx)
            intercept = float(ym - slope * xm)
        else:
            sxx = float(np.sum(x * x))
            if sxx < 1e-12:
                raise ValueError("OLS无截距：某个元素在所有训练对上方差为0，无法估计斜率")
            slope = float(np.sum(x * y) / sxx)
            intercept = 0.0
        return slope, intercept


class RidgeMapping(_ElementwiseMappingBase):
    """逐元素Ridge映射：y = omega*x + epsilon，使用中心化闭式解。"""
    def __init__(self, alpha=1e-3, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)
        if alpha < 0:
            raise ValueError("alpha不能为负")
        self.alpha = float(alpha)
        self.name_ = 'Ridge'

    def _fit_element(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.fit_intercept:
            xm = x.mean()
            ym = y.mean()
            xc = x - xm
            yc = y - ym
            denom = float(np.sum(xc * xc) + self.alpha)
            slope = float(np.sum(xc * yc) / denom) if denom > 0 else 0.0
            intercept = float(ym - slope * xm)
        else:
            denom = float(np.sum(x * x) + self.alpha)
            slope = float(np.sum(x * y) / denom) if denom > 0 else 0.0
            intercept = 0.0
        return slope, intercept


def mapping_diagnostics(model, X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError("mapping_diagnostics需要[N,T,D]输入")
    mapped = np.stack([model.transform(X[k]) for k in range(X.shape[0])], axis=0)
    diff = mapped - Y
    return {
        'rmse': float(np.sqrt(np.mean(diff * diff))),
        'mae': float(np.mean(np.abs(diff))),
        'mean_cosine': float(np.mean([
            np.dot(mapped[k].ravel(), np.asarray(Y)[k].ravel()) /
            (np.linalg.norm(mapped[k].ravel()) * np.linalg.norm(np.asarray(Y)[k].ravel()) + 1e-12)
            for k in range(mapped.shape[0])
        ])),
    }


def align_pair_sequences(x, y, mode='fixed_length', target_len=100):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x/y必须是[T,D]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x/y特征维度不一致")
    if mode == 'fixed_length':
        return fixed_length_sequence(x, target_len), fixed_length_sequence(y, target_len)
    if mode == 'dtw_then_fixed':
        ax, ay, _, _ = dtw_align_pair(x, y)
        return fixed_length_sequence(ax, target_len), fixed_length_sequence(ay, target_len)
    raise ValueError("alignment必须是fixed_length或dtw_then_fixed")


def pair_records_by_command(records, source_language=None, target_language=None):
    """按命令配对；source/target可为空，表示不按language过滤。"""
    grouped = {}
    for r in records:
        grouped.setdefault(r.command, []).append(r)
    pairs = []
    for cmd, items in grouped.items():
        sources = [r for r in items if source_language is None or getattr(r, 'language', '') == source_language]
        targets = [r for r in items if target_language is None or getattr(r, 'language', '') == target_language]
        for s in sources:
            for t in targets:
                if s.utterance_id != t.utterance_id:
                    pairs.append((s, t))
    if not pairs:
        raise ValueError("没有构造出任何训练配对；请检查language过滤和命令标签")
    return pairs


class SVDFeatureTransformer:
    """SVD候选实现。literal_rank_k为论文直译；global_projection为候选解释。"""
    def __init__(self, mode='none', rank=None):
        if mode not in ('none', 'R0_literal_rank_k', 'R1_global_projection'):
            raise ValueError("未知SVD模式")
        self.mode = mode
        self.rank = rank
        self.V_k = None
        self.input_dim_ = None

    def fit(self, sequences):
        seqs = [np.asarray(s, dtype=np.float64) for s in sequences]
        if not seqs:
            raise ValueError("SVD fit需要非空序列列表")
        if self.mode in ('none', 'R0_literal_rank_k'):
            self.input_dim_ = seqs[0].shape[1]
            return self
        if self.mode == 'R1_global_projection':
            F = np.vstack(seqs)
            if not np.all(np.isfinite(F)):
                raise ValueError("SVD输入包含NaN/Inf")
            _, _, vt = np.linalg.svd(F, full_matrices=False)
            k = self.rank if self.rank is not None else min(F.shape)
            k = max(1, min(int(k), vt.shape[0]))
            self.V_k = vt[:k].T
            self.input_dim_ = F.shape[1]
            return self
        raise ValueError("未知SVD模式")

    def transform(self, seq):
        seq = np.asarray(seq, dtype=np.float64)
        if seq.ndim != 2:
            raise ValueError("SVD transform输入必须是[T,D]")
        if self.mode == 'none':
            return seq
        if self.mode == 'R0_literal_rank_k':
            u, s, vt = np.linalg.svd(seq, full_matrices=False)
            k = self.rank if self.rank is not None else min(seq.shape)
            k = max(1, min(int(k), len(s)))
            return (u[:, :k] * s[:k]) @ vt[:k, :]
        if self.mode == 'R1_global_projection':
            if self.V_k is None:
                raise ValueError("请先fit全局SVD")
            if seq.shape[1] != self.V_k.shape[0]:
                raise ValueError("输入特征维度与全局SVD不匹配")
            return seq @ self.V_k
        raise ValueError("未知SVD模式")


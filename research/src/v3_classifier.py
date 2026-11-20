# -*- coding: utf-8 -*-
"""第三阶段判别式分类器（纯 numpy，不依赖 sklearn）。

提供
----
* ``MultinomialLogisticRegression``：L2 正则多项逻辑回归，内置标准化、
  确定性全批量训练（Armijo 回溯线搜索，无随机性）、多分类/二分类通用；
  提供 ``save`` / ``load``（npz + JSON 元数据，带版本字段）。
* ``LinearSVM``：One-vs-Rest + SGD hinge，作为线性对照。
* ``kNNClassifier``：余弦相似度 kNN，作为非参数对照。

设计要点
--------
1. 纯 numpy + 标准库；不 import sklearn/scipy，便于最终系统固定依赖。
2. 目标函数：``-mean log p(y|x) + 0.5*l2*||W||^2``（截距不正则化）。
3. 优化：全批量梯度下降 + Armijo 回溯（确定性），训练过程有 loss 记录；
   给定 seed 只影响 SVM/kNN 中无实际随机性的初始化/打乱，保证同数据同结果。
4. 标准化：``fit`` 时记录训练集 mean/std，``predict`` 复用；可用
   ``standardize=False`` 关闭（消融时由 val 选择）。
5. 模型文件：压缩 npz 内嵌 ``__meta__``（JSON，含 version/class/hyperparams/
   classes/feature_config），数组为 coef_/intercept_/mu_/sd_ 等。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

MODEL_FORMAT_VERSION = '3.0.0'


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

def _as_2d_float(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f'X 必须是二维 [N,D]，当前 shape={X.shape}')
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError(f'X 不能有空维度，当前 shape={X.shape}')
    if not np.all(np.isfinite(X)):
        raise ValueError('X 包含 NaN/Inf')
    return X


def _fit_standardize(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _resolve_path(path) -> Path:
    path = Path(path)
    if path.suffix.lower() != '.npz':
        path = path.with_name(path.name + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_npz(path, cls_name: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    path = _resolve_path(path)
    payload = dict(arrays)
    payload['__meta__'] = np.array(json.dumps(
        {'class': cls_name, 'version': MODEL_FORMAT_VERSION, **meta},
        ensure_ascii=False, sort_keys=True))
    np.savez_compressed(path, **payload)
    return str(path)


def _load_npz(path):
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f'模型文件不存在: {path}')
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['__meta__']))
        arrays = {k: np.array(z[k]) for k in z.files if k != '__meta__'}
    return meta, arrays


def _check_fitted(est) -> None:
    if not hasattr(est, 'classes_'):
        raise RuntimeError('模型尚未 fit')


def _encode_classes(classes) -> Dict[str, Any]:
    """类别标签序列化：保留整数/浮点 dtype，字符串走 object。"""
    arr = np.asarray(classes)
    if arr.dtype.kind in 'iuf':
        return {'dtype': str(arr.dtype), 'values': arr.tolist()}
    return {'dtype': 'str', 'values': [str(v) for v in arr]}


def _decode_classes(meta_classes) -> np.ndarray:
    if isinstance(meta_classes, dict):
        if meta_classes.get('dtype') == 'str':
            return np.asarray(meta_classes['values'], dtype=object)
        return np.asarray(meta_classes['values'], dtype=meta_classes.get('dtype', 'int64'))
    return np.asarray(meta_classes, dtype=object)


def _lbfgs_minimize(fun, theta0: np.ndarray, max_iter: int = 300,
                    memory: int = 10, tol: float = 1e-10,
                    gtol: float = 1e-8) -> tuple:
    """确定性 L-BFGS（Armijo 回溯线搜索，双循环递归）。

    ``fun(theta) -> (loss, grad)``。用于逻辑回归训练：相比全批量梯度下降，
    在高维且高度相关的嵌入（如 fixed40 展平特征）上收敛快得多。
    返回 ``(theta, loss, history)``。
    """
    theta = np.asarray(theta0, dtype=np.float64).copy()
    f, g = fun(theta)
    history = [float(f)]
    s_list, y_list, rho_list = [], [], []
    for _ in range(int(max_iter)):
        gnorm = float(np.linalg.norm(g))
        if gnorm <= gtol:
            break
        # 双循环递归求下降方向 d = -H*g
        q = g.copy()
        alphas = []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * float(np.dot(s, q))
            alphas.append(a)
            q = q - a * y
        if s_list:
            s_last, y_last = s_list[-1], y_list[-1]
            yy = float(np.dot(y_last, y_last))
            gamma = float(np.dot(s_last, y_last)) / yy if yy > 1e-24 else 1.0
        else:
            gamma = 1.0 / max(gnorm, 1e-12)
        r = gamma * q
        for (s, y, rho), a in zip(zip(s_list, y_list, rho_list), reversed(alphas)):
            b = rho * float(np.dot(y, r))
            r = r + s * (a - b)
        d = -r
        gd = float(np.dot(g, d))
        if gd >= 0.0:                     # 非下降方向：重置记忆并退回最速下降
            d = -g
            gd = -gnorm * gnorm
            s_list, y_list, rho_list = [], [], []
        step = 1.0
        ok = False
        for _ in range(40):
            theta_new = theta + step * d
            f_new, g_new = fun(theta_new)
            if f_new <= f + 1e-4 * step * gd:
                ok = True
                break
            step *= 0.5
        if not ok:
            break
        s = theta_new - theta
        y = g_new - g
        sy = float(np.dot(s, y))
        if sy > 1e-14:
            s_list.append(s)
            y_list.append(y)
            rho_list.append(1.0 / sy)
            if len(s_list) > int(memory):
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
        rel = abs(f - f_new) / max(1.0, abs(f))
        theta, f, g = theta_new, f_new, g_new
        history.append(float(f))
        if rel < tol:
            break
    return theta, float(f), history


# ---------------------------------------------------------------------------
# 多项逻辑回归
# ---------------------------------------------------------------------------

class MultinomialLogisticRegression:
    """L2 正则多项逻辑回归（numpy 实现，确定性训练）。

    参数
    ----
    l2 : float
        正则强度（只作用于权重 W，不作用于截距 b）。
    standardize : bool
        是否在内部做特征标准化（默认 True；高维嵌入必须开）。
    epochs : int
        L-BFGS 最大迭代次数（默认 2000，通常远早于上限收敛）。
    tol : float
        相对 loss 改善低于该值时提前停止。
    lr, max_lr, momentum : float
        历史接口字段：当前训练器为确定性 L-BFGS，三者不参与优化
        （保留仅为兼容早期探针脚本与序列化记录）。
    seed : int
        仅保留接口；本实现无随机初始化，给定数据结果唯一。
    class_weight : {'balanced', None}
        'balanced' 时按类别频率反比加权（小类更有影响）。

    属性（fit 后）
    -------------
    classes_ : np.ndarray  排序后的类别标签
    coef_ : np.ndarray [C,D]
    intercept_ : np.ndarray [C]
    mu_, sd_ : 标准化参数
    loss_history_ : list[float]
    feature_config : dict | None  可放置 FeatureSpec.to_dict() 用于部署记录
    """

    def __init__(self, l2: float = 1e-3, standardize: bool = True,
                 lr: float = 0.5, max_lr: float = 2.0, epochs: int = 2000,
                 tol: float = 1e-9, momentum: float = 0.0, seed: int = 0,
                 class_weight: Optional[str] = None):
        if l2 < 0:
            raise ValueError('l2 必须 >= 0')
        if not (0.0 < lr <= 1.0):
            raise ValueError('lr 必须在 (0,1] 内')
        if epochs < 1:
            raise ValueError('epochs 必须 >= 1')
        if class_weight not in (None, 'balanced'):
            raise ValueError("class_weight 只支持 None 或 'balanced'")
        self.l2 = float(l2)
        self.standardize = bool(standardize)
        self.lr = float(lr)
        self.max_lr = float(max_lr)
        self.epochs = int(epochs)
        self.tol = float(tol)
        self.momentum = float(momentum)
        self.seed = int(seed)
        self.class_weight = class_weight
        self.feature_config: Optional[Dict[str, Any]] = None

    # ---- 内部：目标函数 ----------------------------------------------
    def _objective(self, Xs: np.ndarray, Y: np.ndarray, W: np.ndarray,
                   b: np.ndarray, sw: np.ndarray) -> float:
        logits = Xs @ W + b
        z = logits - logits.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True) + 1e-300)
        logp = z - logsumexp
        nll = -np.sum(sw[:, None] * Y * logp) / Xs.shape[0]
        return float(nll + 0.5 * self.l2 * np.sum(W * W))

    def _gradient(self, Xs, Y, W, b, sw):
        N = Xs.shape[0]
        p = _softmax(Xs @ W + b)
        g = (p - Y) * sw[:, None] / N
        gW = Xs.T @ g + self.l2 * W
        gb = g.sum(axis=0)
        return gW, gb

    # ---- 训练 ---------------------------------------------------------
    def fit(self, X, y, feature_config: Optional[Dict[str, Any]] = None):
        X = _as_2d_float(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError('X 与 y 样本数不一致')
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError('至少需要两个类别')
        self.classes_ = classes
        self.n_classes_ = int(classes.size)
        self.n_features_in_ = int(X.shape[1])
        self.feature_config = feature_config
        idx = np.searchsorted(classes, y)
        Y = np.eye(self.n_classes_)[idx]
        if self.class_weight == 'balanced':
            counts = np.bincount(idx, minlength=self.n_classes_).astype(np.float64)
            w = len(y) / (self.n_classes_ * np.maximum(counts, 1.0))
            sw = w[idx]
        else:
            sw = np.ones(len(y), dtype=np.float64)

        if self.standardize:
            self.mu_, self.sd_ = _fit_standardize(X)
            Xs = (X - self.mu_) / self.sd_
        else:
            self.mu_ = np.zeros(X.shape[1])
            self.sd_ = np.ones(X.shape[1])
            Xs = X

        D = Xs.shape[1]
        self._train_X = Xs
        self._train_Y = Y
        self._train_sw = sw

        def fun(theta):
            W = theta[:D * self.n_classes_].reshape(D, self.n_classes_)
            b = theta[D * self.n_classes_:]
            loss = self._objective(Xs, Y, W, b, sw)
            gW, gb = self._gradient(Xs, Y, W, b, sw)
            grad = np.concatenate([gW.reshape(-1), gb])
            return loss, grad

        theta0 = np.zeros(D * self.n_classes_ + self.n_classes_)
        theta, loss, history = _lbfgs_minimize(
            fun, theta0, max_iter=self.epochs, tol=self.tol)
        W = theta[:D * self.n_classes_].reshape(D, self.n_classes_)
        b = theta[D * self.n_classes_:]
        self.coef_ = W.T.copy()          # [C, D]
        self.intercept_ = b.copy()
        self.loss_history_ = history
        self.final_loss_ = float(loss)
        self._train_X = None
        self._train_Y = None
        self._train_sw = None
        return self

    # ---- 推理 ---------------------------------------------------------
    def decision_function(self, X) -> np.ndarray:
        _check_fitted(self)
        X = _as_2d_float(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'特征维度不匹配：模型 {self.n_features_in_}，输入 {X.shape[1]}')
        Xs = (X - self.mu_) / self.sd_
        return Xs @ self.coef_.T + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        return _softmax(self.decision_function(X))

    def predict(self, X) -> np.ndarray:
        _check_fitted(self)
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    @property
    def n_params(self) -> int:
        _check_fitted(self)
        return int(self.coef_.size + self.intercept_.size)

    # ---- 持久化 -------------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        _check_fitted(self)
        return {
            'class': type(self).__name__, 'version': MODEL_FORMAT_VERSION,
            'hyperparams': {'l2': self.l2, 'standardize': self.standardize,
                            'lr': self.lr, 'max_lr': self.max_lr,
                            'epochs': self.epochs, 'tol': self.tol,
                            'momentum': self.momentum, 'seed': self.seed,
                            'class_weight': self.class_weight},
            'classes': _encode_classes(self.classes_),
            'feature_config': self.feature_config,
            'coef': self.coef_.tolist(), 'intercept': self.intercept_.tolist(),
            'mu': self.mu_.tolist(), 'sd': self.sd_.tolist(),
        }

    def save(self, path) -> str:
        _check_fitted(self)
        meta = {
            'hyperparams': {'l2': self.l2, 'standardize': self.standardize,
                            'lr': self.lr, 'max_lr': self.max_lr,
                            'epochs': self.epochs, 'tol': self.tol,
                            'momentum': self.momentum, 'seed': self.seed,
                            'class_weight': self.class_weight},
            'classes': _encode_classes(self.classes_),
            'feature_config': self.feature_config,
        }
        arrays = {'coef': self.coef_, 'intercept': self.intercept_,
                  'mu': self.mu_, 'sd': self.sd_}
        return _save_npz(path, type(self).__name__, meta, arrays)

    @classmethod
    def load(cls, path) -> 'MultinomialLogisticRegression':
        meta, arrays = _load_npz(path)
        if meta.get('class') != cls.__name__:
            raise ValueError(f"模型类型不匹配：文件是 {meta.get('class')}，期望 {cls.__name__}")
        hp = dict(meta.get('hyperparams') or {})
        obj = cls(**{k: v for k, v in hp.items() if k in {
            'l2', 'standardize', 'lr', 'max_lr', 'epochs', 'tol', 'momentum',
            'seed', 'class_weight'}})
        obj.classes_ = _decode_classes(meta['classes'])
        obj.n_classes_ = int(len(obj.classes_))
        obj.coef_ = np.asarray(arrays['coef'], dtype=np.float64)
        obj.intercept_ = np.asarray(arrays['intercept'], dtype=np.float64)
        obj.mu_ = np.asarray(arrays['mu'], dtype=np.float64)
        obj.sd_ = np.asarray(arrays['sd'], dtype=np.float64)
        obj.n_features_in_ = int(obj.coef_.shape[1])
        obj.feature_config = meta.get('feature_config')
        obj.loss_history_ = []
        return obj


# ---------------------------------------------------------------------------
# 线性 SVM（One-vs-Rest + SGD hinge）
# ---------------------------------------------------------------------------

class LinearSVM:
    """One-vs-Rest 线性 SVM：min 0.5*||w||^2 + C * Σ hinge。

    SGD 小批量训练，打乱顺序由 ``seed`` 决定（确定性）。
    概率输出为 decision score 的 softmax，仅用于统一接口，**未校准**。
    """

    def __init__(self, c: float = 1.0, standardize: bool = True,
                 lr: float = 0.01, epochs: int = 300, batch_size: int = 32,
                 seed: int = 0):
        if c <= 0:
            raise ValueError('C 必须 > 0')
        if lr <= 0:
            raise ValueError('lr 必须 > 0')
        if epochs < 1:
            raise ValueError('epochs 必须 >= 1')
        if batch_size < 1:
            raise ValueError('batch_size 必须 >= 1')
        self.c = float(c)
        self.standardize = bool(standardize)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.feature_config: Optional[Dict[str, Any]] = None

    def fit(self, X, y, feature_config: Optional[Dict[str, Any]] = None):
        X = _as_2d_float(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError('X 与 y 样本数不一致')
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError('至少需要两个类别')
        self.n_classes_ = int(self.classes_.size)
        self.n_features_in_ = int(X.shape[1])
        self.feature_config = feature_config
        if self.standardize:
            self.mu_, self.sd_ = _fit_standardize(X)
            Xs = (X - self.mu_) / self.sd_
        else:
            self.mu_ = np.zeros(X.shape[1])
            self.sd_ = np.ones(X.shape[1])
            Xs = X
        idx = np.searchsorted(self.classes_, y)
        N, D = Xs.shape
        C = self.c
        W = np.zeros((self.n_classes_, D))
        b = np.zeros(self.n_classes_)
        class_ids = np.arange(self.n_classes_)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            perm = rng.permutation(N)
            for start in range(0, N, self.batch_size):
                bi = perm[start:start + self.batch_size]
                Xb = Xs[bi]
                yb = idx[bi]
                # OVR hinge 梯度向量化（batch×C 同时计算，与逐类循环数学等价）
                scores = Xb @ W.T + b[None, :]              # [B, C]
                ysign = np.where(yb[:, None] == class_ids[None, :], 1.0, -1.0)
                active = (ysign * scores) < 1.0             # hinge 激活掩码
                coef = ysign * active                       # [B, C]
                scale = C / float(len(bi))
                Wg = -scale * (coef.T @ Xb)                 # [C, D]
                bg = -scale * coef.sum(axis=0)              # [C]
                W = W - self.lr * (W + Wg)
                b = b - self.lr * bg
        self.coef_ = W
        self.intercept_ = b
        return self

    def decision_function(self, X) -> np.ndarray:
        _check_fitted(self)
        X = _as_2d_float(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'特征维度不匹配：模型 {self.n_features_in_}，输入 {X.shape[1]}')
        return ((X - self.mu_) / self.sd_) @ self.coef_.T + self.intercept_

    def predict(self, X) -> np.ndarray:
        _check_fitted(self)
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        _check_fitted(self)
        return _softmax(self.decision_function(X))

    @property
    def n_params(self) -> int:
        _check_fitted(self)
        return int(self.coef_.size + self.intercept_.size)

    def save(self, path) -> str:
        _check_fitted(self)
        meta = {'hyperparams': {'c': self.c, 'standardize': self.standardize,
                                'lr': self.lr, 'epochs': self.epochs,
                                'batch_size': self.batch_size, 'seed': self.seed},
                'classes': _encode_classes(self.classes_),
                'feature_config': self.feature_config}
        return _save_npz(path, type(self).__name__, meta,
                         {'coef': self.coef_, 'intercept': self.intercept_,
                          'mu': self.mu_, 'sd': self.sd_})

    @classmethod
    def load(cls, path) -> 'LinearSVM':
        meta, arrays = _load_npz(path)
        if meta.get('class') != cls.__name__:
            raise ValueError(f"模型类型不匹配：文件是 {meta.get('class')}，期望 {cls.__name__}")
        hp = dict(meta.get('hyperparams') or {})
        obj = cls(**{k: v for k, v in hp.items() if k in {
            'c', 'standardize', 'lr', 'epochs', 'batch_size', 'seed'}})
        obj.classes_ = _decode_classes(meta['classes'])
        obj.n_classes_ = int(len(obj.classes_))
        obj.coef_ = np.asarray(arrays['coef'], dtype=np.float64)
        obj.intercept_ = np.asarray(arrays['intercept'], dtype=np.float64)
        obj.mu_ = np.asarray(arrays['mu'], dtype=np.float64)
        obj.sd_ = np.asarray(arrays['sd'], dtype=np.float64)
        obj.n_features_in_ = int(obj.coef_.shape[1])
        obj.feature_config = meta.get('feature_config')
        return obj


# ---------------------------------------------------------------------------
# kNN（余弦相似度）
# ---------------------------------------------------------------------------

class KNNClassifier:
    """余弦相似度 kNN（uniform 或 similarity 加权投票）。"""

    def __init__(self, k: int = 5, weights: str = 'uniform', seed: int = 0):
        if k < 1:
            raise ValueError('k 必须 >= 1')
        if weights not in ('uniform', 'similarity'):
            raise ValueError("weights 只支持 'uniform' 或 'similarity'")
        self.k = int(k)
        self.weights = weights
        self.seed = int(seed)
        self.feature_config: Optional[Dict[str, Any]] = None

    @staticmethod
    def _normalize(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n = np.where(n < 1e-12, 1.0, n)
        return X / n

    def fit(self, X, y, feature_config: Optional[Dict[str, Any]] = None):
        X = _as_2d_float(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError('X 与 y 样本数不一致')
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.size)
        self.n_features_in_ = int(X.shape[1])
        self.feature_config = feature_config
        self.X_ = self._normalize(X)
        self.y_ = np.searchsorted(self.classes_, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        _check_fitted(self)
        X = self._normalize(_as_2d_float(X))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'特征维度不匹配：模型 {self.n_features_in_}，输入 {X.shape[1]}')
        k = min(self.k, self.X_.shape[0])
        sim = X @ self.X_.T                      # [N, N_train]
        out = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        for i in range(X.shape[0]):
            top = np.argsort(-sim[i], kind='stable')[:k]
            if self.weights == 'uniform':
                w = np.ones(k)
            else:
                w = np.maximum(sim[i, top], 0.0) + 1e-12
            np.add.at(out[i], self.y_[top], w)
        return out / (out.sum(axis=1, keepdims=True) + 1e-300)

    def predict(self, X) -> np.ndarray:
        _check_fitted(self)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def n_params(self) -> int:
        _check_fitted(self)
        return int(self.X_.size)

    def save(self, path) -> str:
        _check_fitted(self)
        meta = {'hyperparams': {'k': self.k, 'weights': self.weights,
                                'seed': self.seed},
                'classes': _encode_classes(self.classes_),
                'feature_config': self.feature_config}
        return _save_npz(path, type(self).__name__, meta,
                         {'X': self.X_, 'y': self.y_, 'mu': np.zeros(0),
                          'sd': np.zeros(0)})

    @classmethod
    def load(cls, path) -> 'KNNClassifier':
        meta, arrays = _load_npz(path)
        if meta.get('class') != cls.__name__:
            raise ValueError(f"模型类型不匹配：文件是 {meta.get('class')}，期望 {cls.__name__}")
        hp = dict(meta.get('hyperparams') or {})
        obj = cls(**{k: v for k, v in hp.items() if k in {'k', 'weights', 'seed'}})
        obj.classes_ = _decode_classes(meta['classes'])
        obj.n_classes_ = int(len(obj.classes_))
        obj.X_ = np.asarray(arrays['X'], dtype=np.float64)
        obj.y_ = np.asarray(arrays['y'], dtype=np.int64)
        obj.n_features_in_ = int(obj.X_.shape[1])
        obj.feature_config = meta.get('feature_config')
        return obj


# ---------------------------------------------------------------------------
# 通用加载
# ---------------------------------------------------------------------------

_CLASSES = {
    'MultinomialLogisticRegression': MultinomialLogisticRegression,
    'LinearSVM': LinearSVM,
    'KNNClassifier': KNNClassifier,
}


def load_model(path):
    """按模型文件内的 class 字段分派加载（带版本检查）。"""
    meta, _ = _load_npz(path)
    name = meta.get('class')
    if name not in _CLASSES:
        raise ValueError(f'未知模型类型: {name}')
    if meta.get('version') != MODEL_FORMAT_VERSION:
        raise ValueError(f"模型版本不兼容：{meta.get('version')} != {MODEL_FORMAT_VERSION}")
    return _CLASSES[name].load(path)

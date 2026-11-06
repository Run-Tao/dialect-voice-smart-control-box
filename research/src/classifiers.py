# -*- coding: utf-8 -*-
"""纯numpy的标准化、Softmax回归和线性SVM。"""
import numpy as np


class StandardScalerNumpy:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X必须是非空二维矩阵")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("请先fit")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, l2=1e-3, learning_rate=0.1, epochs=300):
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.classes_ = None
        self.W = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("至少需要两个类别")
        n, d = X.shape
        c = len(self.classes_)
        Y = np.zeros((n, c))
        for i, label in enumerate(self.classes_):
            Y[:, i] = (y == label).astype(float)
        self.W = np.zeros((d, c))
        self.b = np.zeros(c)
        for _ in range(self.epochs):
            P = _softmax(X @ self.W + self.b)
            gradW = X.T @ (P - Y) / n + self.l2 * self.W
            gradb = np.mean(P - Y, axis=0)
            self.W -= self.learning_rate * gradW
            self.b -= self.learning_rate * gradb
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return _softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LinearSVM:
    def __init__(self, l2=1e-3, learning_rate=0.05, epochs=500):
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.classes_ = None
        self.W = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("至少需要两个类别")
        n, d = X.shape
        c = len(self.classes_)
        Y = np.zeros((n, c))
        for i, label in enumerate(self.classes_):
            Y[:, i] = np.where(y == label, 1.0, -1.0)
        self.W = np.zeros((d, c))
        self.b = np.zeros(c)
        for _ in range(self.epochs):
            scores = X @ self.W + self.b
            margin = 1.0 - Y * scores
            active = (margin > 0).astype(float)
            gradW = -(X.T @ (Y * active)) / n + self.l2 * self.W
            gradb = -np.mean(Y * active, axis=0)
            self.W -= self.learning_rate * gradW
            self.b -= self.learning_rate * gradb
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.W + self.b

    def predict_proba(self, X):
        return _softmax(self.decision_function(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

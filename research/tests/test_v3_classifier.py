# -*- coding: utf-8 -*-
"""第三阶段判别式分类器测试：确定性、收敛、持久化往返、输入校验。"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v3_classifier import (
    MODEL_FORMAT_VERSION, KNNClassifier, LinearSVM,
    MultinomialLogisticRegression, load_model,
)


def make_blobs(n_per_class=25, n_classes=3, D=8, seed=0, spread=0.5):
    """确定性可分类数据：每类一个高斯簇。"""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, D)) * 4.0
    X = np.vstack([centers[c] + spread * rng.standard_normal((n_per_class, D))
                   for c in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X, y


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(seed=1)
        self.model = MultinomialLogisticRegression(l2=1e-2).fit(self.X, self.y)

    def test_fit_predict_accuracy(self):
        pred = self.model.predict(self.X)
        self.assertEqual(pred.shape, (len(self.y),))
        acc = float(np.mean(pred == self.y))
        self.assertGreater(acc, 0.9)
        self.assertEqual(self.model.n_classes_, 3)
        self.assertEqual(self.model.n_features_in_, self.X.shape[1])

    def test_predict_proba(self):
        p = self.model.predict_proba(self.X)
        self.assertEqual(p.shape, (len(self.y), self.model.n_classes_))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(p >= 0.0))
        self.assertTrue(np.all(np.isfinite(p)))

    def test_determinism(self):
        a = MultinomialLogisticRegression(l2=1e-2).fit(self.X, self.y)
        b = MultinomialLogisticRegression(l2=1e-2).fit(self.X, self.y)
        np.testing.assert_array_equal(a.coef_, b.coef_)
        np.testing.assert_array_equal(a.intercept_, b.intercept_)
        np.testing.assert_array_equal(a.predict(self.X), b.predict(self.X))

    def test_convergence_loss_decreases(self):
        h = self.model.loss_history_
        self.assertGreater(len(h), 1)
        self.assertLessEqual(h[-1], h[0])
        self.assertLess(h[-1], 0.2)
        self.assertLess(self.model.final_loss_, 0.2)
        # 严格下降（允许 1e-12 的数值抖动）
        diffs = np.diff(np.asarray(h))
        self.assertTrue(np.all(diffs <= 1e-12))

    def test_l2_shrinks_weights(self):
        weak = MultinomialLogisticRegression(l2=1e-4).fit(self.X, self.y)
        strong = MultinomialLogisticRegression(l2=10.0).fit(self.X, self.y)
        self.assertLess(np.linalg.norm(strong.coef_), np.linalg.norm(weak.coef_))

    def test_standardize_false(self):
        m = MultinomialLogisticRegression(l2=1e-2, standardize=False).fit(self.X, self.y)
        self.assertGreater(float(np.mean(m.predict(self.X) == self.y)), 0.9)
        np.testing.assert_array_equal(m.mu_, np.zeros(self.X.shape[1]))

    def test_binary(self):
        X, y = make_blobs(n_per_class=20, n_classes=2, D=5, seed=2)
        m = MultinomialLogisticRegression(l2=1e-2).fit(X, y)
        self.assertEqual(m.n_classes_, 2)
        self.assertEqual(len(m.classes_), 2)
        self.assertGreater(float(np.mean(m.predict(X) == y)), 0.95)
        p = m.predict_proba(X)
        self.assertEqual(p.shape, (len(y), 2))

    def test_class_weight_balanced(self):
        X, y = make_blobs(n_per_class=5, n_classes=2, D=5, seed=3)
        # 制造不平衡：删除部分类 1 样本
        keep = ~((y == 1) & (np.arange(len(y)) % 2 == 0))
        m = MultinomialLogisticRegression(l2=1e-2,
                                          class_weight='balanced').fit(X[keep], y[keep])
        self.assertTrue(np.all(np.isfinite(m.predict_proba(X))))

    def test_n_params(self):
        self.assertEqual(self.model.n_params,
                         self.model.coef_.size + self.model.intercept_.size)

    def test_feature_config_roundtrip(self):
        spec = {'n_mfcc': 13, 'include_delta': True}
        m = MultinomialLogisticRegression(l2=1e-2).fit(self.X, self.y, feature_config=spec)
        self.assertEqual(m.feature_config, spec)

    def test_errors(self):
        with self.assertRaises(ValueError):
            MultinomialLogisticRegression().fit(self.X, self.y[:3])
        with self.assertRaises(ValueError):
            MultinomialLogisticRegression().fit(self.X, np.zeros(len(self.y)))
        with self.assertRaises(ValueError):
            MultinomialLogisticRegression().fit(np.full_like(self.X, np.nan), self.y)
        with self.assertRaises(RuntimeError):
            MultinomialLogisticRegression().predict(self.X)
        with self.assertRaises(ValueError):
            self.model.predict(self.X[:, :2])


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(seed=4)
        self.model = MultinomialLogisticRegression(l2=1e-2).fit(
            self.X, self.y, feature_config={'demo': True})

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model'
            saved = self.model.save(path)
            self.assertTrue(saved.endswith('.npz'))
            self.assertTrue(Path(saved).exists())
            loaded = MultinomialLogisticRegression.load(saved)
            self.assertEqual(loaded.classes_.tolist(), self.model.classes_.tolist())
            np.testing.assert_array_equal(loaded.coef_, self.model.coef_)
            np.testing.assert_array_equal(loaded.predict(self.X), self.model.predict(self.X))
            self.assertEqual(loaded.feature_config, {'demo': True})

    def test_load_model_dispatch_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self.model.save(Path(tmp) / 'm.npz')
            loaded = load_model(saved)
            self.assertIsInstance(loaded, MultinomialLogisticRegression)
            import json
            with np.load(saved) as z:
                self.assertEqual(int(z['__meta__'].size), 1)
                meta = json.loads(str(z['__meta__']))
            self.assertEqual(meta['version'], MODEL_FORMAT_VERSION)

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            MultinomialLogisticRegression.load(Path('no_such_model_file'))


class TestLinearSVM(unittest.TestCase):
    def test_accuracy_and_determinism(self):
        X, y = make_blobs(seed=5)
        a = LinearSVM(c=1.0, seed=1).fit(X, y)
        b = LinearSVM(c=1.0, seed=1).fit(X, y)
        np.testing.assert_array_equal(a.coef_, b.coef_)
        self.assertGreater(float(np.mean(a.predict(X) == y)), 0.9)
        p = a.predict_proba(X)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)

    def test_save_load(self):
        X, y = make_blobs(seed=6)
        m = LinearSVM(c=1.0, seed=1).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            saved = m.save(Path(tmp) / 'svm')
            loaded = LinearSVM.load(saved)
            np.testing.assert_array_equal(loaded.predict(X), m.predict(X))


class TestKNN(unittest.TestCase):
    def test_accuracy(self):
        X, y = make_blobs(seed=7)
        m = KNNClassifier(k=5).fit(X, y)
        self.assertGreater(float(np.mean(m.predict(X) == y)), 0.9)
        p = m.predict_proba(X)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)

    def test_k_larger_than_train(self):
        X, y = make_blobs(n_per_class=3, n_classes=2, seed=8)
        m = KNNClassifier(k=99).fit(X, y)
        self.assertEqual(m.predict(X).shape, (len(y),))

    def test_save_load(self):
        X, y = make_blobs(seed=9)
        m = KNNClassifier(k=3).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            loaded = KNNClassifier.load(m.save(Path(tmp) / 'knn'))
            np.testing.assert_array_equal(loaded.predict(X), m.predict(X))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)
        with self.assertRaises(ValueError):
            KNNClassifier(weights='bogus')


if __name__ == '__main__':
    unittest.main()

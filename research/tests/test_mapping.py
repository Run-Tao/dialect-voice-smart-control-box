import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mapping import OLSMapping, RidgeMapping, SVDFeatureTransformer


class TestMapping(unittest.TestCase):
    def test_ols_recovers_linear_coefficients(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4, 3))
        Y = 2.5 * X + 3.0
        model = OLSMapping(fit_intercept=True).fit(X, Y)
        self.assertTrue(np.allclose(model.omega_, 2.5, atol=1e-8))
        self.assertTrue(np.allclose(model.epsilon_, 3.0, atol=1e-8))

    def test_intercept_vs_no_intercept(self):
        X = np.arange(10, dtype=float).reshape(10, 1, 1)
        Y = 2.0 * X + 5.0
        with_i = OLSMapping(fit_intercept=True).fit(X, Y)
        without_i = OLSMapping(fit_intercept=False).fit(X, Y)
        self.assertAlmostEqual(with_i.epsilon_[0, 0], 5.0, places=6)
        self.assertAlmostEqual(without_i.epsilon_[0, 0], 0.0, places=6)
        self.assertGreater(abs(with_i.omega_[0, 0] - without_i.omega_[0, 0]), 1e-3)

    def test_ridge_shrinks_with_alpha(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 2, 1))
        Y = 2.0 * X
        small = RidgeMapping(alpha=1e-6).fit(X, Y)
        large = RidgeMapping(alpha=100.0).fit(X, Y)
        self.assertLess(abs(large.omega_[0, 0]), abs(small.omega_[0, 0]))

    def test_transform_shape_and_errors(self):
        X = np.random.randn(10, 5, 2)
        Y = np.random.randn(10, 5, 2)
        model = OLSMapping().fit(X, Y)
        out = model.transform(X[0])
        self.assertEqual(out.shape, X[0].shape)
        with self.assertRaises(ValueError):
            model.transform(np.random.randn(4, 2))
        bad = X.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            OLSMapping().fit(bad, Y)

    def test_constant_variance_raises_ols(self):
        X = np.ones((5, 2, 2))
        Y = np.zeros((5, 2, 2))
        with self.assertRaises(ValueError):
            OLSMapping(fit_intercept=True).fit(X, Y)

    def test_svd_modes_shapes(self):
        seqs = [np.random.randn(20, 5) for _ in range(3)]
        for mode in ['none', 'R0_literal_rank_k', 'R1_global_projection']:
            tr = SVDFeatureTransformer(mode, rank=3).fit(seqs)
            out = tr.transform(seqs[0])
            self.assertEqual(out.ndim, 2)
            self.assertEqual(out.shape[0], 20)


if __name__ == '__main__':
    unittest.main()

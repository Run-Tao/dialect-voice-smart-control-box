import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alignment import dtw_distance_general, dtw_distance_nd, dtw_align_pair


class TestVectorizedDtwEquivalence(unittest.TestCase):
    """dtw_distance_nd 必须与 dtw_distance_general 数值一致。"""

    def test_equivalence_random(self):
        rng = np.random.RandomState(0)
        for _ in range(15):
            n = rng.randint(4, 50)
            m = rng.randint(4, 50)
            a = rng.randn(n, 13).astype(np.float32)
            b = rng.randn(m, 13).astype(np.float32)
            for mf in (None, 20, 40, 60):
                for norm in ('path', 'max_len', 'none'):
                    d1 = dtw_distance_general(a, b, metric='euclidean', norm=norm, max_frames=mf)
                    d2 = dtw_distance_nd(a, b, norm=norm, max_frames=mf)
                    self.assertAlmostEqual(d1, d2, places=9,
                                           msg=f'n={n} m={m} mf={mf} norm={norm}')

    def test_identical_is_zero(self):
        x = np.random.RandomState(1).randn(30, 13)
        self.assertAlmostEqual(dtw_distance_nd(x, x), 0.0, places=9)

    def test_symmetry(self):
        rng = np.random.RandomState(2)
        a = rng.randn(20, 13)
        b = rng.randn(25, 13)
        self.assertAlmostEqual(dtw_distance_nd(a, b), dtw_distance_nd(b, a), places=9)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dtw_distance_nd(np.zeros((5, 13)), np.zeros((5, 10)))

    def test_bad_norm_raises(self):
        with self.assertRaises(ValueError):
            dtw_distance_nd(np.zeros((5, 13)), np.zeros((5, 13)), norm='bogus')

    def test_downsample_respected(self):
        """max_frames 必须同时作用于两端（原实现的双向截断语义）。"""
        long_seq = np.random.RandomState(3).randn(300, 13)
        short_seq = np.random.RandomState(4).randn(60, 13)
        d_limited = dtw_distance_nd(long_seq, short_seq, max_frames=60)
        d_manual = dtw_distance_nd(long_seq[:60] * 0 + long_seq[np.linspace(0, 299, 60).astype(int)],
                                   short_seq, max_frames=None)
        self.assertAlmostEqual(d_limited, d_manual, places=9)


if __name__ == '__main__':
    unittest.main()

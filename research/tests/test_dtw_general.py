import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.alignment import dtw_distance_general


class TestDTWGeneral(unittest.TestCase):
    def test_identical(self):
        x = np.random.randn(20, 13)
        self.assertAlmostEqual(dtw_distance_general(x, x, metric='euclidean'), 0.0, places=6)
        self.assertAlmostEqual(dtw_distance_general(x, x, metric='cosine'), 0.0, places=6)
    def test_norm(self):
        x = np.random.randn(20, 13); y = np.random.randn(25, 13)
        a = dtw_distance_general(x, y, norm='path', max_frames=20)
        b = dtw_distance_general(x, y, norm='max_len', max_frames=20)
        self.assertGreaterEqual(a, 0); self.assertGreaterEqual(b, 0)


if __name__ == '__main__':
    unittest.main()

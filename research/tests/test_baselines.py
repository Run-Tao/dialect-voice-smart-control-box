import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import cosine_similarity_frames, predict_cosine, dtw_distance_normalized, predict_dtw


class TestBaselines(unittest.TestCase):
    def test_cosine_identical(self):
        x = np.random.randn(10, 13).astype(np.float32)
        score = cosine_similarity_frames(x, x)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_cosine_predict_range(self):
        templates = {
            '开灯': np.random.randn(10, 13).astype(np.float32),
            '关灯': np.random.randn(10, 13).astype(np.float32),
        }
        q = templates['开灯']
        p = predict_cosine(q, templates)
        self.assertIn(p.command, templates)

    def test_dtw_basic(self):
        x = np.random.randn(12, 13).astype(np.float32)
        y = np.random.randn(15, 13).astype(np.float32)
        self.assertAlmostEqual(dtw_distance_normalized(x, x), 0.0, places=6)
        self.assertGreaterEqual(dtw_distance_normalized(x, y), 0.0)

    def test_dtw_multi_template(self):
        templates = {
            '开灯': [np.random.randn(10, 13), np.random.randn(12, 13)],
            '关灯': [np.random.randn(11, 13)],
        }
        q = np.random.randn(13, 13)
        p = predict_dtw(q, templates)
        self.assertIn(p.command, templates)


if __name__ == '__main__':
    unittest.main()

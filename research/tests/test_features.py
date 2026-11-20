import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import extract_mfcc
from src.alignment import fixed_length_sequence, dtw_align_pair


class TestFeatures(unittest.TestCase):
    def test_mfcc_shape(self):
        sr = 16000
        t = np.arange(sr) / sr
        audio = np.sin(2 * np.pi * 300 * t).astype(np.float32)
        mfcc = extract_mfcc(audio, sr, n_fft=512, hop_length=160, n_mfcc=13, n_mels=26)
        self.assertEqual(mfcc.ndim, 2)
        self.assertEqual(mfcc.shape[1], 13)
        self.assertGreater(mfcc.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(mfcc)))

    def test_fixed_length(self):
        x = np.random.randn(50, 13).astype(np.float32)
        y = fixed_length_sequence(x, 100)
        self.assertEqual(y.shape, (100, 13))

    def test_dtw_path(self):
        x = np.random.randn(20, 13).astype(np.float32)
        y = np.random.randn(25, 13).astype(np.float32)
        xa, ya, path, dist = dtw_align_pair(x, y)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (x.shape[0] - 1, y.shape[0] - 1))
        self.assertEqual(len(xa), len(ya))
        self.assertGreaterEqual(dist, 0)


if __name__ == '__main__':
    unittest.main()

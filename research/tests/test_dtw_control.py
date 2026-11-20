import tempfile
import unittest
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dtw_model import DTWControlModel


class TestDTWControlModel(unittest.TestCase):
    def test_predict_and_save(self):
        templates = {
            'A': [np.random.randn(10, 5).astype(np.float32)],
            'B': [np.random.randn(10, 5).astype(np.float32)],
        }
        model = DTWControlModel(templates, threshold=0.5, max_frames=10)
        p = model.predict_sequence(templates['A'][0])
        self.assertIn(p.command, {'A', 'B'})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'm.pkl'
            model.save(path)
            loaded = DTWControlModel.load(path)
            self.assertEqual(loaded.threshold, 0.5)
            self.assertEqual(loaded.max_frames, 10)


if __name__ == '__main__':
    unittest.main()

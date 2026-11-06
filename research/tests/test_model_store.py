import tempfile
import unittest
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifiers import StandardScalerNumpy, SoftmaxRegression
from src.model_store import save_bundle, load_bundle


class TestModelStore(unittest.TestCase):
    def test_roundtrip(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array(['A', 'A', 'B', 'B'])
        scaler = StandardScalerNumpy().fit(X)
        model = SoftmaxRegression(epochs=20).fit(scaler.transform(X), y)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'm.pkl'
            save_bundle(p, model, scaler, 0.5, {'k': 'v'})
            b = load_bundle(p)
            self.assertEqual(b['threshold'], 0.5)
            self.assertEqual(b['metadata']['k'], 'v')
            self.assertTrue(hasattr(b['model'], 'predict'))


if __name__ == '__main__':
    unittest.main()

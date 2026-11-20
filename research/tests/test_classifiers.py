import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifiers import StandardScalerNumpy, SoftmaxRegression, LinearSVM


class TestClassifiers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = np.vstack([
            rng.normal(0, 0.3, (20, 4)),
            rng.normal(3, 0.3, (20, 4)),
        ])
        self.y = np.array(['A'] * 20 + ['B'] * 20)
        self.train = np.arange(30)
        self.test = np.arange(30, 40)

    def test_scaler_fit_on_train_only(self):
        scaler = StandardScalerNumpy().fit(self.X[self.train])
        self.assertTrue(np.allclose(scaler.mean_, self.X[self.train].mean(axis=0)))

    def test_logreg_predict_labels(self):
        scaler = StandardScalerNumpy().fit(self.X[self.train])
        Xtr = scaler.transform(self.X[self.train])
        Xte = scaler.transform(self.X[self.test])
        model = SoftmaxRegression(epochs=300).fit(Xtr, self.y[self.train])
        pred = model.predict(Xte)
        self.assertTrue(set(pred).issubset({'A', 'B'}))

    def test_svm_predict_labels(self):
        scaler = StandardScalerNumpy().fit(self.X[self.train])
        Xtr = scaler.transform(self.X[self.train])
        Xte = scaler.transform(self.X[self.test])
        model = LinearSVM(epochs=500).fit(Xtr, self.y[self.train])
        pred = model.predict(Xte)
        self.assertTrue(set(pred).issubset({'A', 'B'}))


if __name__ == '__main__':
    unittest.main()

import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rejection import calibrate_threshold, rejection_metrics


class TestRejection(unittest.TestCase):
    def test_calibration(self):
        conf = [0.9, 0.8, 0.7, 0.4, 0.3]
        correct = [True, True, False, False, False]
        best = calibrate_threshold(conf, correct, min_coverage=0.6)
        self.assertGreaterEqual(best['coverage'], 0.6)
        metrics = rejection_metrics(conf, correct, [0.2, 0.8], best['threshold'])
        self.assertIn('unknown_false_accept_rate', metrics)

    def test_empty_unknown(self):
        metrics = rejection_metrics([0.9], [True], [], 0.5)
        self.assertEqual(metrics['unknown_rejection_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()

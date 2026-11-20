# -*- coding: utf-8 -*-
"""v3 实验脚本中的统计/指标函数单元测试（不跑模型/数据）。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from scripts.run_v3_experiments import (  # noqa: E402
    INTENTS, full_metrics, macro_f1, mcnemar_exact, paired_compare,
)


def fake_recs(n=12):
    recs = []
    for i in range(n):
        recs.append({
            'utterance_id': f'u{i}',
            'expression_group_id': f'g{i // 3}',
            'command': INTENTS[i % 6],
            'condition': 'text_morphology' if i % 2 else 'rate_slow_18',
            'register': 'short_direct' if i % 3 else 'correction',
            'voice': 'v1' if i % 2 else 'v2',
            'text': '',
        })
    return recs


class TestMetrics(unittest.TestCase):
    def test_macro_f1_perfect(self):
        y = np.array([0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(macro_f1(y, y), 1.0)

    def test_full_metrics_shapes(self):
        recs = fake_recs(12)
        y = np.array([INTENTS.index(r['command']) for r in recs])
        pred = y.copy()
        pred[0] = (pred[0] + 1) % 6
        m = full_metrics(recs, y, pred)
        self.assertEqual(m['n'], 12)
        self.assertEqual(len(m['confusion_matrix']), 6)
        self.assertEqual(int(np.sum(m['confusion_matrix'])), 12)
        self.assertLess(m['accuracy'], 1.0)
        self.assertIn('condition', m['stratified_accuracy'])
        self.assertIn('correction', m['stratified_accuracy']['register'])
        self.assertEqual(m['n_expression_groups'], 4)

    def test_mcnemar_exact_known_values(self):
        self.assertAlmostEqual(mcnemar_exact(10, 2), 0.03857421875, places=6)
        self.assertAlmostEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(5, 5), 1.0, places=6)

    def test_paired_compare_detects_majority(self):
        recs = fake_recs(12)
        y = np.array([INTENTS.index(r['command']) for r in recs])
        wrong = y.copy()
        wrong[:6] = (wrong[:6] + 1) % 6
        res = paired_compare(recs, wrong.tolist(), y.tolist(), 'bad', 'good', n_boot=300)
        self.assertEqual(res['good_only_correct'], 6)
        self.assertEqual(res['bad_only_correct'], 0)
        self.assertLess(res['mcnemar_exact_p_two_sided'], 0.05)
        self.assertGreater(res['accuracy_delta_b_minus_a'], 0)
        self.assertIn('ci95', res['group_bootstrap'])


if __name__ == '__main__':
    unittest.main()

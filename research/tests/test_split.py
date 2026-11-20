import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.manifest import ManifestRecord
from src.split import split_by_speaker


def rec(uid, speaker):
    return ManifestRecord(uid, 'x.wav', speaker, 'd', '开灯')


class TestSplit(unittest.TestCase):
    def test_disjoint_and_reproducible(self):
        records = [rec(f'u{i}', f'S{i}') for i in range(6)]
        a = split_by_speaker(records, 0.6, 0.2, 0.2, seed=42)
        b = split_by_speaker(records, 0.6, 0.2, 0.2, seed=42)
        self.assertEqual(a.train_speakers, b.train_speakers)
        self.assertEqual(set(a.train_speakers) & set(a.val_speakers), set())
        self.assertEqual(set(a.train_speakers) & set(a.test_speakers), set())
        self.assertEqual(set(a.val_speakers) & set(a.test_speakers), set())

    def test_too_few_speakers(self):
        records = [rec('u1', 'S1'), rec('u2', 'S2')]
        with self.assertRaises(ValueError):
            split_by_speaker(records)


if __name__ == '__main__':
    unittest.main()

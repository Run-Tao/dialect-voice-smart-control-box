import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.split import split_by_speaker
from src.experiment import run_experiment
from src.results import load_json


class TestExperiment(unittest.TestCase):
    def test_cosine_result_saved(self):
        base = Path(__file__).resolve().parents[1]
        cfg = load_config(base / 'config.json')
        records = load_manifest(base / 'data' / 'smoke_manifest.csv')
        split = split_by_speaker(records, seed=42)
        with tempfile.TemporaryDirectory() as d:
            result, out_dir = run_experiment('cosine', split, cfg, output_root=d)
            self.assertTrue((Path(out_dir) / 'result.json').exists())
            self.assertTrue((Path(out_dir) / 'predictions.csv').exists())
            loaded = load_json(Path(out_dir) / 'result.json')
            self.assertEqual(loaded['method'], 'cosine')
            self.assertTrue(loaded['split']['train_speakers'])
            self.assertTrue(loaded['split']['test_speakers'])
            self.assertFalse(set(loaded['split']['train_speakers']) & set(loaded['split']['test_speakers']))


if __name__ == '__main__':
    unittest.main()

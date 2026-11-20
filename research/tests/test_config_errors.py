import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, validate_config
from src.audio_io import load_wav


class TestConfigAndAudioErrors(unittest.TestCase):
    def test_missing_config(self):
        with self.assertRaises(FileNotFoundError):
            load_config('definitely_missing_config.json')

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            validate_config({'features': {}, 'split': {}, 'alignment': {}, 'outputs': {}})

    def test_missing_audio(self):
        with self.assertRaises(FileNotFoundError):
            load_wav('definitely_missing.wav')


if __name__ == '__main__':
    unittest.main()

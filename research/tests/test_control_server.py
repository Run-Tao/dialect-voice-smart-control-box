import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.control_server import DTWControlServer, MockControlDevice
from src.baselines import Prediction


class DummyDTWModel:
    def __init__(self, threshold):
        self.threshold = threshold
    def predict_sequence(self, seq):
        return Prediction(command='LIGHT_ON', score=0.8, scores={'LIGHT_ON': 0.8}, confidence=0.8)


def make_wav(path):
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(b'\x00\x00' * 1600)


class TestControlServer(unittest.TestCase):
    def test_reject_does_not_change_state(self):
        with tempfile.TemporaryDirectory() as d:
            wav_path = Path(d) / 'a.wav'
            make_wav(wav_path)
            cfg = {'sample_rate': 16000, 'n_fft': 512, 'hop_length': 160, 'n_mfcc': 13,
                   'n_mels': 26, 'fmin': 50, 'fmax': 8000, 'preemphasis': 0.97, 'use_cmvn': True}
            server = DTWControlServer(DummyDTWModel(0.9), cfg)
            result = server.process(wav_path, 1)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['device_response'], 'REJECTED')
            self.assertEqual(result['device_state'], {'light': 'off', 'curtain': 'closed'})

    def test_invalid_audio_does_not_change_state(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {'sample_rate': 16000, 'n_fft': 512, 'hop_length': 160, 'n_mfcc': 13,
                   'n_mels': 26, 'fmin': 50, 'fmax': 8000, 'preemphasis': 0.97, 'use_cmvn': True}
            server = DTWControlServer(DummyDTWModel(0.1), cfg)
            with self.assertRaises(FileNotFoundError):
                server.process(Path(d) / 'missing.wav', 1)
            self.assertEqual(server.device.state, {'light': 'off', 'curtain': 'closed'})

    def test_accept_updates_state(self):
        with tempfile.TemporaryDirectory() as d:
            wav_path = Path(d) / 'a.wav'
            make_wav(wav_path)
            cfg = {'sample_rate': 16000, 'n_fft': 512, 'hop_length': 160, 'n_mfcc': 13,
                   'n_mels': 26, 'fmin': 50, 'fmax': 8000, 'preemphasis': 0.97, 'use_cmvn': True}
            log_path = Path(d) / 'log.jsonl'
            server = DTWControlServer(DummyDTWModel(0.1), cfg, log_path=log_path)
            result = server.process(wav_path, 1)
            self.assertTrue(log_path.exists())
            self.assertEqual(len(log_path.read_text(encoding='utf-8').splitlines()), 1)
            self.assertTrue(result['accepted'])
            self.assertEqual(result['device_response'], 'ACK,1,OK')
            self.assertEqual(result['device_state']['light'], 'on')


if __name__ == '__main__':
    unittest.main()

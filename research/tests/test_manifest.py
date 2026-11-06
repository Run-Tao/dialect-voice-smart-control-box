import csv
import tempfile
import unittest
import wave
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.manifest import load_manifest, ManifestError


def make_wav(path):
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b'\x00\x00' * 1600)


class TestManifest(unittest.TestCase):
    def test_load_ok(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            wav = d / 'a.wav'
            make_wav(wav)
            csv_path = d / 'manifest.csv'
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['utterance_id', 'audio_path', 'speaker_id', 'dialect', 'command'])
                w.writeheader()
                w.writerow({'utterance_id': 'u1', 'audio_path': 'a.wav', 'speaker_id': 'S1', 'dialect': 'd', 'command': '开灯'})
            recs = load_manifest(csv_path)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].command, '开灯')

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            csv_path = d / 'manifest.csv'
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['utterance_id', 'audio_path', 'speaker_id', 'dialect', 'command'])
                w.writeheader()
                w.writerow({'utterance_id': 'u1', 'audio_path': 'missing.wav', 'speaker_id': 'S1', 'dialect': 'd', 'command': '开灯'})
            with self.assertRaises(ManifestError):
                load_manifest(csv_path)

    def test_alias_fields(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            wav = d / 'a.wav'
            make_wav(wav)
            csv_path = d / 'manifest.csv'
            with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['utt_id', 'audio_path', 'speaker_id', 'dialect', 'intent_id', 'text', 'semantic_pair_id'])
                w.writeheader()
                w.writerow({'utt_id': 'u1', 'audio_path': 'a.wav', 'speaker_id': 'S1', 'dialect': 'yue-HK', 'intent_id': 'LIGHT_ON', 'text': '開燈', 'semantic_pair_id': 'LIGHT_ON_direct'})
            recs = load_manifest(csv_path)
            self.assertEqual(recs[0].utterance_id, 'u1')
            self.assertEqual(recs[0].command, 'LIGHT_ON')
            self.assertEqual(recs[0].text, '開燈')


if __name__ == '__main__':
    unittest.main()

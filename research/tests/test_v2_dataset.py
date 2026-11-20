import sys
import tempfile
import csv
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_v2 import (
    assign_expression_groups, normalize_text, jaccard, levenshtein,
    quality_flags_for, row_expression_group, _hash_seed,
)
from src.manifest import ManifestRecord, load_manifest


class TestSplitLeakageFree(unittest.TestCase):
    """表达组划分必须保证同组样本不跨子集。"""

    def _make_groups(self, n_per_cmd=10, registers=('short_direct', 'colloquial')):
        groups = {}
        for cmd in ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN',
                    'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']:
            for reg in registers:
                for i in range(n_per_cmd):
                    gid = f'{cmd}_{reg}_{i:02d}'
                    groups[gid] = {'command': cmd, 'register': reg,
                                   'n_samples': 4, 'voices': {'v1', 'v2'}, 'text': gid}
        return groups

    def test_group_assignment_is_function(self):
        groups = self._make_groups()
        a = assign_expression_groups(groups, seed=42)
        b = assign_expression_groups(groups, seed=42)
        self.assertEqual(a, b, '同一 seed 必须得到同一划分')

    def test_every_group_assigned_once(self):
        groups = self._make_groups()
        a = assign_expression_groups(groups, seed=42)
        self.assertEqual(set(a), set(groups))
        self.assertTrue(all(v in ('train', 'val', 'test') for v in a.values()))

    def test_all_commands_present_in_each_subset(self):
        groups = self._make_groups(n_per_cmd=12)
        a = assign_expression_groups(groups, seed=7)
        for subset in ('train', 'val', 'test'):
            cmds = {groups[g]['command'] for g, s in a.items() if s == subset}
            self.assertEqual(len(cmds), 6, f'{subset} 缺少命令类别: {cmds}')

    def test_different_seeds_differ(self):
        groups = self._make_groups(n_per_cmd=12)
        a = assign_expression_groups(groups, seed=1)
        b = assign_expression_groups(groups, seed=2)
        self.assertNotEqual(a, b)

    def test_ratio_approximately_respected(self):
        groups = self._make_groups(n_per_cmd=12)
        a = assign_expression_groups(groups, seed=42, train_ratio=0.7, val_ratio=0.15)
        n = len(a)
        frac_train = sum(1 for v in a.values() if v == 'train') / n
        self.assertGreater(frac_train, 0.55)
        self.assertLess(frac_train, 0.85)


class TestTextNormalization(unittest.TestCase):
    def test_punctuation_removed(self):
        self.assertEqual(normalize_text('開燈，好嗎？'), normalize_text('開燈好嗎'))

    def test_traditional_simplified_unified(self):
        # 繁體「設備狀態」與簡體「设备状态」规范化后必须一致。
        # 码位：繁 設U+8A2D 備U+5099 態U+614B 狀U+72C0
        #       简 设U+8BBE 备U+5907 态U+6001 状U+72B6
        trad = ''.join(chr(c) for c in (0x8a2d, 0x5099, 0x614b, 0x72c0))
        simp = ''.join(chr(c) for c in (0x8bbe, 0x5907, 0x6001, 0x72b6))
        self.assertEqual(normalize_text(trad), normalize_text(simp))

    def test_jaccard_identical(self):
        self.assertAlmostEqual(jaccard('開燈', '開燈'), 1.0)

    def test_levenshtein_basic(self):
        self.assertEqual(levenshtein('開燈', '關燈'), 1)
        self.assertEqual(levenshtein('abc', 'abc'), 0)

    def test_group_fallback_is_stable(self):
        r = {'command': 'LIGHT_ON', 'text': '開燈。'}
        r2 = {'command': 'LIGHT_ON', 'text': '開燈'}
        self.assertEqual(row_expression_group(r), row_expression_group(r2))


class TestQualityFlags(unittest.TestCase):
    def _stats(self, **kw):
        base = {'sample_rate': 16000, 'channels': 1, 'sampwidth': 2,
                'duration': 2.0, 'rms': 0.05, 'peak': 0.8, 'silence_ratio': 0.2,
                'clipping_ratio': 0.0, 'speech_frame_ratio': 0.7, 'samples': 32000}
        base.update(kw)
        return base

    def test_normal_is_ok(self):
        self.assertEqual(quality_flags_for(self._stats())[0], 'ok')

    def test_empty_excluded(self):
        self.assertEqual(quality_flags_for(self._stats(samples=0, duration=0.0))[0],
                         'excluded')

    def test_wrong_sample_rate_excluded(self):
        self.assertEqual(quality_flags_for(self._stats(sample_rate=48000))[0], 'excluded')

    def test_stereo_excluded(self):
        self.assertEqual(quality_flags_for(self._stats(channels=2))[0], 'excluded')

    def test_very_short_excluded(self):
        self.assertEqual(quality_flags_for(self._stats(duration=0.2))[0], 'excluded')

    def test_near_silent_excluded(self):
        self.assertEqual(quality_flags_for(self._stats(rms=0.0001))[0], 'excluded')

    def test_clipping_flagged_not_excluded(self):
        status, reasons = quality_flags_for(self._stats(clipping_ratio=0.05))
        self.assertEqual(status, 'flagged')
        self.assertIn('clipping', reasons)


class TestManifestBackwardCompatibility(unittest.TestCase):
    def test_old_manifest_without_new_fields_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            import wave
            wav = d / 'a.wav'
            with wave.open(str(wav), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'\x00\x00' * 1600)
            p = d / 'manifest.csv'
            with open(p, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['utterance_id', 'audio_path',
                                                  'speaker_id', 'dialect', 'command'])
                w.writeheader()
                w.writerow({'utterance_id': 'u1', 'audio_path': 'a.wav',
                            'speaker_id': 'S1', 'dialect': 'yue-HK', 'command': 'LIGHT_ON'})
            recs = load_manifest(p)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].expression_group_id, '')
            self.assertEqual(recs[0].quality_status, '')
            self.assertEqual(recs[0].sha256, '')

    def test_new_fields_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            import wave
            wav = d / 'a.wav'
            with wave.open(str(wav), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b'\x00\x00' * 1600)
            p = d / 'manifest.csv'
            fields = ['utterance_id', 'audio_path', 'speaker_id', 'dialect', 'command',
                      'expression_group_id', 'voice_identity', 'rate_percent',
                      'quality_status', 'sha256', 'condition']
            with open(p, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({'utterance_id': 'u1', 'audio_path': 'a.wav', 'speaker_id': 'S1',
                            'dialect': 'yue-HK', 'command': 'LIGHT_ON',
                            'expression_group_id': 'LIGHT_ON_01',
                            'voice_identity': 'edge-zh-HK-HiuGaaiNeural',
                            'rate_percent': '-12', 'quality_status': 'ok',
                            'sha256': 'abc', 'condition': 'rate_slow'})
            r = load_manifest(p)[0]
            self.assertEqual(r.expression_group_id, 'LIGHT_ON_01')
            self.assertEqual(r.voice_identity, 'edge-zh-HK-HiuGaaiNeural')
            self.assertEqual(r.rate_percent, '-12')
            self.assertEqual(r.condition, 'rate_slow')


if __name__ == '__main__':
    unittest.main()

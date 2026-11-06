# -*- coding: utf-8 -*-
"""第三阶段声学前端测试：维度 / 数值范围 / 确定性 / 与旧 extract_mfcc 等价。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import extract_mfcc
from src.features_v3 import (
    DEFAULT_STATS, FeatureSpec, delta_features, extract_sequence,
    fixed_length_embedding, output_dim, pitch_contour, stats_embedding,
)


def sine(freq=220.0, sr=16000, dur=1.0, amp=0.3):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noisy(sr=16000, dur=0.8, seed=0):
    rng = np.random.default_rng(seed)
    return (0.05 * rng.standard_normal(int(sr * dur))).astype(np.float32)


class TestSpecAndDimension(unittest.TestCase):
    def test_default_spec_hashable_and_roundtrip(self):
        a, b = FeatureSpec(), FeatureSpec()
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))          # frozen dataclass 可哈希
        d = a.to_dict()
        self.assertEqual(FeatureSpec.from_dict(d), a)
        # 未知字段忽略（向前兼容）
        d2 = dict(d)
        d2['unknown_future_field'] = 123
        self.assertEqual(FeatureSpec.from_dict(d2), a)

    def test_to_dict_json_serializable(self):
        import json
        spec = FeatureSpec(fmax=7800.0, include_pitch=True)
        text = json.dumps(spec.to_dict())
        self.assertEqual(FeatureSpec.from_dict(json.loads(text)), spec)

    def test_output_dim_matches_extract(self):
        audio = noisy(dur=0.5)
        specs = [
            FeatureSpec(),
            FeatureSpec(include_delta=True, include_delta2=True),
            FeatureSpec(include_delta=True, include_delta2=True, include_pitch=True),
            FeatureSpec(include_delta=True, include_pitch=True, include_energy=True),
            FeatureSpec(n_mels=40, include_mfcc=False, include_logmel=True,
                        include_delta=True, include_delta2=True),
            FeatureSpec(include_delta2=True, include_pitch=True,
                        include_energy=True, include_logmel=True),
        ]
        for spec in specs:
            seq = extract_sequence(audio, 16000, spec)
            self.assertEqual(seq.shape[1], output_dim(spec),
                             msg=f'维度不一致 spec={spec.to_dict()}')
            self.assertEqual(seq.dtype, np.float32)

    def test_invalid_spec(self):
        with self.assertRaises(ValueError):
            FeatureSpec(n_mfcc=30, n_mels=26)
        with self.assertRaises(ValueError):
            FeatureSpec(include_mfcc=False, include_logmel=False,
                        include_pitch=False, include_energy=False)
        with self.assertRaises(ValueError):
            FeatureSpec(fmin=100.0, fmax=50.0)
        with self.assertRaises(ValueError):
            FeatureSpec(f0_min=500.0, f0_max=100.0)


class TestExtractSequence(unittest.TestCase):
    def test_old_mfcc_equivalence(self):
        """include_delta=False 时必须与二期 extract_mfcc 数值等价。"""
        for audio in (sine(), noisy(), sine(140.0, dur=0.6)):
            old = extract_mfcc(audio, 16000)
            new = extract_sequence(audio, 16000, FeatureSpec())
            self.assertEqual(old.shape, new.shape)
            np.testing.assert_allclose(new, old, rtol=0, atol=1e-6)

    def test_equivalence_with_explicit_legacy_kwargs(self):
        audio = noisy(dur=0.7)
        old = extract_mfcc(audio, 16000, n_fft=512, hop_length=160, n_mfcc=13,
                           n_mels=26, fmin=50.0, fmax=8000.0, preemphasis=0.97,
                           use_cmvn=True)
        new = extract_sequence(audio, 16000,
                               FeatureSpec(fmax=8000.0, include_delta=False))
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-6)

    def test_finite_and_bounded(self):
        audio = noisy(dur=0.8)
        spec = FeatureSpec(include_delta=True, include_delta2=True,
                           include_pitch=True, include_energy=True)
        seq = extract_sequence(audio, 16000, spec)
        self.assertTrue(np.all(np.isfinite(seq)))
        self.assertGreater(seq.shape[0], 1)
        # 各通道量级在合理范围（不做硬约束，只防爆炸）
        self.assertLess(float(np.max(np.abs(seq))), 100.0)

    def test_determinism(self):
        audio = noisy(dur=0.7)
        spec = FeatureSpec(include_delta=True, include_pitch=True,
                           include_energy=True)
        a = extract_sequence(audio, 16000, spec)
        b = extract_sequence(audio, 16000, spec)
        np.testing.assert_array_equal(a, b)

    def test_short_audio_padded_like_old(self):
        audio = np.zeros(100, dtype=np.float32)
        seq = extract_sequence(audio, 16000, FeatureSpec())
        self.assertEqual(seq.shape[0], 1)       # 不足一帧 -> 补零到 1 帧
        self.assertTrue(np.all(np.isfinite(seq)))
        old = extract_mfcc(audio, 16000)
        np.testing.assert_allclose(seq, old, rtol=0, atol=1e-6)

    def test_sr_mismatch_and_empty(self):
        with self.assertRaises(ValueError):
            extract_sequence(noisy(), 8000, FeatureSpec())
        with self.assertRaises(ValueError):
            extract_sequence(np.zeros(0, dtype=np.float32), 16000)

    def test_spec_none_defaults(self):
        audio = noisy(dur=0.4)
        a = extract_sequence(audio, 16000, None)
        b = extract_sequence(audio, 16000, FeatureSpec())
        np.testing.assert_array_equal(a, b)


class TestPitch(unittest.TestCase):
    def test_sine_pitch_estimated(self):
        f0_log, voiced = pitch_contour(sine(200.0, dur=1.0), 16000)
        self.assertEqual(f0_log.shape, voiced.shape)
        self.assertGreater(float(np.mean(voiced)), 0.5)
        est = f0_log[voiced > 0]
        # log(200/100) ≈ 0.693
        self.assertAlmostEqual(float(np.median(est)), np.log(2.0), delta=0.06)

    def test_silence_unvoiced_zero(self):
        f0_log, voiced = pitch_contour(np.zeros(16000, dtype=np.float32), 16000)
        self.assertEqual(float(np.sum(voiced)), 0.0)
        self.assertTrue(np.all(f0_log == 0.0))

    def test_pitch_range_60_400(self):
        for f in (80.0, 350.0):
            f0_log, voiced = pitch_contour(sine(f, dur=1.0), 16000)
            if np.sum(voiced) > 0:
                hz = 100.0 * np.exp(np.median(f0_log[voiced > 0]))
                self.assertGreaterEqual(hz, 55.0)
                self.assertLessEqual(hz, 420.0)


class TestDelta(unittest.TestCase):
    def test_delta_shape_and_constant(self):
        seq = np.ones((50, 4), dtype=np.float32) * 3.0
        d = delta_features(seq, width=2)
        self.assertEqual(d.shape, seq.shape)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_delta_linear_ramp(self):
        T, D = 20, 2
        seq = np.arange(T, dtype=np.float64)[:, None] * np.array([1.0, 2.0])[None, :]
        d = delta_features(seq, width=1)
        # 一阶差分（width=1）对线性斜坡为斜率；首尾用 edge padding，故只查内部行
        np.testing.assert_allclose(d[1:-1], np.tile([1.0, 2.0], (T - 2, 1)), atol=1e-9)

    def test_delta_invalid(self):
        with self.assertRaises(ValueError):
            delta_features(np.zeros(5), 2)
        with self.assertRaises(ValueError):
            delta_features(np.zeros((0, 3)), 2)


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.seq = extract_sequence(
            noisy(dur=1.0), 16000,
            FeatureSpec(include_delta=True, include_pitch=True, include_energy=True))

    def test_stats_embedding(self):
        vec = stats_embedding(self.seq)
        self.assertEqual(vec.shape, (len(DEFAULT_STATS) * self.seq.shape[1],))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vec)))
        # 与手工拼接一致
        d = self.seq.shape[1]
        np.testing.assert_allclose(vec[:d], self.seq.mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(vec[d:2 * d], self.seq.std(axis=0), atol=1e-6)

    def test_stats_subset(self):
        vec = stats_embedding(self.seq, stats=('mean', 'std'))
        self.assertEqual(vec.shape, (2 * self.seq.shape[1],))
        with self.assertRaises(ValueError):
            stats_embedding(self.seq, stats=('mean', 'median'))

    def test_fixed_length_embedding(self):
        vec = fixed_length_embedding(self.seq, n_frames=40)
        self.assertEqual(vec.shape, (40 * self.seq.shape[1],))
        self.assertTrue(np.all(np.isfinite(vec)))
        for n in (1, 7, 40, 100):
            v = fixed_length_embedding(self.seq, n_frames=n)
            self.assertEqual(v.shape, (n * self.seq.shape[1],))
        with self.assertRaises(ValueError):
            fixed_length_embedding(self.seq, n_frames=0)

    def test_embeddings_preserve_frame_when_T_equals_n(self):
        seq = np.random.default_rng(0).standard_normal((40, 3)).astype(np.float32)
        v = fixed_length_embedding(seq, 40)
        np.testing.assert_array_equal(v, seq.reshape(-1))

    def test_trim(self):
        full = stats_embedding(self.seq)
        trim = stats_embedding(self.seq, trim=(0.0, 0.5))
        self.assertEqual(full.shape, trim.shape)
        self.assertFalse(np.allclose(full, trim))
        with self.assertRaises(ValueError):
            stats_embedding(self.seq, trim=(0.6, 0.2))

    def test_invalid_sequence(self):
        with self.assertRaises(ValueError):
            stats_embedding(np.zeros(5))
        with self.assertRaises(ValueError):
            fixed_length_embedding(np.zeros((0, 3)))


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""v3 模型工件 / 纯 numpy 推理的单元测试。

不依赖真实训练产物：用随机参数构造一个小工件，校验加载、前向、
schema、温度、阈值文件、TTA、确定性，以及与 PyTorch 训练模型前向的等价性
（torch 不可用时跳过）。
"""
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features_v3 import FeatureSpec  # noqa: E402
from src.v3_cnn_numpy import (  # noqa: E402
    ARCH_NAME, N_FRAMES, collect_arrays, embed, logits, prepare_input,
    resample_to_frames,
)
from src.v3_model import V3Predictor, V3Prediction  # noqa: E402

CLASSES = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']


def random_params(seed=0):
    rng = np.random.RandomState(seed)
    return {
        'conv0.weight': rng.normal(0, 0.1, (64, 80, 5)).astype(np.float32),
        'conv0.bias': rng.normal(0, 0.1, 64).astype(np.float32),
        'conv0.bn.weight': np.ones(64, np.float32),
        'conv0.bn.bias': np.zeros(64, np.float32),
        'conv0.bn.running_mean': np.zeros(64, np.float32),
        'conv0.bn.running_var': np.ones(64, np.float32),
        'conv1.weight': rng.normal(0, 0.1, (128, 64, 5)).astype(np.float32),
        'conv1.bias': rng.normal(0, 0.1, 128).astype(np.float32),
        'conv1.bn.weight': np.ones(128, np.float32),
        'conv1.bn.bias': np.zeros(128, np.float32),
        'conv1.bn.running_mean': np.zeros(128, np.float32),
        'conv1.bn.running_var': np.ones(128, np.float32),
        'conv2.weight': rng.normal(0, 0.1, (128, 128, 3)).astype(np.float32),
        'conv2.bias': rng.normal(0, 0.1, 128).astype(np.float32),
        'conv2.bn.weight': np.ones(128, np.float32),
        'conv2.bn.bias': np.zeros(128, np.float32),
        'conv2.bn.running_mean': np.zeros(128, np.float32),
        'conv2.bn.running_var': np.ones(128, np.float32),
        'fc.weight': rng.normal(0, 0.1, (6, 256)).astype(np.float32),
        'fc.bias': rng.normal(0, 0.1, 6).astype(np.float32),
    }


def spec():
    return FeatureSpec(include_mfcc=False, include_logmel=True, include_delta=True,
                       include_delta2=False, n_mels=40, use_cmvn=True, logmel_cmvn=True)


def save_artifact(path, temperature=2.0, tta_shifts=None):
    meta = {
        'format': 'v3_cnn_ensemble/1.0',
        'arch': ARCH_NAME,
        'classes': CLASSES,
        'seeds': [42],
        'n_seeds': 1,
        'feature_spec': spec().to_dict(),
        'n_frames': N_FRAMES,
        'in_dim': 80,
        'temperature': temperature,
        'val': {'accuracy': 0.9, 'macro_f1': 0.9, 'n': 10},
        'data': {'manifest_sha256': 'x' * 64, 'split_sha256': 'y' * 64},
        'training': {'max_epochs': 1},
        'created_at': '2026-01-01T00:00:00',
    }
    if tta_shifts:
        meta['tta_shifts'] = list(tta_shifts)
    payload = {}
    for k, v in random_params().items():
        payload[f'cnn_s0/{k}'] = v
    payload['__meta__'] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **payload)
    return Path(path)


def make_wav(path, seconds=1.0, seed=0):
    rng = np.random.RandomState(seed)
    audio = (rng.normal(0, 0.1, int(16000 * seconds)) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())


class TestNumpyForward(unittest.TestCase):
    def test_resample_shape(self):
        seq = np.random.RandomState(0).normal(size=(137, 80)).astype(np.float32)
        out = resample_to_frames(seq, 256)
        self.assertEqual(out.shape, (256, 80))
        self.assertEqual(out.dtype, np.float32)
        out2 = resample_to_frames(seq, 137)
        self.assertTrue(np.array_equal(out2, seq))

    def test_prepare_input(self):
        seq = np.zeros((100, 80), dtype=np.float32)
        x = prepare_input(seq)
        self.assertEqual(x.shape, (1, 80, N_FRAMES))

    def test_forward_shapes(self):
        params = {k: v.astype(np.float64) for k, v in random_params().items()}
        x = np.random.RandomState(0).normal(size=(2, 80, N_FRAMES))
        z = logits(params, x)
        self.assertEqual(z.shape, (2, 6))
        e = embed(params, x)
        self.assertEqual(e.shape, (2, 256))
        self.assertTrue(np.all(np.isfinite(z)))

    def test_forward_matches_torch(self):
        try:
            import torch
        except Exception:
            self.skipTest('torch 不可用')
        import torch.nn as nn

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from train_v3_model import CmdCNN  # noqa: E402

        torch.manual_seed(0)
        m = CmdCNN.build(80, 6, 0.3)
        m.eval()
        params = collect_arrays({k: v.detach().numpy() for k, v in m.state_dict().items()})
        x = np.random.RandomState(1).normal(size=(2, 80, N_FRAMES)).astype(np.float32)
        with torch.no_grad():
            zt = m(torch.from_numpy(x)).numpy()
        zn = logits(params, x.astype(np.float64))
        self.assertLess(float(np.max(np.abs(zt - zn))), 1e-4)


class TestV3Predictor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_and_predict_schema(self):
        model_path = save_artifact(self.dir / 'm.npz')
        model = V3Predictor.load(model_path)
        seq = np.random.RandomState(0).normal(size=(150, 80)).astype(np.float32)
        pred = model.predict_features(seq)
        self.assertIsInstance(pred, V3Prediction)
        self.assertIn(pred.command, CLASSES)
        self.assertAlmostEqual(sum(pred.scores.values()), 1.0, places=6)
        self.assertAlmostEqual(pred.confidence, pred.scores[pred.command], places=6)
        self.assertGreaterEqual(pred.margin, 0.0)
        self.assertGreaterEqual(pred.entropy, 0.0)

    def test_predict_wav_file(self):
        model_path = save_artifact(self.dir / 'm.npz')
        wav = self.dir / 'a.wav'
        make_wav(wav)
        model = V3Predictor.load(model_path)
        pred = model.predict(wav)
        self.assertIn(pred.command, CLASSES)
        self.assertIsNotNone(pred.accepted)

    def test_determinism(self):
        model_path = save_artifact(self.dir / 'm.npz')
        model = V3Predictor.load(model_path)
        seq = np.random.RandomState(3).normal(size=(200, 80)).astype(np.float32)
        a = model.predict_features(seq)
        b = model.predict_features(seq)
        self.assertEqual(a.command, b.command)
        self.assertAlmostEqual(a.confidence, b.confidence, places=12)

    def test_threshold_file_loaded(self):
        model_path = save_artifact(self.dir / 'm.npz')
        (self.dir / 'v3_thresholds.json').write_text(json.dumps({
            'selected_threshold': 0.42, 'score': 'max_prob'}), encoding='utf-8')
        model = V3Predictor.load(model_path)
        self.assertAlmostEqual(model.threshold, 0.42)
        pred = model.predict_features(np.zeros((120, 80), dtype=np.float32))
        # 均匀/极低置信度时，阈值 0.42 应拒识（随机权重输出往往接近均匀）
        self.assertFalse(model.is_accepted(pred))

    def test_tta_changes_shape_and_is_deterministic(self):
        model_path = save_artifact(self.dir / 'm.npz', tta_shifts=[-4, 0, 4])
        model = V3Predictor.load(model_path)
        seq = np.random.RandomState(4).normal(size=(160, 80)).astype(np.float32)
        a = model.predict_features(seq)
        b = model.predict_features(seq)
        self.assertEqual(a.command, b.command)
        self.assertAlmostEqual(a.confidence, b.confidence, places=12)
        self.assertEqual(model.info()['tta_shifts'], [-4, 0, 4])

    def test_info_and_bad_format(self):
        model_path = save_artifact(self.dir / 'm.npz')
        model = V3Predictor.load(model_path)
        info = model.info()
        self.assertEqual(info['n_seeds'], 1)
        self.assertEqual(info['temperature'], 2.0)
        bad = self.dir / 'bad.npz'
        np.savez_compressed(bad, __meta__=np.array(json.dumps({'format': 'x'})))
        with self.assertRaises(ValueError):
            V3Predictor.load(bad)

    def test_embed_features(self):
        model_path = save_artifact(self.dir / 'm.npz')
        model = V3Predictor.load(model_path)
        seq = np.random.RandomState(5).normal(size=(90, 80)).astype(np.float32)
        e = model.embed_features(seq)
        self.assertEqual(e.shape, (256,))
        self.assertTrue(np.all(np.isfinite(e)))


if __name__ == '__main__':
    unittest.main()

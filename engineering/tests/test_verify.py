# -*- coding: utf-8 -*-
"""Pass-2 复核测试：偏置可逆性 + 一致性判定 + 门限口径隔离。

最关键的一条是 ``test_acoustic_probs_match_unbiased_forward``：
门限工件是在**无偏**后验上标定的，所以 ``verify`` 解析还原出来的
``acoustic_probs`` 必须与「不施加偏置再前向一次」的结果一致。
偏置一旦渗进门限，拒识口径就换了，已公布的 FAR 不再成立。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.verify import (CONSISTENT, INCONSISTENT, NO_CANDIDATE,
                             Pass2Verifier, unbias)

MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'
THRESHOLDS = ROOT / 'runtime' / 'models' / 'v3_thresholds_vad.json'

needs_model = pytest.mark.skipif(not MODEL.exists(),
                                  reason='runtime 模型未同步，先运行 scripts/sync_runtime.py')


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------
# 纯数学：偏置可逆
# ----------------------------------------------------------------------
def test_unbias_is_exact_inverse():
    """softmax(log p − b·e_c) 必须精确还原加偏置之前的后验。"""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(6)
    for bias in (0.1, 0.3, 0.6, 1.0):
        biased = _softmax(logits + bias * np.eye(6)[2])
        back = unbias(biased, bias, 2)
        assert np.allclose(back, _softmax(logits), atol=1e-10)


def test_unbias_is_identity_without_bias():
    p = np.array([0.1, 0.2, 0.7])
    assert np.allclose(unbias(p, 0.0, 1), p / p.sum())
    assert np.allclose(unbias(p, 0.3, -1), p / p.sum())


def test_unbias_lowers_the_biased_class():
    p = np.array([0.1, 0.2, 0.7])
    back = unbias(p, 0.6, 2)
    assert back[2] < p[2]
    assert abs(back.sum() - 1.0) < 1e-12


# ----------------------------------------------------------------------
# 与真实模型对接
# ----------------------------------------------------------------------
@needs_model
def test_acoustic_probs_match_unbiased_forward():
    """无偏后验 == 不带偏置再前向一次的结果（门限口径不被污染）。"""
    from runtime.features_v3 import extract_sequence
    from runtime.v3_model import V3Predictor

    predictor = V3Predictor.load(str(MODEL), str(THRESHOLDS))
    rng = np.random.default_rng(1)
    t = np.arange(16000) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.01 * rng.standard_normal(16000))

    verifier = Pass2Verifier(predictor, bias=0.3)
    candidate = predictor.classes_[1]
    res = verifier.verify(audio, 16000, candidate=candidate)

    seq = extract_sequence(audio.astype(np.float32), 16000, predictor.feature_spec)
    pred, _ = predictor.predict_detailed(seq)
    plain = np.asarray([pred.scores[c] for c in predictor.classes_],
                        dtype=np.float64)
    plain = plain / plain.sum()

    assert res.biased
    assert np.allclose(res.acoustic_probs, plain, atol=1e-8)
    # 带偏置的后验必须与无偏的不同，否则这条测试没有区分力。
    assert not np.allclose(res.probs, plain, atol=1e-6)


@needs_model
def test_consistency_flags():
    """一致性三态：一致 / 不一致 / 无候选。"""
    from runtime.v3_model import V3Predictor

    predictor = V3Predictor.load(str(MODEL), str(THRESHOLDS))
    rng = np.random.default_rng(2)
    t = np.arange(16000) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.01 * rng.standard_normal(16000))

    verifier = Pass2Verifier(predictor, bias=0.3)
    res = verifier.verify(audio, 16000, candidate=None)
    assert res.consistency == NO_CANDIDATE
    assert not res.biased

    res2 = verifier.verify(audio, 16000, candidate=res.top1)
    assert res2.consistency == CONSISTENT

    other = next(c for c in predictor.classes_ if c != res2.top1)
    res3 = verifier.verify(audio, 16000, candidate=other)
    assert res3.consistency == INCONSISTENT


@needs_model
def test_verify_rejects_empty_audio():
    from runtime.v3_model import V3Predictor

    predictor = V3Predictor.load(str(MODEL), str(THRESHOLDS))
    verifier = Pass2Verifier(predictor, bias=0.3)
    with pytest.raises(ValueError):
        verifier.verify(np.zeros(0, dtype=np.float32), 16000, candidate='LIGHT_ON')

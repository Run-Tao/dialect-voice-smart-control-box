# -*- coding: utf-8 -*-
"""无映射余弦匹配和DTW模板匹配。"""
from dataclasses import dataclass
import numpy as np
from .alignment import dtw_align_pair


@dataclass
class Prediction:
    command: str
    score: float
    scores: dict
    distance: float = None
    confidence: float = None


def cosine_similarity_frames(query, template):
    q = np.asarray(query, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    if q.shape != t.shape:
        raise ValueError(f"余弦匹配要求同形: {q.shape} vs {t.shape}")
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    tn = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(qn * tn, axis=1)))


def predict_cosine(query_fixed, templates):
    scores = {cmd: cosine_similarity_frames(query_fixed, tpl) for cmd, tpl in templates.items()}
    cmd = max(scores, key=scores.get)
    return Prediction(command=cmd, score=float(scores[cmd]), scores=scores)


def _downsample(x, max_frames):
    x = __import__('numpy').asarray(x)
    if max_frames is None or x.shape[0] <= max_frames:
        return x
    idx = __import__('numpy').linspace(0, x.shape[0]-1, int(max_frames)).astype(int)
    return x[idx]


def dtw_distance_normalized(x, y, normalize='path', max_frames=None):
    x = _downsample(x, max_frames)
    y = _downsample(y, max_frames)
    _, _, path, dist = dtw_align_pair(x, y)
    if normalize == 'path':
        denom = max(len(path), 1)
    elif normalize == 'max_len':
        denom = max(x.shape[0], y.shape[0])
    elif normalize == 'none':
        denom = 1.0
    else:
        raise ValueError("normalize必须是path/max_len/none")
    return float(dist / denom)


def predict_dtw(query_seq, templates, normalize='path', temperature=0.2, max_frames=None):
    distances = {}
    for cmd, tpls in templates.items():
        distances[cmd] = min(dtw_distance_normalized(query_seq, t, normalize, max_frames=max_frames) for t in tpls)
    best = min(distances, key=distances.get)
    vals = np.array(list(distances.values()), dtype=np.float64)
    vals = vals - np.min(vals)
    prob = np.exp(-vals / max(temperature, 1e-6))
    prob = prob / np.sum(prob)
    scores = {cmd: float(p) for cmd, p in zip(distances.keys(), prob)}
    return Prediction(
        command=best,
        score=float(scores[best]),
        scores=scores,
        distance=float(distances[best]),
        confidence=float(scores[best]),
    )

# -*- coding: utf-8 -*-
"""方言智控盒 v3 运行时推理模型（纯 numpy，不依赖 PyTorch）。

模型工件由 ``scripts/train_v3_model.py`` 生成：
``data/models/v3_cnn_ensemble.npz``（多种子 1D-CNN + JSON meta），
同目录可选的 ``v3_thresholds.json`` 由 ``scripts/evaluate_v3_rejection.py``
在**验证集 + 校准未知集**上选择阈值后写入；本模块只负责读取，不在推理时
做任何参数选择。

用法::

    from .v3_model import V3Predictor
    m = V3Predictor.load('data/models/v3_cnn_ensemble.npz')
    p = m.predict('data/tts_corpus/wav16k_v2/edge/LIGHT_ON_01_....wav')
    p.command, p.confidence, p.scores

设计要点
--------
1. 特征：``src.features_v3.FeatureSpec``（log-mel 40 + Δ，整句 CMVN），
   线性插值到 256 帧；与训练脚本使用同一实现，避免训练/推理不一致。
2. 多种子概率平均；温度标定只影响置信度数值（不改变 argmax），
   用于拒识阈值与前端展示。
3. 支持测试时时间平移增广（TTA），默认关闭；开启时对固定长度特征做
   ±shift 帧循环平移后平均概率（工件 meta 中的 ``tta_shifts`` 控制）。
4. 阈值：读取同目录 ``v3_thresholds.json`` 的 ``selected_threshold``；
   文件不存在时 ``threshold=0.0``（全部接受），演示脚本可显式覆盖。
5. 纯 numpy 前向，单条 CPU 推理为毫秒级（不含特征提取），便于离线部署。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .features_v3 import FeatureSpec, extract_sequence
from .v3_cnn_numpy import (
    IN_DIM, N_FRAMES, collect_arrays, embed as cnn_embed, logits as cnn_logits,
    prepare_input, softmax_np,
)

MODEL_FORMAT = 'v3_cnn_ensemble/1.0'
THRESHOLDS_FILENAME = 'v3_thresholds.json'


@dataclass
class V3Prediction:
    """与 ``src.baselines.Prediction`` / 控制服务兼容的预测结果。"""

    command: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    distance: Optional[float] = None
    margin: float = 0.0
    entropy: float = 0.0
    accepted: Optional[bool] = None

    def as_dict(self) -> dict:
        return {
            'command': self.command,
            'confidence': float(self.confidence),
            'scores': {str(k): float(v) for k, v in self.scores.items()},
            'distance': self.distance,
            'margin': float(self.margin),
            'entropy': float(self.entropy),
            'accepted': self.accepted,
        }


class V3Predictor:
    """多种子 1D-CNN 集成预测器（纯 numpy 推理）。"""

    def __init__(self, meta: dict, seed_params: List[Dict[str, np.ndarray]],
                 thresholds: Optional[dict] = None):
        self.meta = dict(meta)
        self.classes_ = list(meta['classes'])
        self.n_frames = int(meta.get('n_frames', N_FRAMES))
        self.feature_spec = FeatureSpec.from_dict(meta['feature_spec'])
        self.temperature = float(meta.get('temperature', 1.0))
        self._seed_params = seed_params
        self.tta_shifts = list(meta.get('tta_shifts', []))
        self.thresholds = dict(thresholds or {})
        self.threshold = float(self.thresholds.get('selected_threshold', 0.0))
        self.model_name = f"v3_cnn_ensemble({len(seed_params)}seed,{len(self.classes_)}class)"
        self.model_path: Optional[Path] = None

    # ------------------------------------------------------------- 加载
    @classmethod
    def load(cls, path, thresholds_path=None) -> 'V3Predictor':
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'v3 模型文件不存在: {path}')
        if path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as z:
                if '__meta__' not in z.files:
                    raise ValueError(f'不是 v3 模型工件（缺少 __meta__）: {path}')
                meta = json.loads(str(z['__meta__']))
                if meta.get('format') != MODEL_FORMAT:
                    raise ValueError(
                        f'模型格式不匹配: {meta.get("format")} != {MODEL_FORMAT}')
                arrays = {k: np.array(z[k]) for k in z.files if k != '__meta__'}
        else:
            raise ValueError(f'不支持的 v3 模型文件后缀: {path.suffix}（应为 .npz）')
        n_seeds = int(meta.get('n_seeds', len(meta.get('seeds', []))))
        seed_params = []
        for si in range(n_seeds):
            prefix = f'cnn_s{si}/'
            subset = {k[len(prefix):]: v for k, v in arrays.items() if k.startswith(prefix)}
            if not subset:
                raise ValueError(f'模型缺少第 {si} 个种子的参数')
            seed_params.append({k: v.astype(np.float64) for k, v in subset.items()})
        if thresholds_path is None:
            candidate = path.with_name(THRESHOLDS_FILENAME)
            thresholds_path = candidate if candidate.exists() else None
        thresholds = None
        if thresholds_path is not None and Path(thresholds_path).exists():
            thresholds = json.loads(Path(thresholds_path).read_text(encoding='utf-8'))
        model = cls(meta, seed_params, thresholds=thresholds)
        model.model_path = path
        return model

    # ------------------------------------------------------------- 特征
    def feature_sequence(self, audio_path):
        from .audio_io import load_wav
        audio, sr = load_wav(audio_path)
        if int(sr) != int(self.feature_spec.sample_rate):
            raise ValueError(
                f'采样率不匹配：音频 {sr} Hz，模型要求 {self.feature_spec.sample_rate} Hz')
        return extract_sequence(audio, sr, self.feature_spec)

    def _raw_logits(self, x: np.ndarray) -> np.ndarray:
        """x [1,D,n_frames] -> 平均 logits [1,C]。"""
        outs = [cnn_logits(p, x) for p in self._seed_params]
        return np.mean(np.stack(outs, 0), axis=0)

    def _forward(self, seq: np.ndarray, bias: Optional[float] = None,
                 bias_class: Optional[str] = None) -> dict:
        """一次前向，同时给出部署概率、逐种子概率与嵌入（卷积只算一遍）。

        两种聚合方式**不等价**，不能互相替换：
        - 部署概率 = softmax(各种子 logits 的均值)，与 ``_raw_logits`` 路径逐位一致，
          线上判定只认这个；
        - 逐种子概率 = 各种子分别 softmax，携带集成分歧信息，只供选择性分类
          （``src.selective``）算方差/一致性信号。
        拿逐种子概率的均值当部署置信度会改变线上判定，禁止。

        ``bias`` / ``bias_class`` 是 Pass-2 的关键词偏置（U2-KWS 简化版）：
        给 ``bias_class`` 的 logit 加上 ``bias``。偏置加在**每个种子的 logits 上**，
        因此部署概率与逐种子概率的变化一致；``bias=None`` 时逐位等价于无偏置路径。
        """
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError('seq 必须是二维 [T,D]')
        bias_index = None
        if bias and bias_class and bias_class in self.classes_:
            bias_index = self.classes_.index(bias_class)
        variants = list(self.tta_shifts) if self.tta_shifts else [0]
        base = prepare_input(seq, self.n_frames)  # [1,D,T]
        deployed, seed_rows, emb_rows = [], [], []
        for shift in variants:
            x = base if shift == 0 else np.roll(base, int(shift), axis=2)
            embs = np.stack([cnn_embed(p, x)[0] for p in self._seed_params], axis=0)
            zs = np.stack([embs[s] @ self._seed_params[s]['fc.weight'].T
                           + self._seed_params[s]['fc.bias']
                           for s in range(embs.shape[0])], axis=0)  # [S,C]
            if bias_index is not None:
                zs[:, bias_index] += float(bias)
            deployed.append(softmax_np(zs.mean(axis=0, keepdims=True), self.temperature)[0])
            seed_rows.append(softmax_np(zs, self.temperature))
            emb_rows.append(embs.mean(axis=0))
        return {
            'prob': np.mean(np.stack(deployed, 0), axis=0),
            'seed_probs': np.concatenate(seed_rows, axis=0),   # [S*V,C]
            'embedding': np.mean(np.stack(emb_rows, 0), axis=0),
        }

    def _prediction_from_prob(self, prob: np.ndarray) -> V3Prediction:
        prob = prob / prob.sum()
        order = np.argsort(-prob)
        best = int(order[0])
        second = float(prob[int(order[1])]) if len(order) > 1 else 0.0
        scores = {str(c): float(prob[i]) for i, c in enumerate(self.classes_)}
        # 距离用于兼容旧接口：用负对数似然表示（越小越可信）
        distance = float(-np.log(max(float(prob[best]), 1e-12)))
        entropy = float(-np.sum(prob * np.log(np.clip(prob, 1e-12, None))))
        pred = V3Prediction(
            command=self.classes_[best],
            confidence=float(prob[best]),
            scores=scores,
            distance=distance,
            margin=float(prob[best] - second),
            entropy=entropy,
        )
        pred.accepted = self.is_accepted(pred)
        return pred

    def predict_features(self, seq: np.ndarray, bias: Optional[float] = None,
                         bias_class: Optional[str] = None) -> V3Prediction:
        """输入帧级特征 [T,D]，返回预测结果。

        ``bias`` / ``bias_class`` 为可选的关键词偏置（见 ``_forward``）。
        """
        return self._prediction_from_prob(
            self._forward(seq, bias=bias, bias_class=bias_class)['prob'])

    def predict_detailed(self, seq: np.ndarray, bias: Optional[float] = None,
                         bias_class: Optional[str] = None):
        """返回 ``(V3Prediction, extras)``，extras 含逐种子概率与嵌入。

        给选择性分类用：一次前向拿全所有信号，避免为算方差再跑一遍网络。
        """
        out = self._forward(seq, bias=bias, bias_class=bias_class)
        return self._prediction_from_prob(out['prob']), {
            'seed_probs': out['seed_probs'],
            'embedding': out['embedding'],
        }

    def predict(self, audio_path) -> V3Prediction:
        return self.predict_features(self.feature_sequence(audio_path))

    # 兼容 ``src.dtw_model.DTWControlModel`` 风格
    predict_sequence = predict_features

    # ------------------------------------------------------------- 拒识
    def is_accepted(self, pred: V3Prediction, threshold: Optional[float] = None) -> bool:
        thr = self.threshold if threshold is None else float(threshold)
        return bool(pred.command in self.classes_ and pred.confidence >= thr)

    # ------------------------------------------------------------- 信息
    def info(self) -> dict:
        info = {
            'model_name': self.model_name,
            'format': self.meta.get('format'),
            'classes': self.classes_,
            'n_seeds': len(self._seed_params),
            'temperature': self.temperature,
            'threshold': self.threshold,
            'feature_spec': self.feature_spec.to_dict(),
            'n_frames': self.n_frames,
            'tta_shifts': self.tta_shifts,
            'val': self.meta.get('val'),
            'data': self.meta.get('data'),
            'training': self.meta.get('training'),
            'created_at': self.meta.get('created_at'),
        }
        return info

    def embed_features(self, seq: np.ndarray) -> np.ndarray:
        """返回多种子平均的 256 维嵌入（实验分析用，不改预测）。"""
        x = prepare_input(seq, self.n_frames)
        embs = [cnn_embed(p, x) for p in self._seed_params]
        return np.mean(np.stack(embs, 0), axis=0)[0]

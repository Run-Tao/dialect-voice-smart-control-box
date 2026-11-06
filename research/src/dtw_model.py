# -*- coding: utf-8 -*-
"""DTW模板控制模型：保存模板、预测和阈值。"""
import pickle
from pathlib import Path
import numpy as np
from .baselines import predict_dtw


class DTWControlModel:
    def __init__(self, templates, threshold=0.7, max_frames=60, temperature=0.2):
        self.templates = templates
        self.threshold = float(threshold)
        self.max_frames = int(max_frames)
        self.temperature = float(temperature)
        self.classes_ = sorted(templates.keys())

    @classmethod
    def fit(cls, records, extractor, threshold=0.7, max_frames=60, temperature=0.2):
        templates = {}
        for r in records:
            templates.setdefault(r.command, []).append(extractor.sequence_from_record(r))
        if not templates:
            raise ValueError('没有构建出DTW模板')
        return cls(templates, threshold=threshold, max_frames=max_frames, temperature=temperature)

    def predict_sequence(self, seq):
        return predict_dtw(seq, self.templates, normalize='path',
                           temperature=self.temperature, max_frames=self.max_frames)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'templates': self.templates,
                'threshold': self.threshold,
                'max_frames': self.max_frames,
                'temperature': self.temperature,
                'classes_': self.classes_,
                'model_type': 'dtw',
            }, f)
        return path

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return DTWControlModel(d['templates'], d['threshold'],
                               d['max_frames'], d['temperature'])

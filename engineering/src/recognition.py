# -*- coding: utf-8 -*-
"""DTW 模板识别和用户模板库。"""
import json
from pathlib import Path
import numpy as np


def dtw_distance(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = len(x), len(y)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(x[i - 1] - y[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / max(n, m))


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / max(temperature, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class TemplateBank:
    def __init__(self, commands, temperature=0.5, confidence_threshold=0.55):
        self.commands = list(commands)
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.global_templates = {c: [] for c in self.commands}
        self.user_templates = {}

    def add_global(self, command, feat):
        if command not in self.global_templates:
            raise ValueError('未知命令: ' + command)
        self.global_templates[command].append(np.asarray(feat, dtype=np.float32))

    def add_user(self, speaker, command, feat):
        self.user_templates.setdefault(speaker, {}).setdefault(command, []).append(
            np.asarray(feat, dtype=np.float32)
        )

    def _templates_for(self, command, speaker=None):
        templates = list(self.global_templates.get(command, []))
        if speaker and speaker in self.user_templates:
            templates += list(self.user_templates[speaker].get(command, []))
        return templates

    def predict(self, feat, speaker=None):
        feat = np.asarray(feat, dtype=np.float32)
        distances = []
        for c in self.commands:
            templates = self._templates_for(c, speaker)
            if not templates:
                distances.append(np.inf)
            else:
                distances.append(min(dtw_distance(feat, t) for t in templates))
        distances = np.asarray(distances, dtype=np.float64)
        conf = softmax(-distances, self.temperature)
        idx = int(np.argmin(distances))
        command = self.commands[idx]
        confidence = float(conf[idx])
        return command, confidence, distances

    def save(self, path):
        data = {
            'commands': self.commands,
            'temperature': self.temperature,
            'confidence_threshold': self.confidence_threshold,
            'global_templates': {k: [v.tolist() for v in vs] for k, vs in self.global_templates.items()},
            'user_templates': {sp: {k: [v.tolist() for v in vs] for k, vs in cmds.items()}
                               for sp, cmds in self.user_templates.items()},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    @staticmethod
    def load(path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        bank = TemplateBank(data['commands'], data['temperature'], data['confidence_threshold'])
        for c, items in data['global_templates'].items():
            for arr in items:
                bank.add_global(c, arr)
        for sp, cmds in data['user_templates'].items():
            for c, items in cmds.items():
                for arr in items:
                    bank.add_user(sp, c, arr)
        return bank

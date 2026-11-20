# -*- coding: utf-8 -*-
"""按说话人划分训练集、验证集和测试集。"""
from dataclasses import dataclass
import random
from collections import Counter


@dataclass
class SplitResult:
    train: list
    val: list
    test: list
    seed: int

    @property
    def train_speakers(self):
        return sorted({r.speaker_id for r in self.train})

    @property
    def val_speakers(self):
        return sorted({r.speaker_id for r in self.val})

    @property
    def test_speakers(self):
        return sorted({r.speaker_id for r in self.test})


def split_by_speaker(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    if not records:
        raise ValueError("records为空")
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("划分比例之和必须为1")

    speakers = sorted({r.speaker_id for r in records})
    if len(speakers) < 3:
        raise ValueError("至少需要3位说话人才能做train/val/test划分")

    rng = random.Random(seed)
    shuffled = list(speakers)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("说话人数量不足以形成非空训练集")

    train_spk = set(shuffled[:n_train])
    val_spk = set(shuffled[n_train:n_train + n_val])
    test_spk = set(shuffled[n_train + n_val:])

    train = [r for r in records if r.speaker_id in train_spk]
    val = [r for r in records if r.speaker_id in val_spk]
    test = [r for r in records if r.speaker_id in test_spk]
    return SplitResult(train=train, val=val, test=test, seed=seed)


def split_summary(split):
    def part(records, speakers):
        return {
            'num_utterances': len(records),
            'num_speakers': len(speakers),
            'speakers': sorted(speakers),
            'commands': dict(Counter(r.command for r in records)),
        }

    return {
        'train': part(split.train, split.train_speakers),
        'val': part(split.val, split.val_speakers),
        'test': part(split.test, split.test_speakers),
        'speaker_disjoint': _is_disjoint(split.train_speakers, split.val_speakers, split.test_speakers),
    }


def _is_disjoint(*groups):
    seen = set()
    for g in groups:
        s = set(g)
        if seen & s:
            return False
        seen |= s
    return True

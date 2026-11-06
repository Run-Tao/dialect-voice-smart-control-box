# -*- coding: utf-8 -*-
"""用户注册流程：每个命令录几次，加入个人模板。"""
import numpy as np
from .features import sequence_feature


class EnrollmentSession:
    def __init__(self, bank, fs=16000, samples_per_command=3):
        self.bank = bank
        self.fs = fs
        self.samples_per_command = samples_per_command

    def enroll(self, speaker, command, audio_samples):
        """audio_samples: list of 1D audio arrays。"""
        count = 0
        for audio in audio_samples[:self.samples_per_command]:
            feat = sequence_feature(np.asarray(audio), fs=self.fs)
            self.bank.add_user(speaker, command, feat)
            count += 1
        return count

# -*- coding: utf-8 -*-
"""音频输入 -> 识别/拒识 -> 模拟或真实设备 -> 日志。"""
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from .audio_io import load_wav
from .features import extract_mfcc
from .representations import sentence_vector
from .serial_protocol import MockEsp32Controller, build_command


class MockControlDevice:
    def __init__(self):
        self.controller = MockEsp32Controller()

    def execute(self, seq, intent_id):
        return self.controller.handle_line(build_command(seq, intent_id))

    @property
    def state(self):
        return dict(self.controller.state)


class AudioControlServer:
    def __init__(self, bundle, feature_config, device=None, log_path=None):
        self.bundle = bundle
        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.threshold = float(bundle['threshold'])
        self.feature_config = feature_config
        self.device = device or MockControlDevice()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _features(self, audio_path):
        audio, sr = load_wav(audio_path)
        cfg = self.feature_config
        if sr != int(cfg['sample_rate']):
            raise ValueError(f'采样率不匹配: {sr} vs {cfg["sample_rate"]}')
        seq = extract_mfcc(
            audio, sr,
            n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
            n_mfcc=cfg['n_mfcc'], n_mels=cfg['n_mels'],
            fmin=cfg['fmin'], fmax=cfg['fmax'],
            preemphasis=cfg['preemphasis'], use_cmvn=cfg['use_cmvn']
        )
        return sentence_vector(seq)

    def predict(self, audio_path):
        x = self._features(audio_path).reshape(1, -1)
        x = self.scaler.transform(x)
        proba = self.model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        intent = str(self.model.classes_[idx])
        confidence = float(proba[idx])
        return intent, confidence, {str(c): float(p) for c, p in zip(self.model.classes_, proba)}

    def process(self, audio_path, seq):
        t0 = time.perf_counter()
        intent, confidence, scores = self.predict(audio_path)
        accepted = confidence >= self.threshold
        if accepted:
            response = self.device.execute(seq, intent)
        else:
            response = 'REJECTED\n'
        result = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'audio_path': str(audio_path),
            'seq': int(seq),
            'predicted_intent': intent,
            'confidence': confidence,
            'threshold': self.threshold,
            'accepted': accepted,
            'device_response': response.strip(),
            'device_state': self.device.state,
            'scores': scores,
            'latency_sec': round(time.perf_counter() - t0, 4),
        }
        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result


class DTWControlServer:
    """DTW模板版音频控制服务。"""
    def __init__(self, dtw_model, feature_config, device=None, log_path=None):
        self.model = dtw_model
        self.feature_config = feature_config
        self.device = device or MockControlDevice()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _sequence(self, audio_path):
        audio, sr = load_wav(audio_path)
        cfg = self.feature_config
        if sr != int(cfg['sample_rate']):
            raise ValueError(f'采样率不匹配: {sr} vs {cfg["sample_rate"]}')
        return extract_mfcc(
            audio, sr,
            n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
            n_mfcc=cfg['n_mfcc'], n_mels=cfg['n_mels'],
            fmin=cfg['fmin'], fmax=cfg['fmax'],
            preemphasis=cfg['preemphasis'], use_cmvn=cfg['use_cmvn']
        )

    def predict(self, audio_path):
        p = self.model.predict_sequence(self._sequence(audio_path))
        return p.command, float(p.confidence), p.scores

    def process(self, audio_path, seq):
        t0 = time.perf_counter()
        intent, confidence, scores = self.predict(audio_path)
        accepted = confidence >= self.model.threshold
        if accepted:
            response = self.device.execute(seq, intent)
        else:
            response = 'REJECTED\n'
        result = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'audio_path': str(audio_path),
            'seq': int(seq),
            'model_type': 'dtw',
            'predicted_intent': intent,
            'confidence': confidence,
            'threshold': self.model.threshold,
            'accepted': accepted,
            'device_response': response.strip(),
            'device_state': self.device.state,
            'scores': scores,
            'latency_sec': round(time.perf_counter() - t0, 4),
        }
        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return result

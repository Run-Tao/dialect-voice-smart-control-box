# -*- coding: utf-8 -*-
"""从ManifestRecord提取MFCC序列和句子级特征。"""
import numpy as np
from .audio_io import load_wav
from .features import extract_mfcc
from .alignment import fixed_length_sequence


class FeatureExtractor:
    def __init__(self, feature_config):
        self.cfg = feature_config

    def sequence_from_record(self, record):
        audio, sr = load_wav(record.audio_path)
        cfg = self.cfg
        if sr != int(cfg['sample_rate']):
            raise ValueError(
                f"采样率不一致: 文件{sr}, 配置{cfg['sample_rate']}, 文件={record.audio_path}"
            )
        return extract_mfcc(
            audio, sr,
            n_fft=cfg['n_fft'],
            hop_length=cfg['hop_length'],
            n_mfcc=cfg['n_mfcc'],
            n_mels=cfg['n_mels'],
            fmin=cfg['fmin'],
            fmax=cfg['fmax'],
            preemphasis=cfg['preemphasis'],
            use_cmvn=cfg['use_cmvn'],
        )

    def fixed_sequence_from_record(self, record, target_len):
        return fixed_length_sequence(self.sequence_from_record(record), target_len)


def sentence_vector(seq, use_delta=True):
    """将[T,D]序列转成固定维特征。"""
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[0] == 0:
        raise ValueError("seq必须是[T,D]且T>0")
    parts = [seq.mean(axis=0), seq.std(axis=0)]
    if use_delta:
        if seq.shape[0] > 1:
            delta = np.diff(seq, axis=0)
            parts.extend([delta.mean(axis=0), delta.std(axis=0)])
        else:
            z = np.zeros(seq.shape[1], dtype=np.float64)
            parts.extend([z, z])
    return np.concatenate(parts).astype(np.float32)


def select_template_records(records, template_language=None):
    if template_language is None:
        return list(records)
    selected = [r for r in records if getattr(r, 'language', '') == template_language]
    if not selected:
        raise ValueError(f"没有找到language={template_language}的模板记录")
    return selected


def build_cosine_templates(records, extractor, target_len, template_language=None):
    selected = select_template_records(records, template_language)
    grouped = {}
    for r in selected:
        seq = extractor.fixed_sequence_from_record(r, target_len)
        grouped.setdefault(r.command, []).append(seq)
    templates = {}
    for cmd, seqs in grouped.items():
        templates[cmd] = np.mean(np.stack(seqs, axis=0), axis=0).astype(np.float32)
    if not templates:
        raise ValueError("没有构建出余弦模板")
    return templates


def build_dtw_templates(records, extractor, template_language=None):
    selected = select_template_records(records, template_language)
    templates = {}
    for r in selected:
        seq = extractor.sequence_from_record(r)
        templates.setdefault(r.command, []).append(seq)
    if not templates:
        raise ValueError("没有构建出DTW模板")
    return templates

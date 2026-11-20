# -*- coding: utf-8 -*-
"""第一阶段冒烟检查：不计算实验准确率。"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest, manifest_summary
from src.split import split_by_speaker, split_summary
from src.audio_io import load_wav
from src.features import extract_mfcc
from src.alignment import fixed_length_sequence, dtw_align_pair
from src.results import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--manifest', default='data/smoke_manifest.csv')
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = load_manifest(args.manifest, check_files=True)
    split = split_by_speaker(
        records,
        train_ratio=cfg['split']['train_ratio'],
        val_ratio=cfg['split']['val_ratio'],
        test_ratio=cfg['split']['test_ratio'],
        seed=cfg['split']['seed'],
    )

    feat_cfg = cfg['features']
    audio, sr = load_wav(records[0].audio_path)
    mfcc = extract_mfcc(
        audio, sr,
        n_fft=feat_cfg['n_fft'],
        hop_length=feat_cfg['hop_length'],
        n_mfcc=feat_cfg['n_mfcc'],
        n_mels=feat_cfg['n_mels'],
        fmin=feat_cfg['fmin'],
        fmax=feat_cfg['fmax'],
        preemphasis=feat_cfg['preemphasis'],
        use_cmvn=feat_cfg['use_cmvn'],
    )
    fixed = fixed_length_sequence(mfcc, cfg['alignment']['fixed_length_frames'])

    # 找两条同命令、不同说话人的样本做DTW接口检查
    dtw_info = None
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i].command == records[j].command and records[i].speaker_id != records[j].speaker_id:
                a, _ = load_wav(records[i].audio_path)
                b, _ = load_wav(records[j].audio_path)
                fa = extract_mfcc(a, sr, **{k: feat_cfg[k] for k in [
                    'n_fft', 'hop_length', 'n_mfcc', 'n_mels', 'fmin', 'fmax', 'preemphasis', 'use_cmvn'
                ]})
                fb = extract_mfcc(b, sr, **{k: feat_cfg[k] for k in [
                    'n_fft', 'hop_length', 'n_mfcc', 'n_mels', 'fmin', 'fmax', 'preemphasis', 'use_cmvn'
                ]})
                xa, ya, path, dist = dtw_align_pair(fa, fb)
                dtw_info = {
                    'utterance_a': records[i].utterance_id,
                    'utterance_b': records[j].utterance_id,
                    'aligned_frames': len(path),
                    'distance': dist,
                    'first_path': path[:5],
                    'last_path': path[-5:],
                }
                break
        if dtw_info is not None:
            break

    summary = {
        'note': 'smoke check only; no experimental metrics are reported',
        'manifest': manifest_summary(records),
        'split': split_summary(split),
        'features': {
            'sample_rate_loaded': sr,
            'mfcc_shape': list(mfcc.shape),
            'fixed_length_shape': list(fixed.shape),
            'finite': bool(np.all(np.isfinite(mfcc))),
        },
        'dtw': dtw_info,
    }
    out_dir = Path(cfg['outputs']['directory'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_json(out_dir / 'stage1_summary.json', summary)
    print('冒烟检查完成，结果:', out_path)
    print('MFCC shape:', mfcc.shape)
    print('固定长度 shape:', fixed.shape)
    print('说话人划分不重叠:', summary['split']['speaker_disjoint'])


if __name__ == '__main__':
    main()

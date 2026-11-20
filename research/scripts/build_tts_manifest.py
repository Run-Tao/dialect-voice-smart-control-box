# -*- coding: utf-8 -*-
"""从TTS日志生成manifest样例。路径相对data/corpus。"""
import csv
from pathlib import Path

base = Path(r'D:\Run_Tao\DialectVoice_Research')
out = base / 'data' / 'corpus' / 'tts_manifest_sample.csv'
rows = []

def add_rows(log_path, src_dir, source):
    with open(log_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            stem = Path(r['audio_path']).stem
            rel = Path('..') / 'tts_validation' / src_dir / (stem + '.wav')
            rows.append({
                'utt_id': r['utt_id'],
                'audio_path': str(rel).replace('\\', '/'),
                'language': 'yue',
                'dialect': 'yue-HK',
                'text': r['text'],
                'intent_id': r['intent_id'],
                'expression_type': r['expression_type'],
                'semantic_pair_id': f"{r['intent_id']}_{r['expression_type']}",
                'source': source,
                'speaker_id': r['voice'],
                'is_synthetic': 'true',
                'duration': r['duration_sec'],
                'split_group': source + '_' + r['voice'],
                'text_verified_by': 'PENDING_NATIVE_SPEAKER',
                'quality_flags': 'SYNTHETIC;UNVERIFIED_TEXT',
                'source_model': r['voice'],
                'notes': '仅用于流程测试；不是真人语音'
            })

add_rows(base / 'outputs' / 'tts_canto_log.csv', 'canto_16k', 'canto_tts')
add_rows(base / 'outputs' / 'tts_edge_log.csv', 'edge_16k', 'edge_tts')

with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('manifest:', out, 'rows=', len(rows))

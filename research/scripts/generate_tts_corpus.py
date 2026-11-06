# -*- coding: utf-8 -*-
"""批量生成粤语/普通话TTS语料，统一转16kHz WAV，并写manifest。"""
import argparse
import asyncio
import csv
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import edge_tts
from canto_tts import CantoTTS

base = Path(r'D:\Run_Tao\DialectVoice_Research')
expr_csv = base / 'data' / 'corpus' / 'core_six_expressions.csv'
out_root = base / 'data' / 'tts_corpus'
raw_root = out_root / 'raw'
wav_root = out_root / 'wav16k'
for d in [raw_root/'canto', raw_root/'edge_canto', raw_root/'edge_mandarin',
          wav_root/'canto', wav_root/'edge_canto', wav_root/'edge_mandarin']:
    d.mkdir(parents=True, exist_ok=True)

CANTO_VOICES = ['zh-HK-HiuGaaiNeural', 'zh-HK-HiuMaanNeural', 'zh-HK-WanLungNeural']
MANDARIN_VOICES = ['zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural']

def load_rows():
    with open(expr_csv, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def convert_to_16k(src, dst):
    audio, sr = sf.read(str(src), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        new_len = int(round(len(audio) * 16000 / sr))
        old_t = np.linspace(0, 1, len(audio), endpoint=False)
        new_t = np.linspace(0, 1, new_len, endpoint=False)
        audio = np.interp(new_t, old_t, audio).astype(np.float32)
    sf.write(str(dst), audio, 16000, subtype='PCM_16')
    return sf.info(str(dst))

def gen_edge(text, voice, raw_path):
    asyncio.run(edge_tts.Communicate(text, voice).save(str(raw_path)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-canto', action='store_true')
    parser.add_argument('--skip-edge', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    rows = load_rows()
    log_rows = []
    canto = None
    if not args.skip_canto:
        canto = CantoTTS()

    for r in rows:
        lang = r['language']
        dialect = r['dialect']
        text = r['text']
        intent = r['intent_id']
        expr = r['expression_type']
        pair = r['semantic_pair_id']
        if lang == 'yue':
            dialects = [('canto', 'canto-tts-default', 'canto_tts', None)]
            for voice in CANTO_VOICES:
                dialects.append(('edge_canto', voice, 'edge_tts_canto', voice))
        else:
            dialects = []
            for voice in MANDARIN_VOICES:
                dialects.append(('edge_mandarin', voice, 'edge_tts_mandarin', voice))

        for folder, voice, source, edge_voice in dialects:
            if source == 'canto_tts' and args.skip_canto:
                continue
            if source != 'canto_tts' and args.skip_edge:
                continue
            stem = f"{intent}_{expr}_{voice}"
            raw_ext = '.wav' if source == 'canto_tts' else '.mp3'
            raw_path = raw_root / folder / (stem + raw_ext)
            dst = wav_root / folder / (stem + '.wav')
            if dst.exists() and not args.overwrite:
                info = sf.info(str(dst))
                log_rows.append({
                    'utt_id': stem, 'audio_path': str(dst.relative_to(out_root)).replace('\\','/'),
                    'language': lang, 'dialect': dialect, 'text': text,
                    'intent_id': intent, 'expression_type': expr,
                    'semantic_pair_id': pair, 'source': source, 'speaker_id': voice,
                    'is_synthetic': 'true', 'duration': info.duration,
                    'sample_rate': info.samplerate, 'split_group': f'{source}:{voice}',
                    'text_verified_by': 'PENDING_NATIVE_SPEAKER',
                    'quality_flags': 'SYNTHETIC;UNVERIFIED_TEXT',
                    'generation_sec': '0', 'status': 'skipped_existing', 'error': ''
                })
                continue
            t0 = time.perf_counter()
            err = ''
            try:
                if source == 'canto_tts':
                    canto.synthesize(text, str(raw_path))
                else:
                    gen_edge(text, edge_voice, raw_path)
                info = convert_to_16k(raw_path, dst)
                status = 'ok'
                duration = info.duration
                sr = info.samplerate
            except Exception as exc:
                status = 'fail'
                err = repr(exc)
                duration = ''
                sr = ''
                print('FAIL', stem, err)
            elapsed = time.perf_counter() - t0
            log_rows.append({
                'utt_id': stem, 'audio_path': str(dst.relative_to(out_root)).replace('\\','/') if dst.exists() else '',
                'language': lang, 'dialect': dialect, 'text': text,
                'intent_id': intent, 'expression_type': expr,
                'semantic_pair_id': pair, 'source': source, 'speaker_id': voice,
                'is_synthetic': 'true', 'duration': duration, 'sample_rate': sr,
                'split_group': f'{source}:{voice}',
                'text_verified_by': 'PENDING_NATIVE_SPEAKER',
                'quality_flags': 'SYNTHETIC;UNVERIFIED_TEXT',
                'generation_sec': round(elapsed, 3), 'status': status, 'error': err
            })
            print(status, stem, 'dur=', duration, 'sec=', round(elapsed,2))

    log_path = base / 'outputs' / 'tts_corpus_generation_log.csv'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader(); w.writerows(log_rows)
    manifest_path = out_root / 'manifest.csv'
    with open(manifest_path, 'w', encoding='utf-8-sig', newline='') as f:
        fields = ['utterance_id','audio_path','speaker_id','dialect','command','language',
                  'text','expression_type','semantic_pair_id','source','is_synthetic',
                  'duration','sample_rate','split_group','text_verified_by','quality_flags',
                  'notes']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in log_rows:
            if r['status'] == 'fail':
                continue
            w.writerow({
                'utterance_id': r['utt_id'], 'audio_path': 'wav16k/' + r['audio_path'].split('wav16k/')[-1] if 'wav16k/' in r['audio_path'] else r['audio_path'],
                'speaker_id': r['speaker_id'], 'dialect': r['dialect'], 'command': r['intent_id'],
                'language': r['language'], 'text': r['text'], 'expression_type': r['expression_type'],
                'semantic_pair_id': r['semantic_pair_id'], 'source': r['source'],
                'is_synthetic': r['is_synthetic'], 'duration': r['duration'], 'sample_rate': r['sample_rate'],
                'split_group': r['split_group'], 'text_verified_by': r['text_verified_by'],
                'quality_flags': r['quality_flags'], 'notes': 'TTS synthetic; not real dialect performance'
            })
    print('log:', log_path)
    print('manifest:', manifest_path, 'rows=', len([x for x in log_rows if x['status']!='fail']))

if __name__ == '__main__':
    main()

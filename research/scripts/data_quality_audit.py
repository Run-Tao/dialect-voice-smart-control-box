# -*- coding: utf-8 -*-
"""数据清单、音频质量、重复和划分泄漏审计。"""
import csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.audio_io import load_wav
from src.features import extract_mfcc

base=Path(r'D:\Run_Tao\DialectVoice_Research'); cfg=load_config(base/'config.json')
records=load_manifest(base/'data'/'tts_corpus'/'manifest.csv')
rows=[]
for r in records:
    audio,sr=load_wav(r.audio_path)
    sha=hashlib.sha256(Path(r.audio_path).read_bytes()).hexdigest()
    rms=float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    silence=float(np.mean(np.abs(audio)<0.01)) if len(audio) else 1.0
    clipping=float(np.mean(np.abs(audio)>=0.999)) if len(audio) else 1.0
    # TTS音频统一为16kHz单声道WAV
    rows.append({'utt_id':r.utterance_id,'path':r.audio_path,'sha256':sha,'sr':sr,
                 'samples':len(audio),'duration':len(audio)/sr if sr else 0,
                 'rms':rms,'silence_ratio':silence,'clipping_ratio':clipping,
                 'language':r.language,'source':r.source,'speaker':r.speaker_id,
                 'command':r.command,'expression':r.expression_type,'text':r.text,
                 'dialect':r.dialect})
report={'counts':{'manifest_rows':len(records),'actual_files':len({x['path'] for x in rows}),
                  'languages':dict(Counter(x['language'] for x in rows)),
                  'sources':dict(Counter(x['source'] for x in rows)),
                  'commands':dict(Counter(x['command'] for x in rows)),
                  'expressions':dict(Counter(x['expression'] for x in rows)),
                  'speakers':dict(Counter(x['speaker'] for x in rows))}}
report['audio']={'sample_rates':dict(Counter(x['sr'] for x in rows)),
                 'duration_min':min(x['duration'] for x in rows),
                 'duration_max':max(x['duration'] for x in rows),
                 'empty_files':sum(1 for x in rows if x['samples']==0),
                 'very_short_files':sum(1 for x in rows if x['duration']<0.3),
                 'very_long_files':sum(1 for x in rows if x['duration']>10.0),
                 'high_silence_files':sum(1 for x in rows if x['silence_ratio']>0.95),
                 'clipping_files':sum(1 for x in rows if x['clipping_ratio']>0.01)}
# exact duplicates
hashes=Counter(x['sha256'] for x in rows)
report['exact_duplicate_hashes']={k:v for k,v in hashes.items() if v>1}
keys=Counter((x['text'],x['speaker'],x['source'],x['command'],x['expression']) for x in rows)
report['duplicate_text_speaker_source_groups']=sum(1 for v in keys.values() if v>1)
# near duplicate on mean MFCC; only same text candidates to reduce cost and be meaningful
by_text={}
for x in rows:
    by_text.setdefault(x['text'],[]).append(x)
near=[]
for text,items in by_text.items():
    if len(items)<2: continue
    vecs=[]
    for x in items:
        a,sr=load_wav(x['path'])
        f=extract_mfcc(a,sr,n_fft=cfg['features']['n_fft'],hop_length=cfg['features']['hop_length'],
                       n_mfcc=cfg['features']['n_mfcc'],n_mels=cfg['features']['n_mels'],
                       fmin=cfg['features']['fmin'],fmax=cfg['features']['fmax'],
                       preemphasis=cfg['features']['preemphasis'],use_cmvn=cfg['features']['use_cmvn'])
        v=np.concatenate([f.mean(axis=0),f.std(axis=0)]); v=v/(np.linalg.norm(v)+1e-9); vecs.append(v)
    M=np.stack(vecs); sim=M@M.T
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            if sim[i,j]>0.999:
                near.append({'text':text,'a':items[i]['utt_id'],'b':items[j]['utt_id'],'cosine':float(sim[i,j])})
report['near_duplicate_pairs_same_text_cos_gt_0_999']=near
# split leakage: speaker split on yue
yue=[r for r in records if r.language=='yue']
from src.split import split_by_speaker
sp=split_by_speaker(yue,seed=42)
report['speaker_split']={'train_speakers':sp.train_speakers,'val_speakers':sp.val_speakers,
                         'test_speakers':sp.test_speakers,
                         'disjoint':len(set(sp.train_speakers)&set(sp.val_speakers)|set(sp.train_speakers)&set(sp.test_speakers)|set(sp.val_speakers)&set(sp.test_speakers))==0}
# expression split leakage
tr={r.utterance_id for r in yue if r.expression_type in {'direct','colloquial','polite'}}
va={r.utterance_id for r in yue if r.expression_type=='question'}
te={r.utterance_id for r in yue if r.expression_type=='correction'}
report['expression_split']={'train':len(tr),'val':len(va),'test':len(te),
                            'id_overlap_train_test':len(tr&te),
                            'text_overlap_train_test':len({r.text for r in yue if r.utterance_id in tr}&{r.text for r in yue if r.utterance_id in te}),
                            'speaker_overlap_train_test':len({r.speaker_id for r in yue if r.utterance_id in tr}&{r.speaker_id for r in yue if r.utterance_id in te})}
report['note']='TTS synthetic only; no real human speaker data'
out=base/'outputs'/'data_quality_audit.json'; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='near_duplicate_pairs_same_text_cos_gt_0_999'},ensure_ascii=False,indent=2))
print('near_duplicate_count=',len(near)); print('saved',out)

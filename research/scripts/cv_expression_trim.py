# -*- coding: utf-8 -*-
"""表达留一交叉验证（每折每类1条测试样本），评估trim变体。"""
import json,time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.manifest import load_manifest
from src.audio_io import load_wav
from src.features import extract_mfcc
from src.alignment import dtw_distance_general
from src.metrics import classification_metrics
base=Path(r'D:\Run_Tao\DialectVoice_Research')
recs=[r for r in load_manifest(base/'data'/'tts_corpus'/'manifest.csv') if r.language=='yue']
def trim(audio,sr):
    frame=int(0.02*sr); hop=int(0.01*sr); e=np.array([np.sum(audio[i:i+frame]**2) for i in range(0,len(audio)-frame+1,hop)]); idx=np.where(e>=np.max(e)*(10**(-3.5)))[0]
    return audio[max(0,idx[0]*hop-int(0.03*sr)):min(len(audio),(idx[-1]+1)*hop+frame+int(0.03*sr))] if len(idx) else audio
cache={}
for r in recs:
    a,sr=load_wav(r.audio_path); a=trim(a,sr); cache[r.utterance_id]=extract_mfcc(a,sr,n_fft=512,hop_length=160,n_mfcc=13,n_mels=26,fmin=50,fmax=8000,preemphasis=0.97,use_cmvn=True)
def one_per_class(rs):
    out=[]; counts={}
    for r in sorted(rs,key=lambda x:(x.command,x.utterance_id)):
        if counts.get(r.command,0)<1: out.append(r); counts[r.command]=counts.get(r.command,0)+1
    return out
folds={}
for expr in sorted({r.expression_type for r in recs}):
    tr=[r for r in recs if r.expression_type!=expr]; te=one_per_class([r for r in recs if r.expression_type==expr]); templates={}
    for r in tr: templates.setdefault(r.command,[]).append(cache[r.utterance_id])
    y=[]; p=[]; t0=time.perf_counter()
    for r in te:
        ds={c:min(dtw_distance_general(cache[r.utterance_id],t,metric='euclidean',norm='path',max_frames=60) for t in ts) for c,ts in templates.items()}; p.append(min(ds,key=ds.get)); y.append(r.command)
    m=classification_metrics(np.array(y),np.array(p)); m['n']=len(te); m['mean_latency_sec']=(time.perf_counter()-t0)/len(te); folds[expr]=m
acc=[v['accuracy'] for v in folds.values()]; f1=[v['macro_f1'] for v in folds.values()]
out={'protocol':'leave-one-expression-out; all 4 voices in each fold; one test sample per class; trim variant','folds':folds,'mean_accuracy':float(np.mean(acc)),'std_accuracy':float(np.std(acc)),'mean_macro_f1':float(np.mean(f1)),'std_macro_f1':float(np.std(f1)),'note':'synthetic TTS only'}
path=base/'outputs'/'cv_expression_trim.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2)); print('saved',path)

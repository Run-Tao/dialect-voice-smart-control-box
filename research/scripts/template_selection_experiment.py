# -*- coding: utf-8 -*-
"""模板选择/聚合对照：全模板min、medoid单模板、全模板top2均值。"""
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
train=[r for r in recs if r.speaker_id in {'zh-HK-HiuGaaiNeural','zh-HK-HiuMaanNeural'} and r.expression_type in {'direct','colloquial','polite'}]
val=[r for r in recs if r.speaker_id=='zh-HK-WanLungNeural' and r.expression_type=='question']
test=[r for r in recs if r.speaker_id=='canto-tts-default' and r.expression_type=='correction']
def trim(audio,sr):
    frame=int(0.02*sr); hop=int(0.01*sr); e=np.array([np.sum(audio[i:i+frame]**2) for i in range(0,len(audio)-frame+1,hop)]); idx=np.where(e>=np.max(e)*(10**(-3.5)))[0]
    return audio[max(0,idx[0]*hop-int(0.03*sr)):min(len(audio),(idx[-1]+1)*hop+frame+int(0.03*sr))] if len(idx) else audio
cache={}
for r in recs:
    a,sr=load_wav(r.audio_path); a=trim(a,sr); cache[r.utterance_id]=extract_mfcc(a,sr,n_fft=512,hop_length=160,n_mfcc=13,n_mels=26,fmin=50,fmax=8000,preemphasis=0.97,use_cmvn=True)
def dist(a,b): return dtw_distance_general(a,b,metric='euclidean',norm='path',max_frames=60)
# 全模板
templates={}
for r in train: templates.setdefault(r.command,[]).append(cache[r.utterance_id])
# medoid：每类选到同类其他模板总距离最小的模板
medoid={}
for c,ts in templates.items():
    scores=[sum(dist(t,u) for u in ts) for t in ts]; medoid[c]=[ts[int(np.argmin(scores))]]
def eval_tpl(rs,tpl,aggregate='min'):
    y=[]; p=[]; t0=time.perf_counter()
    for r in rs:
        q=cache[r.utterance_id]; ds={}
        for c,ts in tpl.items():
            vals=sorted(dist(q,t) for t in ts)
            ds[c]=float(np.mean(vals[:2])) if aggregate=='mean_top2' and len(vals)>1 else float(vals[0])
        p.append(min(ds,key=ds.get)); y.append(r.command)
    m=classification_metrics(np.array(y),np.array(p)); m['mean_latency_sec']=(time.perf_counter()-t0)/max(len(rs),1); return m
res={'all_min':{'val':eval_tpl(val,templates,'min'),'test':eval_tpl(test,templates,'min')},
     'medoid_min':{'val':eval_tpl(val,medoid,'min'),'test':eval_tpl(test,medoid,'min')},
     'all_top2':{'val':eval_tpl(val,templates,'mean_top2'),'test':eval_tpl(test,templates,'mean_top2')}}
best=max(res,key=lambda k:(res[k]['val']['accuracy'],res[k]['val']['macro_f1']))
out={'best_on_val':best,'results':res,'note':'synthetic TTS strict split; trim feature; selected on val only'}
path=base/'outputs'/'template_selection_experiment.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('best',best)
for k,v in res.items(): print(k,'val',round(v['val']['accuracy'],3),round(v['val']['macro_f1'],3),'test',round(v['test']['accuracy'],3),round(v['test']['macro_f1'],3))
print('saved',path)

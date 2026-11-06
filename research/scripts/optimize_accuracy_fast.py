# -*- coding: utf-8 -*-
"""快速变体筛选：只用val选参数，test只评估一次，不做CV。"""
import json,time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.audio_io import load_wav
from src.features import extract_mfcc
from src.alignment import dtw_distance_general
from src.metrics import classification_metrics
base=Path(r'D:\Run_Tao\DialectVoice_Research'); cfg=load_config(base/'config.json')
recs=[r for r in load_manifest(base/'data'/'tts_corpus'/'manifest.csv') if r.language=='yue']
train=[r for r in recs if r.speaker_id in {'zh-HK-HiuGaaiNeural','zh-HK-HiuMaanNeural'} and r.expression_type in {'direct','colloquial','polite'}]
val=[r for r in recs if r.speaker_id=='zh-HK-WanLungNeural' and r.expression_type=='question']
test=[r for r in recs if r.speaker_id=='canto-tts-default' and r.expression_type=='correction']
def trim_silence(audio,sr,threshold_db=-35):
    frame=int(0.02*sr); hop=int(0.01*sr)
    if len(audio)<frame: return audio
    e=np.array([np.sum(audio[i:i+frame]**2) for i in range(0,len(audio)-frame+1,hop)])
    idx=np.where(e>=np.max(e)*(10**(threshold_db/10)))[0]
    if len(idx)==0: return audio
    return audio[max(0,idx[0]*hop-int(0.03*sr)):min(len(audio),(idx[-1]+1)*hop+frame+int(0.03*sr))]
def seq(audio,sr,v):
    if v['trim']: audio=trim_silence(audio,sr)
    f=extract_mfcc(audio,sr,n_fft=v['n_fft'],hop_length=v['hop_length'],n_mfcc=v['n_mfcc'],n_mels=v['n_mels'],fmin=50,fmax=8000,preemphasis=0.97,use_cmvn=v['cmvn'])
    if v['delta']:
        d=np.diff(f,axis=0); d=np.vstack([np.zeros((1,f.shape[1])),d]); f=np.concatenate([f,d],axis=1)
    return f
variants=[
 ('baseline',{'n_fft':512,'hop_length':160,'n_mfcc':13,'n_mels':26,'cmvn':True,'delta':False,'trim':False,'metric':'euclidean','norm':'path','max_frames':60}),
 ('delta',{'n_fft':512,'hop_length':160,'n_mfcc':13,'n_mels':26,'cmvn':True,'delta':True,'trim':False,'metric':'euclidean','norm':'path','max_frames':60}),
 ('mfcc20',{'n_fft':512,'hop_length':160,'n_mfcc':20,'n_mels':40,'cmvn':True,'delta':False,'trim':False,'metric':'euclidean','norm':'path','max_frames':60}),
 ('trim',{'n_fft':512,'hop_length':160,'n_mfcc':13,'n_mels':26,'cmvn':True,'delta':False,'trim':True,'metric':'euclidean','norm':'path','max_frames':60}),
 ('cosine',{'n_fft':512,'hop_length':160,'n_mfcc':13,'n_mels':26,'cmvn':True,'delta':False,'trim':False,'metric':'cosine','norm':'path','max_frames':60}),
 ('max40',{'n_fft':512,'hop_length':160,'n_mfcc':13,'n_mels':26,'cmvn':True,'delta':False,'trim':False,'metric':'euclidean','norm':'path','max_frames':40}),
]
cache={n:{} for n,_ in variants}
for r in recs:
    a,sr=load_wav(r.audio_path)
    for n,v in variants: cache[n][r.utterance_id]=seq(a,sr,v)
def evalv(rs,v,train_recs,name):
    templates={}
    for r in train_recs: templates.setdefault(r.command,[]).append(cache[name][r.utterance_id])
    y=[]; p=[]; t0=time.perf_counter(); errors=[]
    for r in rs:
        q=cache[name][r.utterance_id]; ds={c:min(dtw_distance_general(q,t,metric=v['metric'],norm=v['norm'],max_frames=v['max_frames']) for t in ts) for c,ts in templates.items()}; pred=min(ds,key=ds.get)
        y.append(r.command); p.append(pred)
        if pred!=r.command: errors.append({'utt':r.utterance_id,'true':r.command,'pred':pred,'distances':ds})
    m=classification_metrics(np.array(y),np.array(p)); m['mean_latency_sec']=(time.perf_counter()-t0)/max(len(rs),1); m['errors']=errors
    return m
res={}; best=None
for name,v in variants:
    m=evalv(val,v,train,name); res[name]={'val':m}; key=(m['accuracy'],m['macro_f1'])
    if best is None or key>best[0]: best=(key,name,v)
bn=best[1]; bv=best[2]
res[bn]['test']=evalv(test,bv,train,bn)
res['baseline']['test']=evalv(test,variants[0][1],train,'baseline')
out={'protocol':'train=2 voices x 3 expressions; val=1 voice x question; test=1 voice x correction; no CV; selected on val only','best':bn,'best_config':bv,'variants':res,'note':'synthetic TTS only'}
path=base/'outputs'/'accuracy_optimization_fast.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('best',bn,bv)
for n,d in res.items(): print(n,'val',round(d['val']['accuracy'],3),round(d['val']['macro_f1'],3),'test',round(d.get('test',{}).get('accuracy',0),3) if 'test' in d else '-')
print('saved',path)

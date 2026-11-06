# -*- coding: utf-8 -*-
"""严格开发/测试划分下的特征和DTW变体选择，带特征缓存。"""
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
    energies=[np.sum(audio[i:i+frame]**2) for i in range(0,len(audio)-frame+1,hop)]
    e=np.array(energies); thr=np.max(e)*(10**(threshold_db/10)); idx=np.where(e>=thr)[0]
    if len(idx)==0: return audio
    start=max(0,idx[0]*hop-int(0.03*sr)); end=min(len(audio),(idx[-1]+1)*hop+frame+int(0.03*sr))
    return audio[start:end]

def make_seq(audio,sr,v):
    if v.get('trim'): audio=trim_silence(audio,sr)
    f=extract_mfcc(audio,sr,n_fft=v['n_fft'],hop_length=v['hop_length'],n_mfcc=v['n_mfcc'],n_mels=v['n_mels'],fmin=v['fmin'],fmax=v['fmax'],preemphasis=v['preemphasis'],use_cmvn=v['cmvn'])
    if v.get('delta'):
        d=np.diff(f,axis=0); d=np.vstack([np.zeros((1,f.shape[1])),d]); f=np.concatenate([f,d],axis=1)
    return f

variants=[]
basev={'n_fft':512,'hop_length':160,'n_mfcc':13,'n_mels':26,'fmin':50.0,'fmax':8000.0,'preemphasis':0.97,'cmvn':True,'delta':False,'trim':False,'metric':'euclidean','norm':'path','max_frames':60,'aggregate':'min'}
variants.append(('baseline',basev))
for name,over in [('delta',{'delta':True}),('mfcc20',{'n_mfcc':20,'n_mels':40}),('no_cmvn',{'cmvn':False}),('trim',{'trim':True}),('cosine',{'metric':'cosine'}),('max40',{'max_frames':40}),('delta_trim',{'delta':True,'trim':True})]:
    v=basev.copy(); v.update(over); variants.append((name,v))

# 一次加载音频，按变体缓存特征
cache={name:{} for name,_ in variants}
for r in recs:
    audio,sr=load_wav(r.audio_path)
    for name,v in variants:
        cache[name][r.utterance_id]=make_seq(audio,sr,v)

def predict(q,templates,v):
    ds={}
    for cmd,tpls in templates.items():
        vals=sorted(dtw_distance_general(q,t,metric=v['metric'],norm=v['norm'],max_frames=v['max_frames']) for t in tpls)
        ds[cmd]=float(np.mean(vals[:2])) if v.get('aggregate')=='mean_top2' and len(vals)>1 else float(vals[0])
    return min(ds,key=ds.get),ds

def evaluate(rs,v,train_recs,name):
    templates={}
    for r in train_recs: templates.setdefault(r.command,[]).append(cache[name][r.utterance_id])
    y=[]; p=[]; errors=[]; t0=time.perf_counter()
    for r in rs:
        pred,ds=predict(cache[name][r.utterance_id],templates,v); y.append(r.command); p.append(pred)
        if pred!=r.command: errors.append({'utt':r.utterance_id,'true':r.command,'pred':pred,'distances':ds})
    m=classification_metrics(np.array(y),np.array(p)); m['mean_latency_sec']=(time.perf_counter()-t0)/max(len(rs),1); m['errors']=errors
    return m

results={}; best=None
for name,v in variants:
    m=evaluate(val,v,train,name); results[name]={'val':m}; key=(m['accuracy'],m['macro_f1'])
    if best is None or key>best[0]: best=(key,name,v)
best_name=best[1]; best_v=best[2]
results[best_name]['test']=evaluate(test,best_v,train,best_name)
results['baseline']['test']=evaluate(test,basev,train,'baseline')

def one_per_class(rs):
    out=[]; counts={}
    for r in sorted(rs,key=lambda x:(x.command,x.utterance_id)):
        if counts.get(r.command,0)<1: out.append(r); counts[r.command]=counts.get(r.command,0)+1
    return out
expr_folds={}
for expr in sorted({r.expression_type for r in recs}):
    tr=[r for r in recs if r.expression_type!=expr]; te=one_per_class([r for r in recs if r.expression_type==expr])
    m=evaluate(te,best_v,tr,best_name); expr_folds[expr]={'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'n':len(te)}
voice_folds={}
for spk in sorted({r.speaker_id for r in recs}):
    tr=[r for r in recs if r.speaker_id!=spk]; te=one_per_class([r for r in recs if r.speaker_id==spk])
    m=evaluate(te,best_v,tr,best_name); voice_folds[spk]={'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'n':len(te)}
out={'protocol':'train=2 voices x 3 expressions; val=1 voice x question; test=1 voice x correction; selected on val only',
     'best':best_name,'best_config':best_v,'variants':results,
     'expression_leave_one_out':expr_folds,'voice_leave_one_out':voice_folds,
     'note':'synthetic TTS only; no real dialect performance'}
path=base/'outputs'/'accuracy_optimization.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('best',best_name,best_v)
for n,d in results.items(): print(n,'val',round(d['val']['accuracy'],3),round(d['val']['macro_f1'],3),'test',round(d.get('test',{}).get('accuracy',0),3) if 'test' in d else '-')
print('CV expr',expr_folds); print('CV voice',voice_folds); print('saved',path)

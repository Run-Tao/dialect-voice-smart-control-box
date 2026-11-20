# -*- coding: utf-8 -*-
"""探索性对照：不用于最终结论；test只报告，不用于选择。"""
from pathlib import Path
import json
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.split import split_by_speaker
from src.representations import FeatureExtractor
from src.classifiers import StandardScalerNumpy, SoftmaxRegression
from src.baselines import dtw_distance_normalized, predict_cosine
from src.metrics import classification_metrics

base=Path(r'D:\Run_Tao\DialectVoice_Research'); cfg=load_config(base/'config.json')
records=load_manifest(base/'data/tts_corpus/manifest_yue.csv')
split=split_by_speaker(records,seed=42)
ex=FeatureExtractor(cfg['features'])

def seq(r): return ex.sequence_from_record(r)
def mf(r, mode):
    s=seq(r); parts=[s.mean(0)]
    if mode in ('meanstd','meanstddelta'): parts.append(s.std(0))
    if mode=='meanstddelta':
        d=np.diff(s,axis=0); parts.extend([d.mean(0),d.std(0)])
    return np.concatenate(parts)

def knn_predict(Xtr,ytr,Xte,k=3,metric='cosine'):
    if metric=='cosine':
        Xtr=Xtr/(np.linalg.norm(Xtr,axis=1,keepdims=True)+1e-9); Xte=Xte/(np.linalg.norm(Xte,axis=1,keepdims=True)+1e-9); sim=Xte@Xtr.T; idx=np.argsort(-sim,axis=1)[:,:k]
    else:
        d=((Xte[:,None,:]-Xtr[None,:,:])**2).sum(-1); idx=np.argsort(d,axis=1)[:,:k]
    out=[]
    for ii in idx:
        vals,counts=np.unique(ytr[ii],return_counts=True); out.append(vals[np.argmax(counts)])
    return np.array(out)

def centroid_predict(Xtr,ytr,Xte):
    classes=sorted(set(ytr)); C=np.stack([Xtr[ytr==c].mean(0) for c in classes]); C=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-9); Z=Xte/(np.linalg.norm(Xte,axis=1,keepdims=True)+1e-9); return np.array([classes[i] for i in np.argmax(Z@C.T,axis=1)])

res={'note':'probe only; synthetic TTS; test not used for selection','classifiers':{},'dtw':{}}
for mode in ['mean','meanstd','meanstddelta']:
    Xtr=np.stack([mf(r,mode) for r in split.train]); ytr=np.array([r.command for r in split.train]); Xva=np.stack([mf(r,mode) for r in split.val]); yva=np.array([r.command for r in split.val]); Xte=np.stack([mf(r,mode) for r in split.test]); yte=np.array([r.command for r in split.test])
    sc=StandardScalerNumpy().fit(Xtr); Xtrs=sc.transform(Xtr); Xvas=sc.transform(Xva); Xtes=sc.transform(Xte)
    for name,model in [('softmax',SoftmaxRegression(l2=0.001,learning_rate=0.1,epochs=1000))]:
        model.fit(Xtrs,ytr); res['classifiers'][f'{name}_{mode}']={'val':classification_metrics(yva,model.predict(Xvas)),'test':classification_metrics(yte,model.predict(Xtes))}
    for k in [1,3,5,7]:
        res['classifiers'][f'knn{k}_{mode}']={'val':classification_metrics(yva,knn_predict(Xtrs,ytr,Xvas,k)),'test':classification_metrics(yte,knn_predict(Xtrs,ytr,Xtes,k))}
    res['classifiers'][f'centroid_{mode}']={'val':classification_metrics(yva,centroid_predict(Xtrs,ytr,Xvas)),'test':classification_metrics(yte,centroid_predict(Xtrs,ytr,Xtes))}

def dtw_feat(r, kind):
    s=seq(r)
    if kind=='mfcc': return s
    if kind=='mfcc_delta':
        d=np.diff(s,axis=0); d=np.vstack([np.zeros((1,s.shape[1])),d]); return np.concatenate([s,d],axis=1)
    raise ValueError(kind)
for kind in ['mfcc','mfcc_delta']:
    for nf in [30,40,60]:
        templates={}
        for r in split.train:
            templates.setdefault(r.command,[]).append(dtw_feat(r,kind))
        pred=[]
        for r in split.test:
            q=dtw_feat(r,kind); ds={c:min(dtw_distance_normalized(q,t,max_frames=nf) for t in ts) for c,ts in templates.items()}; pred.append(min(ds,key=ds.get))
        yte=np.array([r.command for r in split.test])
        res['dtw'][f'{kind}_max{nf}']={'test':classification_metrics(yte,np.array(pred))}
out=base/'outputs'/'probe_optimizations.json'; out.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(res,ensure_ascii=False,indent=2)); print('saved',out)

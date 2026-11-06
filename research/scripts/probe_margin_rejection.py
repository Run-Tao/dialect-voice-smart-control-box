# -*- coding: utf-8 -*-
"""探索置信度+top2 margin双条件拒识；阈值在验证/未知校准集选择。"""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.representations import FeatureExtractor
from src.dtw_model import DTWControlModel

base=Path(r'D:\Run_Tao\DialectVoice_Research'); cfg=load_config(base/'config.json'); ex=FeatureExtractor(cfg['features'])
model=DTWControlModel.load(base/'data'/'models'/'tts_control_model_dtw.pkl')
yue=[r for r in load_manifest(base/'data'/'tts_corpus'/'manifest.csv') if r.language=='yue']
val=[r for r in yue if r.expression_type=='question']; test=[r for r in yue if r.expression_type=='correction']
unknown=load_manifest(base/'data'/'tts_unknown'/'manifest.csv'); texts=sorted({r.text for r in unknown}); cal_texts=set(texts[:len(texts)//2]); ucal=[r for r in unknown if r.text in cal_texts]; utest=[r for r in unknown if r.text not in cal_texts]

def eval_set(rs, known):
 conf=[]; margin=[]; correct=[]
 for r in rs:
  p=model.predict_sequence(ex.sequence_from_record(r)); vals=sorted(p.scores.values(),reverse=True)
  conf.append(vals[0]); margin.append(vals[0]-vals[1] if len(vals)>1 else 0.0)
  if known: correct.append(p.command==r.command)
 return np.array(conf),np.array(margin),np.array(correct,dtype=bool)
vc,vm,vcorr=eval_set(val,True); tc,tm,tcorr=eval_set(test,True); ucc,ucm,_=eval_set(ucal,False); utc,utm,_=eval_set(utest,False)
best=None
for ct in np.linspace(0,0.9,19):
 for mt in np.linspace(0,0.8,17):
  ca=(ucc>=ct)&(ucm>=mt); far=float(np.mean(ca))
  ka=(vc>=ct)&(vm>=mt); cov=float(np.mean(ka)); acc=float(np.mean(vcorr[ka])) if ka.any() else 0
  if far<=0.1 and cov>=0.2:
   key=(cov,acc,-far)
   if best is None or key>best[0]: best=(key,ct,mt,cov,acc,far)
if best is None: best=((0,0,0),0,0,0,0,1)
_,ct,mt,cov,acc,far=best
ta=(tc>=ct)&(tm>=mt); ua=(utc>=ct)&(utm>=mt)
res={'note':'probe only; synthetic TTS; thresholds selected on val+unknown calibration','current_threshold':model.threshold,
     'best':{'conf_threshold':float(ct),'margin_threshold':float(mt),'cal_far':float(far),'val_coverage':float(cov),'val_accepted_accuracy':float(acc)},
     'test_known':{'coverage':float(ta.mean()),'accepted_accuracy':float(tcorr[ta].mean()) if ta.any() else 0.0},
     'unknown_test_accept_rate':float(ua.mean())}
path=base/'outputs'/'margin_rejection_probe.json'; path.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(res,ensure_ascii=False,indent=2)); print('saved',path)

# -*- coding: utf-8 -*-
"""粤语单方言训练 vs 粤语+普通话混合模板训练（小规模代理实验）。"""
import json, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.split import split_by_speaker
from src.representations import FeatureExtractor
from src.dtw_model import DTWControlModel
from src.metrics import classification_metrics

base=Path(r'D:\Run_Tao\DialectVoice_Research'); cfg=load_config(base/'config.json'); ex=FeatureExtractor(cfg['features'])
allrec=load_manifest(base/'data'/'tts_corpus'/'manifest.csv'); yue=[r for r in allrec if r.language=='yue']; mand=[r for r in allrec if r.language=='cmn']
split=split_by_speaker(yue,seed=42)
# 每个命令取2条测试样本，共12条，覆盖6类
test=[]; counts={}
for r in split.test:
 if counts.get(r.command,0)<2:
  test.append(r); counts[r.command]=counts.get(r.command,0)+1
# 普通话音色仅用Xiaoxiao，减少模板数和运行时间
mand_x=[r for r in mand if r.speaker_id=='zh-CN-XiaoxiaoNeural']
max_frames=int(cfg['baselines']['dtw'].get('max_frames',60))

def eval_model(model, records):
 t0=time.perf_counter(); pred=[]
 for r in records: pred.append(model.predict_sequence(ex.sequence_from_record(r)).command)
 dt=time.perf_counter()-t0
 y=np.array([r.command for r in records]); p=np.array(pred)
 return classification_metrics(y,p), dt/max(len(records),1), sum(len(v) for v in model.templates.values())

yue_model=DTWControlModel.fit(split.train,ex,threshold=0.0,max_frames=max_frames)
mixed_model=DTWControlModel.fit(split.train+mand_x,ex,threshold=0.0,max_frames=max_frames)
ym,yl,yt=eval_model(yue_model,test); mm,ml,mt=eval_model(mixed_model,test)
out={'note':'small synthetic proxy; 12 yue test utterances; Mandarin Xiaoxiao added to mixed template pool',
     'test_size':len(test),'max_frames':max_frames,
     'yue_only':{'accuracy':ym['accuracy'],'macro_f1':ym['macro_f1'],'templates':yt,'mean_latency_sec':yl},
     'yue_plus_mandarin':{'accuracy':mm['accuracy'],'macro_f1':mm['macro_f1'],'templates':mt,'mean_latency_sec':ml}}
path=base/'outputs'/'mixed_training_experiment.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2)); print('saved',path)

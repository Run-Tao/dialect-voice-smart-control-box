# -*- coding: utf-8 -*-
"""个性化注册原型：同一音色用direct+polite注册，测试correction。"""
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest
from src.representations import FeatureExtractor
from src.dtw_model import DTWControlModel
from src.metrics import classification_metrics

base=Path(r'D:\Run_Tao\DialectVoice_Research'); cfg=load_config(base/'config.json'); ex=FeatureExtractor(cfg['features'])
recs=[r for r in load_manifest(base/'data'/'tts_corpus'/'manifest.csv') if r.language=='yue']
max_frames=int(cfg['baselines']['dtw']['max_frames'])
all_y=[]; all_p=[]; per={}
for spk in sorted({r.speaker_id for r in recs}):
 enroll=[r for r in recs if r.speaker_id==spk and r.expression_type in {'direct','polite'}]
 test=[r for r in recs if r.speaker_id==spk and r.expression_type=='correction']
 if not enroll or not test: continue
 model=DTWControlModel.fit(enroll,ex,threshold=0.0,max_frames=max_frames)
 pred=[]; y=[]
 for r in test:
  pred.append(model.predict_sequence(ex.sequence_from_record(r)).command); y.append(r.command)
 m=classification_metrics(np.array(y),np.array(pred)); per[spk]=m; all_y.extend(y); all_p.extend(pred)
out={'note':'synthetic TTS; personalized enrollment scenario; same speaker used for enrollment and test, not speaker-independent',
     'enroll_expressions':['direct','polite'],'test_expression':'correction','per_speaker':per,
     'overall':classification_metrics(np.array(all_y),np.array(all_p))}
path=base/'outputs'/'enrollment_prototype.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2)); print('saved',path)

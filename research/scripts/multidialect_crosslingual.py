# -*- coding: utf-8 -*-
"""小规模跨语言/跨方言拒识测试，控制样本数以保证可运行。"""
import csv, json, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest, ManifestRecord
from src.representations import FeatureExtractor
from src.dtw_model import DTWControlModel

base=Path(r'D:\Run_Tao\DialectVoice_Research'); cfg=load_config(base/'config.json'); ex=FeatureExtractor(cfg['features'])
model=DTWControlModel.load(base/'data'/'models'/'tts_control_model_dtw.pkl')
yue=[r for r in load_manifest(base/'data'/'tts_corpus'/'manifest.csv') if r.language=='yue']
mand=[r for r in load_manifest(base/'data'/'tts_corpus'/'manifest.csv') if r.language=='cmn']
unk=load_manifest(base/'data'/'tts_unknown'/'manifest.csv')[:8]
sh=[]
shm=base/'data'/'tts_multidialect'/'shanghai_real_ood'/'manifest.csv'
if shm.exists():
 with open(shm,'r',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   sh.append(ManifestRecord(r['utterance_id'], str(shm.parent/r['audio_path']), r.get('speaker_id',''), 'shanghai-real', 'UNKNOWN', language='zh', notes='OOD real Shanghai speech'))
 sh=sh[:8]

def eval_ood(records):
 conf=[]; pred=[]; t0=time.perf_counter()
 for r in records:
  p=model.predict_sequence(ex.sequence_from_record(r)); conf.append(p.confidence); pred.append(p.command)
 dt=time.perf_counter()-t0
 return {'n':len(records),'accepted_rate':float(np.mean(np.array(conf)>=model.threshold)),
         'rejection_rate':float(np.mean(np.array(conf)<model.threshold)),
         'mean_confidence':float(np.mean(conf)),'mean_latency_sec':dt/max(len(records),1),
         'predictions':pred}
out={'threshold':model.threshold,'max_frames':model.max_frames,
     'unknown_tts':eval_ood(unk),
     'shanghai_real_ood':eval_ood(sh) if sh else {'error':'no shanghai samples'},
     'mandarin_crosslingual':eval_ood(mand[:12])}
# Mandarin forced accuracy on subset
mand12=mand[:12]; pred=[]; t0=time.perf_counter()
for r in mand12:
 p=model.predict_sequence(ex.sequence_from_record(r)); pred.append(p.command)
true=np.array([r.command for r in mand12]); pr=np.array(pred)
out['mandarin_crosslingual']['forced_accuracy']=float(np.mean(true==pr))
path=base/'outputs'/'multidialect_crosslingual.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2)); print('saved',path)

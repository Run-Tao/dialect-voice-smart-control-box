# -*- coding: utf-8 -*-
"""严格划分：训练/测试同时隔离说话人(音色)和表达模板。"""
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
train_speakers={'zh-HK-HiuGaaiNeural','zh-HK-HiuMaanNeural'}
val_speakers={'zh-HK-WanLungNeural'}
test_speakers={'canto-tts-default'}
train_expr={'direct','colloquial','polite'}; val_expr={'question'}; test_expr={'correction'}
train=[r for r in recs if r.speaker_id in train_speakers and r.expression_type in train_expr]
val=[r for r in recs if r.speaker_id in val_speakers and r.expression_type in val_expr]
test=[r for r in recs if r.speaker_id in test_speakers and r.expression_type in test_expr]
model=DTWControlModel.fit(train,ex,threshold=0.0,max_frames=int(cfg['baselines']['dtw']['max_frames']))
def pred(rs):
 p=[model.predict_sequence(ex.sequence_from_record(r)) for r in rs]
 y=np.array([r.command for r in rs]); q=np.array([x.command for x in p]); c=np.array([x.confidence for x in p])
 return classification_metrics(y,q), float(np.mean(c))
vm,vc=pred(val); tm,tc=pred(test)
out={'note':'strict split: speaker(s) and expression template both held out; synthetic TTS; tiny test set',
     'sizes':{'train':len(train),'val':len(val),'test':len(test)},
     'train_speakers':sorted(train_speakers),'val_speakers':sorted(val_speakers),'test_speakers':sorted(test_speakers),
     'train_expressions':sorted(train_expr),'val_expressions':sorted(val_expr),'test_expressions':sorted(test_expr),
     'val':vm,'test':tm,'val_mean_confidence':vc,'test_mean_confidence':tc}
path=base/'outputs'/'strict_split_experiment.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2)); print('saved',path)

# -*- coding: utf-8 -*-
"""多方言可行性小规模实验：粤语训练/测试、普通话跨语言、上海真实语料OOD拒识、混合训练。"""
import csv, json, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.manifest import load_manifest, ManifestRecord
from src.split import split_by_speaker
from src.representations import FeatureExtractor
from src.dtw_model import DTWControlModel
from src.metrics import classification_metrics
from src.rejection import rejection_metrics

base=Path(r'D:\Run_Tao\DialectVoice_Research')
cfg=load_config(base/'config.json'); ex=FeatureExtractor(cfg['features']); max_frames=int(cfg['baselines']['dtw'].get('max_frames',60))
all_yue=[r for r in load_manifest(base/'data'/'tts_corpus'/'manifest.csv') if r.language=='yue']
mandarin=[r for r in load_manifest(base/'data'/'tts_corpus'/'manifest.csv') if r.language=='cmn']
yue_split=split_by_speaker(all_yue,seed=42)
train=[r for r in yue_split.train]
test=[r for r in yue_split.test]
control_train=[r for r in all_yue if r.expression_type in {'direct','colloquial','polite'}]
control_val=[r for r in all_yue if r.expression_type=='question']
control_test=[r for r in all_yue if r.expression_type=='correction']
unknown=load_manifest(base/'data'/'tts_unknown'/'manifest.csv')

# 上海真实语料OOD
sh_manifest=base/'data'/'tts_multidialect'/'shanghai_real_ood'/'manifest.csv'
sh=[]
if sh_manifest.exists():
    with open(sh_manifest,'r',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            sh.append(ManifestRecord(r['utterance_id'], str(sh_manifest.parent/r['audio_path']), r.get('speaker_id',''), 'shanghai-real', 'UNKNOWN', language='zh', notes='OOD real Shanghai conversational speech'))

def predict(model, records):
    out=[]; t0=time.perf_counter()
    for r in records:
        out.append(model.predict_sequence(ex.sequence_from_record(r)))
    elapsed=time.perf_counter()-t0
    return out, elapsed/ max(len(records),1)

def top1_metrics(model, records):
    preds,_=predict(model,records)
    y=np.array([r.command for r in records]); p=np.array([x.command for x in preds])
    return classification_metrics(y,p)

def ood_accept(model, records):
    preds,lat=predict(model,records)
    conf=np.array([p.confidence for p in preds])
    return {'coverage':float(np.mean(conf>=model.threshold)),
            'rejection_rate':float(np.mean(conf<model.threshold)),
            'mean_confidence':float(np.mean(conf)) if len(conf) else 0,
            'mean_latency_sec':lat}

result={'note':'synthetic TTS except Shanghai real OOD; small sample; not real dialect performance',
        'max_frames':max_frames,
        'corpus_counts':{'yue':len(all_yue),'mandarin':len(mandarin),'shanghai_real_ood':len(sh),'unknown_tts':len(unknown)}}

# 评估模型：Cantonese train expressions -> test correction
eval_model=DTWControlModel.fit(control_train, ex, threshold=0.68, max_frames=max_frames)
test_preds,_=predict(eval_model, control_test)
ytest=np.array([r.command for r in control_test]); ypred=np.array([p.command for p in test_preds]); yconf=np.array([p.confidence for p in test_preds])
result['yue_control']={'val_accuracy':None,'test_accuracy':classification_metrics(ytest,ypred)['accuracy'],
    'test_macro_f1':classification_metrics(ytest,ypred)['macro_f1'],
    'known_coverage':float(np.mean(yconf>=eval_model.threshold)),
    'known_accepted_accuracy':float(np.mean((ytest==ypred)[yconf>=eval_model.threshold])) if np.any(yconf>=eval_model.threshold) else 0,
    'threshold':eval_model.threshold,
    'mean_latency_sec':float(np.mean([0])) if False else 0}
# latency separately
_,lat=predict(eval_model,control_test); result['yue_control']['mean_latency_sec']=lat
result['unknown_tts_rejection']=ood_accept(eval_model,unknown)
if sh:
    result['shanghai_real_ood_rejection']=ood_accept(eval_model,sh)
    result['shanghai_real_ood_rejection']['note']='OOD conversational speech; no command labels'
# Mandarin cross-language: forced predictions, acceptance, accepted accuracy
mpreds,mlat=predict(eval_model,mandarin); mconf=np.array([p.confidence for p in mpreds]); mtrue=np.array([r.command for r in mandarin]); mpred=np.array([p.command for p in mpreds]); macc=mconf>=eval_model.threshold
result['mandarin_cross_language']={'num':len(mandarin),'forced_accuracy':float(np.mean(mtrue==mpred)),'coverage':float(np.mean(macc)),'accepted_accuracy':float(np.mean((mtrue==mpred)[macc])) if macc.any() else 0.0,'mean_confidence':float(np.mean(mconf)),'mean_latency_sec':float(mlat)}

# 混合训练：粤语训练 vs 粤语+普通话音色训练，测试均为粤语test
yue_only=DTWControlModel.fit(train, ex, threshold=0.0, max_frames=max_frames)
mixed=DTWControlModel.fit(train+mandarin, ex, threshold=0.0, max_frames=max_frames)
result['mixed_training']={'yue_only_test':top1_metrics(yue_only,test), 'yue_plus_mandarin_test':top1_metrics(mixed,test), 'yue_only_templates':sum(len(v) for v in yue_only.templates.values()), 'mixed_templates':sum(len(v) for v in mixed.templates.values())}
# latency for mixed on test
_,lat_mixed=predict(mixed,test); result['mixed_training']['mixed_mean_latency_sec']=lat_mixed
_,lat_yue=predict(yue_only,test); result['mixed_training']['yue_mean_latency_sec']=lat_yue

out=base/'outputs'/'multidialect_feasibility.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2)); print('saved',out)

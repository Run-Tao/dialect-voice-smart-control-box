# -*- coding: utf-8 -*-
"""从HF datasets-server取少量上海话真实语音作为OOD拒识测试，不做命令训练。"""
import json, urllib.request, urllib.parse
from pathlib import Path
import soundfile as sf
import numpy as np

base=Path(r'D:\Run_Tao\DialectVoice_Research')
out=base/'data'/'tts_multidialect'/'shanghai_real_ood'
raw=out/'raw'; wav16=out/'wav16k'
raw.mkdir(parents=True,exist_ok=True); wav16.mkdir(parents=True,exist_ok=True)
dataset='TingChen-ppmc/Shanghai_Dialect_Conversational_Speech_Corpus'
url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':dataset,'config':'default','split':'train','offset':0,'length':20})
with urllib.request.urlopen(url, timeout=60) as f:
    data=json.load(f)
rows=[]
for r in data.get('rows',[]):
    row=r['row']
    audio=row.get('audio')
    # 数据集服务通常返回 [{"src": "..."}]
    if isinstance(audio,list) and audio:
        src=audio[0].get('src') or audio[0].get('url')
    elif isinstance(audio,dict):
        src=audio.get('src') or audio.get('url')
    else:
        src=None
    if not src: continue
    idx=r['row_idx']
    raw_path=raw/f'sh_{idx:04d}.wav'
    if not raw_path.exists():
        urllib.request.urlretrieve(src, raw_path)
    audio_arr,sr=sf.read(str(raw_path),dtype='float32',always_2d=False)
    if audio_arr.ndim>1: audio_arr=audio_arr.mean(axis=1)
    if sr!=16000:
        n=int(round(len(audio_arr)*16000/sr))
        audio_arr=np.interp(np.linspace(0,1,n,endpoint=False),np.linspace(0,1,len(audio_arr),endpoint=False),audio_arr).astype(np.float32)
    dst=wav16/f'sh_{idx:04d}.wav'
    sf.write(str(dst),audio_arr,16000,subtype='PCM_16')
    rows.append({'utterance_id':f'sh_{idx:04d}','audio_path':str(dst.relative_to(out)).replace('\\','/'),
                 'speaker_id':row.get('speaker_id',''),'transcription':row.get('transcription',''),
                 'duration':len(audio_arr)/16000,'sample_rate':16000})
import csv
with open(out/'manifest.csv','w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('saved',len(rows),'samples to',out)

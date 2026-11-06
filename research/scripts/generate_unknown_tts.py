# -*- coding: utf-8 -*-
"""生成未知命令TTS，用于拒识评估。"""
import asyncio
import csv
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import edge_tts
from canto_tts import CantoTTS

base = Path(r'D:\Run_Tao\DialectVoice_Research')
out = base / 'data' / 'tts_unknown'
raw = out / 'raw'; wav = out / 'wav16k'
raw.mkdir(parents=True, exist_ok=True); wav.mkdir(parents=True, exist_ok=True)
phrases = [
    '開冷氣', '閂電視', '播音樂', '今日天氣點樣',
    '較大聲啲', '開熱水器', '幫我訂外賣', '隨機講嘢'
]
voices = ['zh-HK-HiuGaaiNeural', 'zh-HK-HiuMaanNeural']
rows=[]

def convert(src, dst):
    audio, sr = sf.read(str(src), dtype='float32', always_2d=False)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        n = int(round(len(audio)*16000/sr))
        audio = np.interp(np.linspace(0,1,n,endpoint=False), np.linspace(0,1,len(audio),endpoint=False), audio).astype(np.float32)
    sf.write(str(dst), audio, 16000, subtype='PCM_16')
    return sf.info(str(dst))

# edge voices
for vi, voice in enumerate(voices, start=1):
    for pi, text in enumerate(phrases, start=1):
        stem = f'UNKNOWN_edge_{voice}_{pi:02d}'
        rp = raw / (stem + '.mp3')
        dp = wav / (stem + '.wav')
        t0=time.perf_counter()
        try:
            asyncio.run(edge_tts.Communicate(text, voice).save(str(rp)))
            info = convert(rp, dp)
            status='ok'; err=''
        except Exception as exc:
            status='fail'; err=repr(exc); info=None
        rows.append({'utt_id':stem,'intent_id':'UNKNOWN','text':text,'source':'edge_tts_unknown',
                     'speaker_id':voice,'audio_path':str(dp),'duration':info.duration if info else '',
                     'sample_rate':info.samplerate if info else '','status':status,'error':err,
                     'generation_sec':round(time.perf_counter()-t0,3)})
        print(stem,status)

# canto tts
tts=CantoTTS()
for pi, text in enumerate(phrases, start=1):
    stem=f'UNKNOWN_canto_{pi:02d}'
    rp=raw/(stem+'.wav'); dp=wav/(stem+'.wav')
    t0=time.perf_counter()
    try:
        tts.synthesize(text, str(rp))
        info=convert(rp, dp); status='ok'; err=''
    except Exception as exc:
        status='fail'; err=repr(exc); info=None
    rows.append({'utt_id':stem,'intent_id':'UNKNOWN','text':text,'source':'canto_tts_unknown',
                 'speaker_id':'canto-tts-default','audio_path':str(dp),
                 'duration':info.duration if info else '','sample_rate':info.samplerate if info else '',
                 'status':status,'error':err,'generation_sec':round(time.perf_counter()-t0,3)})
    print(stem,status)

log=base/'outputs'/'unknown_tts_log.csv'
with open(log,'w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
manifest=out/'manifest.csv'
with open(manifest,'w',encoding='utf-8-sig',newline='') as f:
    fields=['utterance_id','audio_path','speaker_id','dialect','command','language','text','expression_type','semantic_pair_id','source','is_synthetic','duration','sample_rate','split_group','text_verified_by','quality_flags','notes']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows:
        if r['status']!='ok': continue
        w.writerow({'utterance_id':r['utt_id'],'audio_path':'wav16k/'+Path(r['audio_path']).name,
                    'speaker_id':r['speaker_id'],'dialect':'yue-HK','command':'UNKNOWN',
                    'language':'yue','text':r['text'],'expression_type':'unknown',
                    'semantic_pair_id':r['utt_id'],'source':r['source'],'is_synthetic':'true',
                    'duration':r['duration'],'sample_rate':r['sample_rate'],
                    'split_group':'unknown','text_verified_by':'PENDING_NATIVE_SPEAKER',
                    'quality_flags':'SYNTHETIC;UNVERIFIED_TEXT;UNKNOWN_INTENT',
                    'notes':'unknown command for rejection only'})
print('log',log); print('manifest',manifest)

import json,csv
from pathlib import Path
import soundfile as sf, numpy as np
base=Path(r'D:\Run_Tao\DialectVoice_Research\data\tts_multidialect\shanghai_real_ood')
rows=json.loads((base/'rows.json').read_text(encoding='utf-8-sig'))['rows']
out=base/'wav16k'; out.mkdir(exist_ok=True)
meta=[]
for r in rows:
    idx=r['row_idx']; src=base/'raw'/f'sh_{idx:04d}.wav'
    if not src.exists(): continue
    a,sr=sf.read(str(src),dtype='float32',always_2d=False)
    if a.ndim>1: a=a.mean(axis=1)
    if sr!=16000:
        n=int(round(len(a)*16000/sr)); a=np.interp(np.linspace(0,1,n,endpoint=False),np.linspace(0,1,len(a),endpoint=False),a).astype(np.float32)
    dst=out/f'sh_{idx:04d}.wav'; sf.write(str(dst),a,16000,subtype='PCM_16')
    meta.append({'utterance_id':f'sh_{idx:04d}','audio_path':str(dst.relative_to(base)).replace('\\','/'),'speaker_id':r['row'].get('speaker_id',''),'transcription':r['row'].get('transcription',''),'duration':len(a)/16000,'sample_rate':16000})
with open(base/'manifest.csv','w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(meta[0].keys())); w.writeheader(); w.writerows(meta)
print(len(meta))

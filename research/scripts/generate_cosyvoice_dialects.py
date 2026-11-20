# -*- coding: utf-8 -*-
"""通过CosyVoice3官方Space生成上海/四川话小样本。需人工核验方言质量。"""
import argparse, csv, time
from pathlib import Path
import numpy as np
import soundfile as sf
from gradio_client import Client, handle_file

base=Path(r'D:\Run_Tao\DialectVoice_Research')
expr_csv=base/'data'/'corpus'/'core_six_expressions.csv'
out_root=base/'data'/'tts_multidialect'
log_path=base/'outputs'/'cosyvoice_multidialect_log.csv'
log_path.parent.mkdir(parents=True,exist_ok=True)
INSTRUCT={
 'shanghai':'You are a helpful assistant. 请用上海话表达。<|endofprompt|>',
 'sichuan':'You are a helpful assistant. 请用四川话表达。<|endofprompt|>',
}
PROMPTS=[
 ('xiaoxiao', base/'data'/'tts_corpus'/'wav16k'/'edge_mandarin'/'LIGHT_ON_direct_zh-CN-XiaoxiaoNeural.wav'),
 ('yunxi', base/'data'/'tts_corpus'/'wav16k'/'edge_mandarin'/'LIGHT_ON_direct_zh-CN-YunxiNeural.wav'),
]

def convert(src,dst):
    audio,sr=sf.read(str(src),dtype='float32',always_2d=False)
    if audio.ndim>1: audio=audio.mean(axis=1)
    if sr!=16000:
        n=int(round(len(audio)*16000/sr))
        audio=np.interp(np.linspace(0,1,n,endpoint=False),np.linspace(0,1,len(audio),endpoint=False),audio).astype(np.float32)
    sf.write(str(dst),audio,16000,subtype='PCM_16')
    return sf.info(str(dst))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dialect', choices=['shanghai','sichuan'], required=True)
    ap.add_argument('--max-calls', type=int, default=12)
    args=ap.parse_args()
    dialect=args.dialect
    instr=INSTRUCT[dialect]
    out_dir=out_root/dialect/'wav16k'; out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    with open(expr_csv,'r',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['language']=='cmn' and r['expression_type'] in ('direct','colloquial','polite'):
                rows.append(r)
    client=Client('https://funaudiollm-fun-cosyvoice3-0-5b.hf.space')
    calls=0
    logs=[]
    for r in rows:
        for prompt_name,prompt_path in PROMPTS:
            stem=f"{dialect}_{r['intent_id']}_{r['expression_type']}_{prompt_name}"
            dst=out_dir/(stem+'.wav')
            if dst.exists():
                continue
            if calls >= args.max_calls:
                break
            t0=time.perf_counter(); err=''; status='fail'
            try:
                res=client.predict(r['text'],'instruct','',handle_file(str(prompt_path)),None,instr,0,False,'Zh',api_name='/generate_audio')
                src=Path(res) if isinstance(res,str) else Path(res.get('path',''))
                if not src.exists(): raise FileNotFoundError(str(src))
                info=convert(src,dst); status='ok'
            except Exception as exc:
                err=repr(exc); info=None
            elapsed=time.perf_counter()-t0; calls+=1
            log={'utt_id':stem,'dialect':dialect,'language':'zh','text':r['text'],
                 'intent_id':r['intent_id'],'expression_type':r['expression_type'],
                 'semantic_pair_id':r['semantic_pair_id'],'source':'cosyvoice3_space_instruct',
                 'speaker_id':f'cosyvoice_prompt_{prompt_name}','audio_path':str(dst),
                 'duration':info.duration if info else '','sample_rate':info.samplerate if info else '',
                 'status':status,'error':err,'generation_sec':round(elapsed,2),
                 'text_verified_by':'PENDING_NATIVE_SPEAKER',
                 'quality_flags':'SYNTHETIC_DIALECT_INSTRUCT;UNVERIFIED_DIALECT_QUALITY'}
            logs.append(log); print(status,stem,round(elapsed,2),err)
        if calls>=args.max_calls: break
    # append log
    exists=log_path.exists()
    with open(log_path,'a',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(logs[0].keys()) if logs else [
            'utt_id','dialect','language','text','intent_id','expression_type','semantic_pair_id',
            'source','speaker_id','audio_path','duration','sample_rate','status','error','generation_sec',
            'text_verified_by','quality_flags'])
        if not exists: w.writeheader()
        w.writerows(logs)
    # rebuild combined manifest from all successful per-dialect files and log rows
    all_rows=[]
    if log_path.exists():
        with open(log_path,'r',encoding='utf-8-sig') as f:
            all_rows=list(csv.DictReader(f))
    man=out_root/'manifest.csv'
    with open(man,'w',encoding='utf-8-sig',newline='') as f:
        fields=['utterance_id','audio_path','speaker_id','dialect','command','language','text','expression_type','semantic_pair_id','source','is_synthetic','duration','sample_rate','split_group','text_verified_by','quality_flags','notes']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in all_rows:
            if r.get('status')!='ok': continue
            apath=Path(r['audio_path'])
            if not apath.exists(): continue
            rel=apath.relative_to(out_root)
            w.writerow({'utterance_id':r['utt_id'],'audio_path':str(rel).replace('\\','/'),'speaker_id':r['speaker_id'],
                        'dialect':r['dialect'],'command':r['intent_id'],'language':'zh','text':r['text'],
                        'expression_type':r['expression_type'],'semantic_pair_id':r['semantic_pair_id'],
                        'source':r['source'],'is_synthetic':'true','duration':r['duration'],'sample_rate':r['sample_rate'],
                        'split_group':r['source']+':'+r['speaker_id'],'text_verified_by':r['text_verified_by'],
                        'quality_flags':r['quality_flags'],'notes':'CosyVoice3 instruct dialect; native quality pending'})
    print('manifest',man)
if __name__=='__main__': main()

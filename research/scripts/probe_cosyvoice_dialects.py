# -*- coding: utf-8 -*-
from gradio_client import Client, handle_file
from pathlib import Path
import time, json
base=Path(r'D:\Run_Tao\DialectVoice_Research')
out=base/'data'/'tts_cosyvoice_probe'; out.mkdir(parents=True,exist_ok=True)
prompt=str(base/'data'/'tts_corpus'/'wav16k'/'canto'/'LIGHT_ON_direct_canto-tts-default.wav')
c=Client('https://funaudiollm-fun-cosyvoice3-0-5b.hf.space')
items=[
 ('shanghai','You are a helpful assistant. 请用上海话表达。<|endofprompt|>','开灯'),
 ('sichuan','You are a helpful assistant. 请用四川话表达。<|endofprompt|>','开灯'),
]
rows=[]
for name,instr,text in items:
 t0=time.perf_counter()
 try:
  res=c.predict(text,'instruct','',handle_file(prompt),None,instr,0,False,'Zh',api_name='/generate_audio')
  p=Path(res) if isinstance(res,str) else Path(res.get('path',''))
  dst=out/f'{name}_test.wav'
  if p.exists(): dst.write_bytes(p.read_bytes())
  status='ok' if dst.exists() else 'fail'
  err=''
 except Exception as exc:
  status='fail'; err=repr(exc); dst=''
 rows.append({'dialect':name,'text':text,'instruct':instr,'status':status,'error':err,'path':str(dst),'elapsed':round(time.perf_counter()-t0,2)})
 print(json.dumps(rows[-1],ensure_ascii=False))
(base/'outputs'/'cosyvoice_dialect_probe.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')


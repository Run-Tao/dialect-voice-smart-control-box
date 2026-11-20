from gradio_client import Client, handle_file
from pathlib import Path
import json
base=Path(r'D:\Run_Tao\DialectVoice_Research'); prompt=str(base/'data'/'tts_corpus'/'wav16k'/'edge_mandarin'/'LIGHT_ON_direct_zh-CN-XiaoxiaoNeural.wav'); c=Client('https://funaudiollm-fun-cosyvoice3-0-5b.hf.space')
try:
 res=c.predict('开灯','instruct','',handle_file(prompt),None,'You are a helpful assistant. 请用四川话表达。<|endofprompt|>',0,False,'Zh',api_name='/generate_audio'); print('OK',res)
except Exception as e: print('FAIL',repr(e))

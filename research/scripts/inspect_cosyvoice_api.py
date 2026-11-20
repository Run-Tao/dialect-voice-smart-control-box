import json
from gradio_client import Client
c=Client('https://funaudiollm-fun-cosyvoice3-0-5b.hf.space')
info=json.loads(c.view_api(return_format='json'))
for ep in info.get('named_endpoints',{}):
    if 'generate_audio' in ep:
        print(json.dumps(info['named_endpoints'][ep], ensure_ascii=False, indent=2))

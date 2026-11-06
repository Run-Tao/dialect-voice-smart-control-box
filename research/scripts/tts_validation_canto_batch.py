# -*- coding: utf-8 -*-
"""canto-tts 粤语小样本验证。候选文本必须由母语者核验。"""
import csv
import time
from pathlib import Path
import soundfile as sf
from canto_tts import CantoTTS

base = Path(r'D:\Run_Tao\DialectVoice_Research')
out_dir = base / 'data' / 'tts_validation' / 'canto'
out_dir.mkdir(parents=True, exist_ok=True)
log_path = base / 'outputs' / 'tts_canto_log.csv'
log_path.parent.mkdir(parents=True, exist_ok=True)

items = [
    ('LIGHT_ON', 'direct', '開燈'),
    ('LIGHT_ON', 'polite', '唔該幫我開燈'),
    ('LIGHT_OFF', 'direct', '熄燈'),
    ('LIGHT_OFF', 'polite', '唔該幫我熄燈'),
    ('CURTAIN_OPEN', 'direct', '打開窗簾'),
    ('CURTAIN_OPEN', 'polite', '唔該幫我拉開窗簾'),
    ('CURTAIN_CLOSE', 'direct', '拉埋窗簾'),
    ('CURTAIN_CLOSE', 'polite', '唔該幫我拉埋窗簾'),
    ('ALL_OFF', 'direct', '全部熄晒'),
    ('ALL_OFF', 'polite', '唔該全部關掉'),
    ('STATUS_ALL', 'direct', '而家咩狀態'),
    ('STATUS_ALL', 'polite', '可唔可以講下而家咩情況'),
]

t_init=time.perf_counter()
tts = CantoTTS()
init_sec = time.perf_counter()-t_init
rows=[]
for i,(intent,expr,text) in enumerate(items, start=1):
    out = out_dir / f'{intent}_{expr}_{i:02d}.wav'
    t0=time.perf_counter()
    tts.synthesize(text, str(out))
    elapsed=time.perf_counter()-t0
    info=sf.info(str(out))
    rows.append({
        'utt_id': f'canto_{i:02d}', 'intent_id': intent,
        'expression_type': expr, 'text': text, 'voice': 'canto-tts-default',
        'audio_path': str(out), 'language': 'yue', 'dialect': 'yue-HK',
        'is_synthetic': 'true', 'source': 'canto_tts',
        'sample_rate': info.samplerate, 'duration_sec': info.duration,
        'frames': info.frames, 'generation_sec': round(elapsed,3),
        'init_sec': round(init_sec,3),
        'status': 'ok' if out.exists() and out.stat().st_size > 0 else 'fail',
        'text_verified_by': 'PENDING_NATIVE_SPEAKER',
        'notes': '候选粤语文本，未由母语者核验；CPU-first ONNX'
    })
    print(rows[-1]['utt_id'], rows[-1]['status'], text, info.samplerate, info.duration, round(elapsed,3))

with open(log_path,'w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('日志:', log_path)

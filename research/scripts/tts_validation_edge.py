# -*- coding: utf-8 -*-
"""edge-tts 粤语小样本验证。候选粤语文本必须由母语者核验。"""
import asyncio
import csv
import time
from pathlib import Path
import edge_tts
import soundfile as sf

base = Path(r'D:\Run_Tao\DialectVoice_Research')
out_dir = base / 'data' / 'tts_validation' / 'edge'
out_dir.mkdir(parents=True, exist_ok=True)
log_path = base / 'outputs' / 'tts_edge_log.csv'
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
# 额外两条男声对照，不改变12条主测试集
extra = [
    ('LIGHT_ON', 'direct_male', '開燈', 'zh-HK-WanLungNeural'),
    ('STATUS_ALL', 'direct_male', '而家咩狀態', 'zh-HK-WanLungNeural'),
]

rows = []
async def synth(text, voice, path):
    comm = edge_tts.Communicate(text, voice)
    await comm.save(str(path))

def record(intent, expr, text, voice, idx, extra_flag=False):
    out = out_dir / f'{intent}_{expr}_{idx}.mp3'
    t0 = time.perf_counter()
    asyncio.run(synth(text, voice, out))
    elapsed = time.perf_counter() - t0
    try:
        info = sf.info(str(out))
        sr, duration, frames = info.samplerate, info.duration, info.frames
    except Exception as exc:
        sr, duration, frames = '', '', ''
        print('soundfile读取失败:', out, exc)
    rows.append({
        'utt_id': f'edge_{idx:02d}', 'intent_id': intent,
        'expression_type': expr, 'text': text, 'voice': voice,
        'audio_path': str(out), 'language': 'yue', 'dialect': 'yue-HK',
        'is_synthetic': 'true', 'source': 'edge_tts',
        'sample_rate': sr, 'duration_sec': duration, 'frames': frames,
        'generation_sec': round(elapsed, 3),
        'status': 'ok' if out.exists() and out.stat().st_size > 0 else 'fail',
        'extra_voice_comparison': extra_flag,
        'text_verified_by': 'PENDING_NATIVE_SPEAKER',
        'notes': '候选粤语文本，未由母语者核验；在线服务，非官方API'
    })
    print(rows[-1]['utt_id'], rows[-1]['status'], text, sr, duration, round(elapsed, 3))

for i, (intent, expr, text) in enumerate(items, start=1):
    record(intent, expr, text, 'zh-HK-HiuMaanNeural', i)
for j, (intent, expr, text, voice) in enumerate(extra, start=13):
    record(intent, expr, text, voice, j, True)

with open(log_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print('日志:', log_path)

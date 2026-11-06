# -*- coding: utf-8 -*-
"""使用同一模拟设备连续执行六类命令。演示音频来自TTS训练分布，不是泛化评估。"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.model_store import load_bundle
from src.control_server import AudioControlServer, MockControlDevice

base = Path(r'D:\Run_Tao\DialectVoice_Research')
cfg = load_config(base / 'config.json')
bundle = load_bundle(base / 'data' / 'models' / 'tts_control_model_demo.pkl')
log = base / 'outputs' / 'control_demo_log.jsonl'
if log.exists(): log.unlink()
server = AudioControlServer(bundle, cfg['features'], MockControlDevice(), log)
files = [
    'LIGHT_ON_polite_canto-tts-default.wav',
    'LIGHT_OFF_direct_canto-tts-default.wav',
    'CURTAIN_OPEN_direct_canto-tts-default.wav',
    'CURTAIN_CLOSE_direct_canto-tts-default.wav',
    'ALL_OFF_direct_canto-tts-default.wav',
    'STATUS_ALL_direct_canto-tts-default.wav',
]
for seq, name in enumerate(files, start=1):
    path = base / 'data' / 'tts_corpus' / 'wav16k' / 'canto' / name
    result = server.process(path, seq)
    print(seq, name, '->', result['predicted_intent'],
          'conf=', round(result['confidence'], 3),
          'accepted=', result['accepted'],
          'state=', result['device_state'])
print('final_state=', server.device.state)
print('log=', log)

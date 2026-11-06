# -*- coding: utf-8 -*-
"""DTW版端到端模拟控制演示。音频来自TTS训练分布，不是泛化评估。"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.dtw_model import DTWControlModel
from src.control_server import DTWControlServer, MockControlDevice

base = Path(r'D:\Run_Tao\DialectVoice_Research')
cfg = load_config(base / 'config.json')
model = DTWControlModel.load(base / 'data' / 'models' / 'tts_control_model_dtw_demo.pkl')
log = base / 'outputs' / 'dtw_control_demo_log.jsonl'
if log.exists(): log.unlink()
server = DTWControlServer(model, cfg['features'], MockControlDevice(), log)
files = [
    'LIGHT_ON_polite_canto-tts-default.wav',
    'LIGHT_OFF_direct_canto-tts-default.wav',
    'CURTAIN_OPEN_direct_canto-tts-default.wav',
    'CURTAIN_CLOSE_direct_canto-tts-default.wav',
    'ALL_OFF_direct_canto-tts-default.wav',
    'STATUS_ALL_direct_canto-tts-default.wav',
]
for seq, name in enumerate(files, start=1):
    result = server.process(base / 'data' / 'tts_corpus' / 'wav16k' / 'canto' / name, seq)
    print(seq, result['predicted_intent'], 'conf=', round(result['confidence'],3),
          'accepted=', result['accepted'], 'state=', result['device_state'])
print('final_state=', server.device.state)
print('log=', log)

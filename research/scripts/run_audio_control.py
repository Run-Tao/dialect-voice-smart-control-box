# -*- coding: utf-8 -*-
"""音频文件 -> 识别/拒识 -> 模拟设备控制。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.model_store import load_bundle
from src.control_server import AudioControlServer, MockControlDevice

base = Path(r'D:\Run_Tao\DialectVoice_Research')
parser = argparse.ArgumentParser()
parser.add_argument('--audio', required=True)
parser.add_argument('--model', default=str(base / 'data' / 'models' / 'tts_control_model.pkl'))
parser.add_argument('--seq', type=int, default=1)
parser.add_argument('--log', default=str(base / 'outputs' / 'control_log.jsonl'))
args = parser.parse_args()
cfg = load_config(base / 'config.json')
bundle = load_bundle(args.model)
server = AudioControlServer(bundle, cfg['features'], MockControlDevice(), args.log)
result = server.process(args.audio, args.seq)
print(json.dumps(result, ensure_ascii=False, indent=2))

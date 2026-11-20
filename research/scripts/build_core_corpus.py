# -*- coding: utf-8 -*-
"""生成核心六类意图定义、表达模板和TTS manifest样例。"""
import csv
from pathlib import Path

base = Path(r'D:\Run_Tao\DialectVoice_Research')
out_dir = base / 'data' / 'corpus'
out_dir.mkdir(parents=True, exist_ok=True)

intents = [
    ('LIGHT_ON', '开灯', '開燈', 'lighting', 'LED_ON', 'yes'),
    ('LIGHT_OFF', '关灯', '熄燈', 'lighting', 'LED_OFF', 'yes'),
    ('CURTAIN_OPEN', '开窗帘', '打開窗簾', 'curtain', 'SERVO_OPEN', 'yes'),
    ('CURTAIN_CLOSE', '关窗帘', '拉埋窗簾', 'curtain', 'SERVO_CLOSE', 'yes'),
    ('ALL_OFF', '全部关闭', '全部熄晒', 'combination', 'ALL_OFF', 'yes'),
    ('STATUS_ALL', '查询设备状态', '而家咩狀態', 'status', 'STATUS', 'yes'),
]
with open(out_dir / 'core_six_intents.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['intent_id', 'mandarin', 'cantonese_candidate', 'intent_group', 'hardware_action', 'in_mvp'])
    w.writerows(intents)

exprs = {
    'LIGHT_ON': {
        'direct': ('开灯', '開燈', 'clear', 'true', 'false', 'false'),
        'colloquial': ('把灯打开', '幫我開燈', 'clear', 'true', 'false', 'false'),
        'polite': ('麻烦开一下灯', '唔該幫我開燈', 'clear', 'true', 'false', 'false'),
        'question': ('能不能开灯？', '可唔可以開燈？', 'clear', 'true', 'false', 'false'),
        'correction': ('不是关灯，是开灯', '唔係熄燈，係開燈', 'clear', 'true', 'false', 'true'),
    },
    'LIGHT_OFF': {
        'direct': ('关灯', '熄燈', 'clear', 'true', 'false', 'false'),
        'colloquial': ('把灯关掉', '幫我熄燈', 'clear', 'true', 'false', 'false'),
        'polite': ('请关一下灯', '唔該幫我熄燈', 'clear', 'true', 'false', 'false'),
        'question': ('灯能关一下吗？', '可唔可以熄燈？', 'clear', 'true', 'false', 'false'),
        'correction': ('不是开灯，是关灯', '唔係開燈，係熄燈', 'clear', 'true', 'false', 'true'),
    },
    'CURTAIN_OPEN': {
        'direct': ('开窗帘', '打開窗簾', 'clear', 'true', 'false', 'false'),
        'colloquial': ('把窗帘拉开', '幫我拉開窗簾', 'clear', 'true', 'false', 'false'),
        'polite': ('麻烦打开窗帘', '唔該幫我打開窗簾', 'clear', 'true', 'false', 'false'),
        'question': ('窗帘能打开吗？', '可唔可以打開窗簾？', 'clear', 'true', 'false', 'false'),
        'correction': ('不是关窗帘，是开窗帘', '唔係拉埋窗簾，係打開窗簾', 'clear', 'true', 'false', 'true'),
    },
    'CURTAIN_CLOSE': {
        'direct': ('关窗帘', '拉埋窗簾', 'clear', 'true', 'false', 'false'),
        'colloquial': ('把窗帘拉上', '幫我拉埋窗簾', 'clear', 'true', 'false', 'false'),
        'polite': ('请关一下窗帘', '唔該幫我拉埋窗簾', 'clear', 'true', 'false', 'false'),
        'question': ('窗帘能拉上吗？', '可唔可以拉埋窗簾？', 'clear', 'true', 'false', 'false'),
        'correction': ('不是开窗帘，是关窗帘', '唔係打開窗簾，係拉埋窗簾', 'clear', 'true', 'false', 'true'),
    },
    'ALL_OFF': {
        'direct': ('全部关闭', '全部熄晒', 'clear', 'true', 'false', 'false'),
        'colloquial': ('把所有设备都关掉', '幫我全部閂晒', 'clear', 'true', 'false', 'false'),
        'polite': ('麻烦全部关闭', '唔該全部關掉', 'clear', 'true', 'false', 'false'),
        'question': ('能不能全部关掉？', '可唔可以全部熄晒？', 'clear', 'true', 'false', 'false'),
        'correction': ('不是只关灯，是全部关闭', '唔係淨係熄燈，係全部熄晒', 'clear', 'true', 'false', 'true'),
    },
    'STATUS_ALL': {
        'direct': ('查询设备状态', '而家咩狀態', 'clear', 'true', 'false', 'false'),
        'colloquial': ('看看现在设备什么情况', '睇下而家啲設備咩情況', 'clear', 'true', 'false', 'false'),
        'polite': ('麻烦告诉我设备状态', '唔該講下設備狀態', 'clear', 'true', 'false', 'false'),
        'question': ('现在设备是什么状态？', '而家啲設備係咩狀態？', 'clear', 'true', 'false', 'false'),
        'correction': ('不是问灯，是问所有设备状态', '唔係問燈，係問全部設備狀態', 'clear', 'true', 'false', 'true'),
    },
}

rows = []
for intent, variants in exprs.items():
    for expr_type, (zh, yue, clarity, executable, context, distractor) in variants.items():
        pair = f'{intent}_{expr_type}'
        for lang, dialect, text in [('cmn', 'mandarin', zh), ('yue', 'yue-HK', yue)]:
            rows.append({
                'utt_id': f'{pair}_{lang}_text',
                'audio_path': '',
                'language': lang,
                'dialect': dialect,
                'text': text,
                'intent_id': intent,
                'expression_type': expr_type,
                'semantic_pair_id': pair,
                'source': 'candidate_text',
                'speaker_id': '',
                'is_synthetic': 'false',
                'duration': '',
                'split_group': pair,
                'text_verified_by': 'PENDING_NATIVE_SPEAKER',
                'quality_flags': 'UNVERIFIED_TEXT',
                'semantic_clarity': clarity,
                'direct_executable': executable,
                'requires_context': context,
                'contains_distractor': distractor,
                'notes': '粤语文本为候选写法，必须母语者核验'
            })
with open(out_dir / 'core_six_expressions.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print('意图表:', out_dir / 'core_six_intents.csv')
print('表达表:', out_dir / 'core_six_expressions.csv', 'rows=', len(rows))

# 云端服务（PC 版方言智能家居引擎适配层）

本服务是「端云协同方言语音指令收录、转写与语义理解系统」的云端。
**识别与理解不重建模型**：直接复用 `D:\Projects\dialect-smart-home` 的
`dialect_home.pipeline.Pipeline`（SenseVoice-small 粤语 ASR → 粤语归一化 → 封闭域规则 NLU → 模拟执行器），
本目录只做三件事：HTTP 协议适配、WAV 语料存档、人工标注状态回写。

```
ESP32-S3 终端 ──multipart WAV──▶ POST /api/v1/speech/understand
                                    │  CloudEngine.understand()
                                    │    └─ dialect_home.Pipeline.process_wav()
                                    ▼
                              契约 JSON（intent/slots/confidence/reply_text）
ESP32-S3 终端 ──{sample_id,is_correct}──▶ POST /api/v1/speech/feedback
                                    ▼
                          dataset/metadata.jsonl 标注回写
```

## 启动

```bat
:: 方式一（推荐）
run_cloud.bat
:: 方式二
set PYTHONUTF8=1
D:\Projects\dialect-smart-home\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\Projects\dialect-smart-home\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

引擎依赖（funasr/onnxruntime/rapidfuzz/soundfile 等）随 PC 版工程安装；
`asr_nlu_pipeline.py` 会自动把 `../dialect-smart-home/src` 加入 `sys.path`，
也可 `pip install -e ../../dialect-smart-home` 做可编辑安装。
首次请求会加载 SenseVoice 模型（约 10~30s），`/healthz` 的 `engine_ready` 可查询状态。
云端不做本机 TTS 播报（注入 `Tts(enabled=False)`），避免每次设备上传时 PC 喇叭响。

## 端点与契约

### POST /api/v1/speech/understand（multipart/form-data）

| 字段 | 说明 |
|---|---|
| file | 16kHz/16bit/单声道 WAV |
| device_id | 终端编号，如 esp32-s3-001 |
| timestamp | 终端录音时刻 ISO8601 |
| sample_id | 可选；终端生成的样本号，缺省由服务端生成 |

响应（与终端 `network_client.h` 的 `understand_result_t` 一一对应）：

```json
{
  "sample_id": "20261120_132601_001",
  "dialect": "cantonese",
  "raw_text": "帮我开着厅嗰盏灯",
  "normalized_text": "帮我打开客厅的灯",
  "intent": "device_control",
  "slots": {"location": "living_room", "device": "light", "action": "on"},
  "confidence": 0.95,
  "reply_text": "好嘅，已经帮你开咗客厅灯"
}
```

- `intent=device_control` 且 `confidence>=0.7` 时终端才执行 GPIO（门槛在终端侧 `main.c`）。
- 未识别到指令：`intent="unknown"`、`confidence=0.0`、`reply_text` 为粤语致歉句。
- `slots` 除 location/device/action 外保留数值槽（`delta`/`temperature`）。
- `action` 词表：on/off/brighter/dimmer/temp_up/temp_down/set_temp/open/close/
  speed_up/speed_down/channel_up/channel_down/volume_up/volume_down。

### POST /api/v1/speech/feedback（application/json）

```json
{"sample_id": "20261120_132601_001", "is_correct": true}
```

可选 `corrected_text`（识别有误时的人工纠正文本）。服务端把该样本
`annotation.state` 由 `pending` 改为 `annotated`，写入 `is_correct/feedback_at`，
原子替换 `metadata.jsonl`（tmp + replace）。未知 sample_id 返回 404。

### GET /healthz

`{"status":"ok","engine_ready":true,"engine_root":"...","samples":N}`

## curl 自测

```bash
curl --noproxy '*' -s http://127.0.0.1:8000/healthz

curl --noproxy '*' -s -X POST http://127.0.0.1:8000/api/v1/speech/understand \
  -F "file=@../../dialect-smart-home/data/corpus/yue01.wav" \
  -F "device_id=esp32-s3-001" \
  -F "timestamp=2026-11-20T13:26:01Z" \
  -F "sample_id=20261120_132601_001"

curl --noproxy '*' -s -X POST http://127.0.0.1:8000/api/v1/speech/feedback \
  -H "Content-Type: application/json" \
  -d '{"sample_id":"20261120_132601_001","is_correct":true}'
```

## 数据集目录

```
dataset/
  audio/{sample_id}.wav        # 终端上传的原始录音，永久存档
  metadata.jsonl               # 每行一条：契约字段 + device_id/timestamp/received_at
                               #   + annotation{state: pending|annotated, is_correct,
                               #     corrected_text, feedback_at}
```

该数据集即 PC 版评测语料的真实增量来源：`annotated && is_correct=false` 的样本
可回流到 `dialect-smart-home` 的归一化词典/命令库别名修订。
# 方言智控盒第一阶段验证报告

日期：2026-10-18

## 1. 代码审查结论

- 现有核心模块可运行：manifest、split、audio_io、features、alignment、representations、baselines、classifiers、metrics、experiment、mapping。
- 现有测试：28项通过（含新增串口协议测试和manifest别名测试）。
- 已实现并验证：MFCC、固定长度、DTW、余弦/DTW基线、Softmax/线性SVM、OLS/Ridge、结果保存。
- 已实现但仅合成验证：OLS/Ridge、SVD R0/R1、映射实验。
- 尚未实现：真实ESP32固件、真实串口硬件闭环、实时麦克风入口、真实粤语数据、拒识校准、真实智能音箱接入。

## 2. 发现并修复的问题

1. TTS manifest 使用 `utt_id`/`intent_id`，而现有 `load_manifest` 只接受 `utterance_id`/`command`。
   - 修改：`src/manifest.py` 增加别名字段和可选文本/来源字段。
   - 影响：兼容原manifest和TTS manifest，不破坏现有测试。
2. 现有 `audio_io.py` 只支持PCM WAV。
   - 处理：TTS音频统一用 `soundfile` 转为16kHz单声道PCM WAV后再进入现有算法。
3. 基础Python环境没有 soundfile、edge-tts、canto-tts、pyserial。
   - 处理：新建隔离环境 `.venv_tts`，不修改原环境。

## 3. TTS验证

### edge-tts
- 版本：7.3.1
- 状态：可用
- 声音：zh-HK-HiuGaaiNeural、zh-HK-HiuMaanNeural、zh-HK-WanLungNeural
- 生成：12条主测试 + 2条男声对照
- 格式：MP3，24kHz
- 日志：`outputs/tts_edge_log.csv`
- 音频：`data/tts_validation/edge/`

### canto-tts
- 版本：0.1.4，Apache-2.0，CPU-first ONNX
- 状态：可用；首次需从HuggingFace下载模型
- 模型下载：本机需设置 `HF_ENDPOINT=https://hf-mirror.com` 才能成功
- 生成：12条
- 格式：WAV，48kHz
- 日志：`outputs/tts_canto_log.csv`
- 音频：`data/tts_validation/canto/`

### 未验证项
- 发音自然度、粤语特征、数字读法、多音字处理均未由粤语母语者试听。
- 所有粤语文本均为候选写法，`text_verified_by=PENDING_NATIVE_SPEAKER`。

## 4. 核心六类语料

- 意图表：`data/corpus/core_six_intents.csv`（6类）
- 表达表：`data/corpus/core_six_expressions.csv`（30个语义配对 × 2语言 = 60行）
- TTS manifest样例：`data/corpus/tts_manifest_sample.csv`（26条，含canto/edge）
- 已标注 `semantic_pair_id`、`expression_type`、`direct_executable`、`requires_context`、`contains_distractor`。

## 5. 最小算法冒烟

使用canto-tts的12条音频，按表达模板留出：
- direct训练、polite测试：分类器Accuracy=0.333，DTW Accuracy=0.167
- polite训练、direct测试：分类器Accuracy=0.333，DTW Accuracy=0.333

这些数值只说明流程可运行，样本仅12条且为合成语音，不能作为真实粤语识别性能。

## 6. 串口协议模拟

协议：
```text
CMD,<seq>,<intent_id>\n
ACK,<seq>,OK\n
ACK,<seq>,DUP\n
ERR,<seq>,CODE\n
STATE,<seq>,LIGHT=on,CURTAIN=open\n
```

模拟测试已覆盖：正常命令、状态查询、未知命令、格式错误、重复seq。
真实ESP32-S3硬件未接入，未验证实物串口和舵机供电。

## 7. 主要风险

1. 粤语文本未经母语者核验。
2. TTS到真人语音存在域差异。
3. 当前只有12条合成样本，算法结果无研究意义。
4. canto-tts模型需镜像下载，换机器可能失败。
5. edge-tts为在线非官方接口，存在可用性和条款风险。
6. 无ESP32实物，串口/供电/舵机压降未知。

## 8. 下一阶段优先级

1. 粤语母语者核验文本并试听TTS。
2. 采集3–5位真人粤语六类命令。
3. 按说话人/表达模板划分数据。
4. 用真实数据跑直接分类、DTW、拒识。
5. 接入ESP32-S3实物，验证ACK/心跳/舵机供电。
6. 只有在真实配对数据可用时再评估OLS/Ridge映射。

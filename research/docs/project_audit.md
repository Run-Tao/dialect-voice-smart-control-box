# 项目审计报告

日期：2026-10-18

## 1. 项目状态

- 当前目录不是git仓库，没有版本历史，无法做提交级追溯。
- 核心代码位于 `src/`，实验脚本位于 `scripts/`，测试位于 `tests/`。
- 数据清单 `data/tts_corpus/manifest.csv` 共180行，实际存在180个WAV文件。
- 实际执行 `python -m unittest discover -s tests -v`：40项通过，0失败，0跳过。
- 工作环境基础Python为3.12.14；TTS依赖在隔离环境 `.venv_tts`。

## 2. 数据审计结果

- 语言：粤语120条，普通话60条。
- 来源：canto-tts 30条，edge-tts粤语90条，edge-tts普通话60条。
- 六类意图各30条，五类表达各36条。
- 音频：全部16kHz单声道WAV，时长1.2到4.64秒，无空音频、无高静音、无削波。
- 精确重复文件：0。
- 同文本、同音色、同来源重复组：0。
- 近重复检查：同一文本不同音色的MFCC余弦大于0.999的样本对约210个。

## 3. 关键发现

1. 说话人隔离划分只隔离音色，没有隔离文本。同一文本在不同TTS音色中同时出现在训练和测试，构成文本级泄漏风险。
2. 表达模板划分隔离了表达类型，但训练和测试共享全部4个粤语音色，属于非说话人隔离评估。
3. 严格划分条件下（同时隔离音色和表达），训练36条、验证6条、测试6条，DTW测试Accuracy仅0.167，Macro-F1 0.067。该测试集很小，但说明现有0.625结果依赖已见音色和表达模板。
4. 评估模型阈值0.68，演示模型阈值0.90，二者是不同模型、不同用途；不是同一模型矛盾配置，但缺少统一配置说明。
5. 上海话只成功生成1条CosyVoice3样本；四川话无成功音频。上海真实语料20条仅可用于OOD拒识，不能作为命令监督数据。
6. 未接入真实ESP32-S3；串口结果均为模拟验证。

## 4. 证据与结果文件

| 项目 | 实际证据 |
|---|---|
| 数据清单 | `data/tts_corpus/manifest.csv` |
| 数据检查 | `outputs/data_quality_audit.json` |
| 说话人隔离DTW | `outputs/tts_yue_baselines_max60/dtw/result.json` |
| 控制模型指标 | `outputs/control_model_metrics.json` |
| DTW控制模型指标 | `outputs/dtw_control_metrics.json` |
| 拒识阈值权衡 | `outputs/rejection_metrics.json` |
| 严格划分实验 | `outputs/strict_split_experiment.json` |
| 跨语言/跨方言拒识 | `outputs/multidialect_crosslingual.json` |
| 混合训练代理 | `outputs/mixed_training_experiment.json` |
| 个性化注册原型 | `outputs/enrollment_prototype.json` |

## 5. 待解决问题

- 没有真人语音，所有命令识别指标均为合成TTS。
- 文本级泄漏风险尚未通过重新采集数据解决；只能通过严格划分和文档说明控制解释。
- 严格划分测试集仅6条，统计不确定性很大。
- 上海话和四川话缺少命令级数据。
- 真实ESP32-S3串口和舵机供电未验证。
- 项目无git，需要建立版本管理。

# 方言智控盒软件阶段报告

日期：2026-10-18

## 完成内容

- 统一语料管理：manifest读取、TTS生成日志、统计与重复检查。
- TTS批量生成：canto-tts本地粤语；edge-tts粤语/普通话；自动转16kHz WAV。
- 语料规模：180条TTS；粤语120条，普通话60条；6类意图各30条；5种表达类型各36条。
- 基线实验：cosine、DTW、Softmax、线性SVM、OLS、Ridge。
- 评估模型：说话人隔离基线；表达模板留出训练/验证/测试。
- 拒识：置信度阈值校准、覆盖率、接受准确率、未知命令误接受率。
- 端到端控制：音频→MFCC→分类→拒识→模拟ESP32→设备状态→JSONL日志。
- 串口协议：CMD/ACK/ERR/DUP/STATE；模拟测试通过。
- 测试：31项单元测试通过。

## 实际结果（仅合成TTS，不代表真实粤语性能）

### 说话人隔离基线（120条粤语TTS）

| 方法 | Accuracy | Macro-F1 |
|---|---:|---:|
| cosine | 0.2667 | 0.2238 |
| DTW | 0.3667 | 0.3609 |
| logreg | 0.1667 | 0.0490 |
| svm | 0.1667 | 0.0476 |
| OLS | 0.1333 | 0.0430 |
| Ridge | 0.1333 | 0.0430 |

### 评估模型（表达模板留出）

- train: direct/colloquial/polite
- val: question
- test: correction
- chosen model: softmax
- val accuracy: 0.5000
- test accuracy: 0.1667
- test macro-F1: 0.1267
- threshold: 0.50
- accepted coverage: 0.75
- accepted accuracy: 0.2222

### 拒识

- 已知测试集：24条
- 未知命令：24条
- 阈值0.50：已知覆盖率0.75，未知误接受率0.6667
- 保守阈值0.90：已知覆盖率0.125，未知误接受率0.1667

### 映射对照

- 数据：canto-tts粤语 ↔ edge-tts普通话Xiaoxiao，30个语义配对
- 句子级映射，不是逐帧映射
- OLS：因部分特征零方差失败（已记录）
- Ridge：accuracy=0.1667，macro-F1=0.0556
- 基线直接分类：accuracy=0.1667，macro-F1=0.0833

### 端到端演示

使用全部合成粤语TTS训练的演示模型，阈值0.90，依次执行：
LIGHT_ON, LIGHT_OFF, CURTAIN_OPEN, CURTAIN_CLOSE, ALL_OFF, STATUS_ALL。
6条全部正确识别并产生ACK，设备状态按预期更新。演示音频来自训练分布，不是泛化评估。

## 遗留问题

- 无真人粤语数据；所有结论仅限合成TTS。
- 粤语文本未由母语者核验。
- 跨说话人/跨表达泛化性能低。
- 拒识在低阈值下误接受率较高；高阈值覆盖率低。
- 无真实ESP32-S3硬件与串口实测。
- OLS在句子级特征上存在零方差问题；Ridge可运行但效果有限。

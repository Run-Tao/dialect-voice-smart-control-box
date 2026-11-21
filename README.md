# 方言智居 DialectHome

**智能家居方言语音指令转换器** —— 把粤语 / 带口音普通话的自然口语指令，
离线转换为智能家居结构化控制命令并语音反馈。本科工程导论课程项目 · 完整交付。

```
「唔该帮我熄咗客厅盏灯」 → {device: light, action: turn_off, room: 客厅} → 已关闭客厅的灯
「冷气较低两度」         → {device: ac, action: temp_down, delta: 2}    → 客厅空调温度已调低2度
```

## 核心设计

- **耳朵**：SenseVoice-Small（原生粤语 yue，非自回归，CPU 实时）+ FSMN-VAD；faster-whisper 备用后端
- **大脑**：繁简/粤语口语词归一化 + **拼音模糊域双通道理解**（n/l、zh/z、ang/an 等口音混淆归一），
  吸收 ASR 同音错字；命令库 YAML 外置，加设备/说法不改代码
- **手**：可插拔执行器 —— 虚拟家居模拟器（5 房间 × 6 设备）/ MQTT / Home Assistant
- **嘴**：pyttsx3 离线语音反馈（edge-tts 在线可选）
- **界面**：Web 浏览器录音（前端直接采 16k PCM 封 WAV，零 ffmpeg 依赖）+ CLI + Python API

## 实测指标（TTS 合成语料口径，40 条 = 26 粤语 + 14 普通话）

| 指标 | 结果 |
|---|---|
| 意图 / 房间 / 槽位 / 完全正确率 | **100% / 100% / 100% / 100%** |
| 端到端延迟 均值 / P95（i5-1240P CPU） | **0.73 s / 1.45 s** |
| 单元测试 | 51 passed |

> 口径声明：合成语料指标 ≠ 真实录音指标。真实录音验证规程见 [docs/05](docs/05-测试报告.md)。

## 快速开始

```bat
cd D:\Projects\dialect-smart-home
D:\anaconda3\envs\py_study\python.exe -m venv .venv
set PYTHONUTF8=1
.venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
.venv\Scripts\python.exe scripts\download_models.py
run_web.bat
```

浏览器打开 http://127.0.0.1:8000 ，点麦克风说粤语指令；或文本框直输调试。
命令行：`.venv\Scripts\python.exe src\dialect_home\cli.py --mic 5` / `--wav 文件` / `--text 文本`。
详见 [docs/06 使用手册](docs/06-使用手册.md)。

## 交付文档

| 文档 | 内容 |
|---|---|
| [01 方案研究与头脑风暴](docs/01-方案研究与头脑风暴.md) | 5 条路线对比、选型论证、调研来源 |
| [02 总体技术方案](docs/02-总体技术方案.md) | 架构、模块契约、理解层算法、延迟预算 |
| [03 硬件使用方案](docs/03-硬件使用方案.md) | 桌面级 ¥110 / 树莓派级 ¥675~810 两套方案、BOM、接线、部署 |
| [04 软件设计说明](docs/04-软件设计说明.md) | 模块细节、扩展指南 |
| [05 测试报告](docs/05-测试报告.md) | 实测数字、修复迭代、线程基准、浏览器实测、真实验证规程 |
| [06 使用手册](docs/06-使用手册.md) | 安装/使用/接真实家居/FAQ |
| [成本表.csv](docs/成本表.csv) | 三套方案 BOM |

## 目录结构

```
config/commands.yaml      命令库（设备/房间/动作/粤语别名/反馈模板）
config/settings.yaml      运行配置（ASR/理解阈值/执行器/TTS/服务）
src/dialect_home/         源码：audio·normalize·textutils·understand·pipeline·cli·server
src/dialect_home/asr/     SenseVoice / faster-whisper / mock 三后端
src/dialect_home/executor/ 模拟器 / MQTT / Home Assistant 三执行器
web/                      浏览器录音前端
scripts/                  模型下载 · 语料合成 · 端到端评测 · 线程基准
tests/                    51 条单元测试（零模型依赖）
data/corpus/              40 条合成语料 + manifest
data/results/             评测结果 · 转写 · 日志 · UI 截图
models/                   SenseVoiceSmall + fsmn-vad（已下载，可离线搬移）
```

## 复现评测

```bat
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\make_test_corpus.py --proxy http://127.0.0.1:7890
.venv\Scripts\python.exe scripts\run_eval.py --backend sensevoice
.venv\Scripts\python.exe scripts\bench_threads.py
```

## 后续工作

真实粤语录音采集与分条件评测（规程见 docs/05 §8）；唤醒词层；
其他方言（闽南语/四川话）归一化词表扩展；树莓派真机部署联调。

模型许可：SenseVoiceSmall Apache-2.0（ModelScope iic/SenseVoiceSmall）。
# 第三阶段可演示原型：语音控制服务 + Web 演示台 + CLI

本文档说明「方言智控盒」第三阶段（v3）可演示原型的启动方式、演示脚本、接口与
已知限制。原型可离线运行，设备为**模拟设备**，输入为 **TTS 合成测试音频**（不是
实时麦克风）。

## 1. 目标与闭环

演示目标是让用户实际看到完整链路：

```
TTS 测试音频 → 识别器（v3 CNN 主模型 / v2 DTW 回退）→ 阈值判定（接受 / 拒识）
            → 模拟设备状态机（灯 / 窗帘）→ 中文状态反馈 → JSONL 审计日志
```

页面与命令行都能直接看到：**识别命令、置信度、是否拒识、设备状态变化、操作日志、延迟**。

## 2. 新增文件

| 文件 | 作用 |
| --- | --- |
| `src/control_server_v3.py` | `MockDeviceV3` 六类命令状态机、`V3ControlServer` 控制服务、`load_predictor` 模型适配器 |
| `scripts/demo_v3.py` | CLI 演示：单条音频 / 六类命令自动序列 / 交互模式 |
| `scripts/run_web_demo.py` | Flask 本地演示台（默认 `127.0.0.1:8770`），JSON API |
| `web/demo_catalog.py` | 从测试划分 + 清单自动挑选未见表达测试音频与未知指令音频 |
| `web/page.py` | 演示台单页（CSS/JS 全部内联，无 CDN 依赖，离线可用） |
| `tests/test_v3_control.py` | 状态机、拒识、阈值、日志、模型适配、Web API、音频目录测试（35 项） |
| `docs/demo_v3.md` | 本文档 |
| `outputs/v3_demo_log.jsonl` | 演示操作日志（JSONL，运行时追加） |

不修改二期既有 `src/*.py`、`data/`、`config.json`、`README.md`、`docs/` 既有文档，
不引入新依赖（仅使用已安装的 Flask 3.1 与 numpy）。

## 3. 环境与依赖

- Python：`D:\anaconda3\python.exe`（3.13.9），在项目根目录 `D:\Run_Tao\DialectVoice_Research` 下运行。
- 已安装：`numpy`、`Flask 3.1.2`；无需 CDN/联网。
- PowerShell 下建议先设置 UTF-8：

```powershell
$env:PYTHONIOENCODING='utf-8'
```

## 4. 快速开始

### 4.1 命令行演示（推荐先跑，验证全链路）

```powershell
cd D:\Run_Tao\DialectVoice_Research
$env:PYTHONIOENCODING='utf-8'

# 六类命令各一条未见表达测试音频 + 一条未知指令（默认行为）
D:\anaconda3\python.exe scripts\demo_v3.py --sequence

# 单条音频
D:\anaconda3\python.exe scripts\demo_v3.py --audio data\tts_corpus\wav16k\canto\LIGHT_ON_polite_canto-tts-default.wav --show-scores

# 交互模式：逐条按 Enter 处理，输入 q 退出
D:\anaconda3\python.exe scripts\demo_v3.py --interactive

# 覆盖拒识阈值（v2 DTW demo 模型默认 0.9，调低可看到更多「接受」）
D:\anaconda3\python.exe scripts\demo_v3.py --sequence --threshold 0.5

# 使用第三阶段 v3 主模型（推荐：3 种子 CNN 集成 + 阈值 0.93）
D:\anaconda3\python.exe scripts\demo_v3.py --sequence --model data\models\v3_cnn_ensemble.npz
```

> **默认模型说明**：CLI 与 Web 的默认模型仍是二期 v2 DTW 演示模型
> `data/models/tts_control_model_dtw_demo.pkl`（阈值 0.9），这样不依赖 v3 工件也能跑通；
> **第三阶段主模型是 `data/models/v3_cnn_ensemble.npz`（同目录 `v3_thresholds.json`
> 给出阈值 0.93），必须显式加 `--model data\models\v3_cnn_ensemble.npz` 才会使用**。
> 两者可随时切换，演示台/控制服务代码无需改动。

CLI 会输出逐条事件流（音频来源/表达组/音色/文本、识别命令、置信度与阈值、
设备反馈、设备状态、延迟），以及汇总表格、最终设备状态和日志路径。

### 4.2 Web 演示台

```powershell
cd D:\Run_Tao\DialectVoice_Research
$env:PYTHONIOENCODING='utf-8'

# 默认模型 data/models/tts_control_model_dtw_demo.pkl（v2 DTW，阈值 0.9），监听 127.0.0.1:8770
D:\anaconda3\python.exe scripts\run_web_demo.py

# 指定端口 / 阈值
D:\anaconda3\python.exe scripts\run_web_demo.py --port 8771 --threshold 0.5

# 使用第三阶段 v3 主模型（3 种子 CNN 集成，阈值由同目录 v3_thresholds.json 提供）
D:\anaconda3\python.exe scripts\run_web_demo.py --model data\models\v3_cnn_ensemble.npz --port 8771
```

> 默认模型同样是二期 v2 DTW 演示模型；要用 v3 主模型演示，请显式指定
> `--model data\models\v3_cnn_ensemble.npz`。此时 `/api/health` 返回的
> `model_name` 为 `v3_cnn_ensemble(3seed,6class)`、`threshold` 为 `0.93`。

浏览器打开 `http://127.0.0.1:8770/` 即可，页面特性：

- 六个命令按钮（開燈 / 關燈 / 打開窗簾 / 關閉窗簾 / 全屋關閉 / 查詢全屋狀態），
  每个按钮对应一条来自 `track_a_expression_split.csv` **test 划分（未参与训练）**
  的 TTS 音频；
- 「未知指令（應拒識）」按钮：优先用 `data/tts_corpus/unknown_v3.csv`，缺失时回退
  `outputs/unknown_tts_log.csv` 或 `data/tts_unknown/`；
- 拒识阈值滑块：默认取模型阈值，每次点击按当前滑块值判定，可直接对比同一音频
  「接受 / 拒识」的差异；
- 识别结果卡：命令中文名、原始英文命令、置信度进度条、接受/拒识徽标、本次阈值、
  设备反馈、延迟、拒识原因；
- 模拟设备卡片（燈 / 窗簾）实时状态；
- 操作日志表（最近 10 条，自动刷新）与「重置設備狀態」按钮（重置不清审计日志）。

按 `Ctrl+C` 停止服务。演示结束后请关闭进程，不要长期占用端口。

## 5. 模型适配（可插拔识别器）

`src/control_server_v3.py` 中的 `load_predictor(path)` 按以下顺序加载：

1. **优先 v3 模型**：`src/v3_model.py` 已提供 `V3Predictor.load(path)`，`load_predictor`
   会优先使用。当前 v3 主模型为 `data/models/v3_cnn_ensemble.npz`（3 种子 1D-CNN 集成，
   纯 numpy 推理），`V3Predictor.load` 会自动读取同目录的 `v3_thresholds.json`
   （`selected_threshold=0.93`，温度标定 0.45）。接口只需实现 `predict(audio_path)`，
   返回对象含 `command` / `confidence` / `scores` / `distance` 中的任意子集（也接受
   字典或 `(command, confidence, scores)` 元组）。
2. **回退 v2 DTW pickle**：`DTWControlModel.save` 生成的模板 pickle，使用二期声学前端
   （`config.json` 的 features：MFCC + CMVN）提特征后做 DTW 模板匹配。

模型文件不存在、格式不识别时会给出清晰中文错误提示。`V3ControlServer` 的构造参数：

```python
V3ControlServer(predictor, threshold=None, device=None, log_path=None, model_name=None)
# threshold 缺省时使用 predictor.threshold；process(audio_path, seq=None, threshold=None)
# 的 threshold 可对单次调用覆盖。
```

## 6. 设备状态机语义

| 命令 | 灯（light） | 窗簾（curtain） | 中文反馈示例 |
| --- | --- | --- | --- |
| `LIGHT_ON` | 開 | 不變 | 已開燈 |
| `LIGHT_OFF` | 關 | 不變 | 已關燈 |
| `CURTAIN_OPEN` | 不變 | 開 | 窗簾已打開 |
| `CURTAIN_CLOSE` | 不變 | 關 | 窗簾已關閉 |
| `ALL_OFF` | 關 | 關 | 全屋已關閉 |
| `STATUS_ALL` | 不變 | 不變 | 狀態查詢：燈：開；窗簾：開 |

- 被拒识时**不执行设备命令、不改变状态**，返回 `已拒識：…` 与 `device_response="REJECTED"`；
- 未知命令（不在六类白名单内）即使置信度很高也会被拒识（`rejection_reason=unknown_command`）；
- 同一 `seq` 重复执行不重复改状态（与二期串口模拟设备一致的幂等保护）。

## 7. 日志格式（JSONL，可审计）

每次操作追加一行到 `outputs/v3_demo_log.jsonl`（UTF-8，`ensure_ascii=False`），字段同时
兼容二期控制服务风格与 v3 语义别名：

```json
{
  "timestamp": "2026-10-21T15:02:11",
  "audio_path": "…\\LIGHT_ON_polite_canto-tts-default.wav",
  "seq": 1,
  "model_type": "dtw_v2",
  "model_name": "dtw_v2(6类)",
  "predicted_intent": "LIGHT_ON",
  "confidence": 1.0,
  "threshold": 0.9,
  "accepted": true,
  "device_response": "已開燈",
  "device_state": {"light": "on", "curtain": "closed"},
  "scores": {"LIGHT_ON": 1.0, "LIGHT_OFF": 0.0, "…": 0.0},
  "latency_sec": 2.076,
  "predicted": "LIGHT_ON",
  "command_label": "開燈",
  "distance": 0.0,
  "rejection_reason": null,
  "state": {"light": "on", "curtain": "closed"},
  "state_text": "燈：開；窗簾：關",
  "previous_state": {"light": "off", "curtain": "closed"},
  "state_changed": true,
  "response": "已開燈",
  "latency": 2.076
}
```

关键审计字段：`timestamp`、`audio_path`、`seq`、`predicted_intent`、`accepted`、
`threshold`、`state`、`response`、`latency_sec`。`GET /api/log?limit=N` 读取最近 N 条
（最新在前，`1 ≤ N ≤ 100`）。

## 8. HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 演示台页面（内联 CSS/JS，离线可用） |
| GET | `/api/health` | 健康检查，返回模型名与阈值 |
| GET | `/api/catalog` | 六个命令样本 + 未知指令样本的音频目录 |
| GET | `/api/state` | 当前设备状态（`state` / `state_text` / `threshold` / `model_name`） |
| POST | `/api/predict` | 请求体 `{"audio_path": "...", "threshold": 0.5}`（threshold 可选，0~1） |
| GET | `/api/log?limit=10` | 最近操作日志（JSONL 反序列化后最新在前） |
| POST | `/api/reset` | 重置模拟设备状态（保留审计日志） |

错误约定：缺少/非法参数返回 400，音频文件不存在返回 404，处理异常返回 500，
响应体均为 `{"error": "中文错误信息"}`。后端对推理与日志追加加了互斥锁，串行处理
请求，避免 Flask 多线程下状态竞争。

## 9. 实际验证记录（2026-10-21）

### 9.1 环境与测试

- Python 3.13.9（`D:\anaconda3\python.exe`），Flask 3.1.2，项目根目录运行。
- `python -m unittest discover -s tests`：**全量测试通过（本次收尾验证时 197 项，全部 OK）**，
  其中包含原有 65 项、`tests/test_v3_control.py` 35 项，以及并行任务新增的数据/声学/模型测试。
- 模型：默认 `data/models/tts_control_model_dtw_demo.pkl`（v2 DTW 模板，6 类，阈值 0.9）；
  第三阶段主模型 `data/models/v3_cnn_ensemble.npz`（3 种子 CNN 集成 + `v3_thresholds.json`，阈值 0.93）。

### 9.2 CLI 实测（`scripts/demo_v3.py --sequence`）

```
[1/7] LIGHT_ON_polite_canto-tts-default.wav（LIGHT_ON_09，canto-tts-default，「唔該幫我開燈」）
      識別：開燈 (LIGHT_ON) 置信度 1.000 >= 閾值 0.90 → 接受；已開燈；燈：開；窗簾：關
[2/7] LIGHT_OFF_08_edge_...wav（「唔該熄燈」）
      識別：關燈 (LIGHT_OFF) 置信度 0.524 < 閾值 0.90 → 拒識；狀態不變
[3/7] CURTAIN_OPEN_polite_canto-tts-default.wav → 打開窗簾 1.000 → 接受；窗簾已打開
[4/7] CURTAIN_CLOSE_13_edge_...wav → 關閉窗簾 0.563 < 0.90 → 拒識；狀態不變
[5/7] ALL_OFF_colloquial_canto-tts-default.wav → 全屋關閉 1.000 → 接受；全屋已關閉
[6/7] STATUS_ALL_polite_canto-tts-default.wav → 查詢 1.000 → 接受；只查詢不改狀態
[7/7] UNK_V3_01_zh-HK-HiuGaaiNeural.wav（文本「開冷氣」）
      識別：開燈 (LIGHT_ON) 置信度 0.288 < 0.90 → 拒識；狀態不變
日誌：outputs/v3_demo_log.jsonl（累計 7 條）
```

v2 DTW demo 模型对未见表达的置信度偏低（0.52 / 0.56），阈值 0.9 下会拒识，这本身
就是「拒识路径」的演示；用 `--threshold 0.5` 可看到这两条被接受并正确改变设备状态
（实测 `LIGHT_OFF 0.524`、`threshold=0.5` → 接受、`已關燈`）。

### 9.3 Web 实测（真实 HTTP，非仅 test_client）

- `python scripts/run_web_demo.py --port 8770` 启动后：
  - `GET /api/health` → 200；`GET /` → **200**，21KB HTML，包含「方言智控盒」「六類命令」
    「模擬設備狀態」「操作日誌」「拒識閾值」等中文文案，页面无任何外部 `http(s)` 资源引用；
  - `GET /api/catalog` → 六条 test 划分音频 + `UNK_V3_01_zh-HK-HiuGaaiNeural.wav`（未知）；
  - `POST /api/predict`（開燈音频，阈值 0.9）→ 200，`predicted=LIGHT_ON, confidence=1.0,
    accepted=true, response=已開燈, state={"light":"on","curtain":"closed"}`；
  - `POST /api/predict`（未知音频，阈值 0.9）→ 200，`confidence=0.288, accepted=false,
    rejection_reason=low_confidence`，状态保持不变；
  - 阈值 0.5 时关灯音频 → `accepted=true, response=已關燈`；
  - `GET /api/log` 返回最新记录；`POST /api/reset` 后状态回到「燈：關；窗簾：關」，
    审计日志仍保留；
  - 非法参数（缺 `audio_path`、阈值 1.5、文件不存在）分别返回 400/400/404 与中文错误信息。
- 验证完成后已停止 Flask 进程并确认 `8770` 端口释放，未长期占用端口。

### 9.4 v3 CNN 主模型实测（2026-10-21 收尾复验）

模型工件：`data/models/v3_cnn_ensemble.npz`（3 种子 1D-CNN 集成，纯 numpy 推理）+
同目录 `data/models/v3_thresholds.json`（`selected_threshold=0.93`、温度标定 0.45；
阈值仅在验证集 + 未知校准集上选择，test 指标只作参考记录）。`load_predictor` 会优先
探测 `src/v3_model.py` 的 `V3Predictor.load`，因此**只需 `--model data\models\v3_cnn_ensemble.npz`
即可切换到主模型，控制服务/页面代码无需修改**。

**CLI 实测**

```powershell
D:\anaconda3\python.exe scripts\demo_v3.py --sequence --model data\models\v3_cnn_ensemble.npz
```

```
[1/7] LIGHT_ON_polite_canto-tts-default.wav  → 開燈 (LIGHT_ON) 1.000 ≥ 0.93 接受 → 已開燈
[2/7] LIGHT_OFF_08_edge_...wav               → 關燈 (LIGHT_OFF) 1.000 ≥ 0.93 接受 → 已關燈
[3/7] CURTAIN_OPEN_polite_canto-...wav       → 誤判 關閉窗簾 (CURTAIN_CLOSE) 0.543 < 0.93 拒識，狀態不變
[4/7] CURTAIN_CLOSE_13_edge_...wav           → 關閉窗簾 (CURTAIN_CLOSE) 0.998 接受 → 窗簾已關閉
[5/7] ALL_OFF_colloquial_canto-...wav        → 全屋關閉 (ALL_OFF) 0.999 接受 → 全屋已關閉
[6/7] STATUS_ALL_polite_canto-...wav         → 查詢全屋狀態 (STATUS_ALL) 1.000 接受 → 只查詢不改狀態
[7/7] UNK_V3_01（文本「開冷氣」）             → 打開窗簾 (CURTAIN_OPEN) 0.755 < 0.93 拒識，狀態不變
```

结果：7 条中 **5 条正确接受**（LIGHT_ON / LIGHT_OFF / CURTAIN_CLOSE / ALL_OFF / STATUS_ALL，
其中 4 条置信度 ≈1.0），2 条拒识（CURTAIN_OPEN 被误判为 CURTAIN_CLOSE 0.543；未知
「開冷氣」被判为 CURTAIN_OPEN 0.755），与 Lead 的实测一致。单条推理 **15–51 ms**。

**Web 实测**

```powershell
D:\anaconda3\python.exe scripts\run_web_demo.py --model data\models\v3_cnn_ensemble.npz --port 8771
```

- `GET /api/health` → 200：`model_name=v3_cnn_ensemble(3seed,6class)`、`threshold=0.93`；
- `GET /` → 200，约 22 KB 中文离线页面；`GET /api/catalog` → 200（6 条命令 + 未知样本）；
- `POST /api/predict`（LIGHT_ON 音频）→ 200：`accepted=true, confidence=1.0,
  state={"light":"on","curtain":"closed"}, response=已開燈`，本次 test_client 复验
  `latency_sec=0.039`（Lead 真实 HTTP 实测约 0.184 s）；
- 验证后已停止服务并释放 8771 端口。

**v2 DTW 与 v3 CNN 对比**

| 维度 | v2 DTW（默认回退模型） | v3 CNN（第三阶段主模型） |
| --- | --- | --- |
| 工件 | `data/models/tts_control_model_dtw_demo.pkl` | `data/models/v3_cnn_ensemble.npz` + `v3_thresholds.json` |
| 方法 | MFCC13+CMVN → DTW 模板匹配（20 模板/类） | log-mel40+Δ 特征 + 3 种子 1D-CNN 集成，温度标定 0.45，纯 numpy 前向 |
| 阈值 | 0.9 | 0.93（验证集+未知校准集选择） |
| CLI 7 条结果 | 4 接受 / 3 拒识（LIGHT_OFF、CURTAIN_CLOSE、未知均因低置信度拒识） | 5 接受 / 2 拒识（CURTAIN_OPEN 误判、未知拒识） |
| 单条延迟 | 约 2–15 s（随机器负载波动） | 约 15–51 ms |
| 参考指标 | 二期 test Acc≈0.55（TTS 未见表达） | `known_coverage=0.890`、`accepted_accuracy=0.944`、`unknown_false_accept_rate=0.361`（`v3_thresholds.json` 的 test 参考记录） |
| 推荐用途 | 无 v3 工件时回退、与二期对照 | 第三阶段正式演示（更快、更稳定） |

> 说明：以上均为**模拟设备 + TTS 合成测试音频**（非真实麦克风/真实 ESP32）；test 集
> 六类各仅 5–6 条，指标波动较大。CURTAIN_OPEN ↔ CURTAIN_CLOSE 仍是主要混淆方向
> （与二期结论一致），CURTAIN_OPEN 的 canto 样本在 v3 下被误判并拒识，属当前已知问题。

## 10. 已知限制

1. **模拟设备**：`MockDeviceV3` 只维护 `{light, curtain}` 状态，不驱动真实 ESP32/串口；
   二期 `src/serial_protocol.py` 的串口链路保留未改，v3 未接入硬件。
2. **输入为 TTS 合成测试音频**：来自 `data/tts_corpus` 的 test 表达组，非实时麦克风；
   没有 VAD/端点检测/流式识别，也不是真实粤语口音录音。
3. **两个模型的阈值与已知混淆**：默认 v2 DTW 模型阈值 0.9，部分未见表达置信度偏低会拒识
   （属预期行为）；v3 主模型阈值 0.93（`v3_thresholds.json`），实测 CURTAIN_OPEN 的 canto
   样本被误判为 CURTAIN_CLOSE（0.543）而拒识，CURTAIN_OPEN ↔ CURTAIN_CLOSE 仍是主要混淆
   方向。演示时可用 `--threshold` 或页面滑块对比，但不要据此宣称更高准确率。
4. **未知指令拒识依赖阈值 + 闭集分类**：未知音频可能被误判为某个已知命令（通常置信度
   低），因此只在低置信度时被拒；v3 阈值文件记录的校准集 `unknown_false_accept_rate=0.361`，
   即约 36% 的未知样本仍会被误接受，没有独立 OOD 检测器；调低阈值会增加误接受风险。
   （演示用例「開冷氣」在 v3 下 0.755 < 0.93 被拒，只是该样本的结果，不代表全部未知。）
5. **CPU 延迟差异大**：v3 CNN 单条约 15–51 ms（毫秒级，纯 numpy 前向，不含启动时间）；
   回退的 v2 DTW 模板匹配约 1~15 秒（随机器负载波动），Web 点击后需等待「識別中…」。
   两者都不是实时流式系统。
6. **Web 服务为 Flask 开发服务器**，仅监听 `127.0.0.1`，无认证/限流，不要暴露公网；
   推理与日志写入用全局锁串行化，未做并发压力测试。
7. **日志只追加不轮转**：`outputs/v3_demo_log.jsonl` 会持续增长，可用 `--reset-log` 手动
   清空；重置设备状态不会清空日志。
8. **模型选择与工件依赖**：默认仍是 v2 DTW 模型；第三阶段主模型需显式
   `--model data\models\v3_cnn_ensemble.npz`。`V3Predictor.load` 会自动读取**同目录**的
   `v3_thresholds.json`；若只拷贝 `.npz` 而缺少该 JSON，阈值会回落到 0.0（等于全部接受），
   因此迁移模型时必须两个文件一起拷贝，或用 `--threshold` 显式指定阈值。
9. **CLI 抬头的小瑕疵**：v3 模型的「模型類型」一行显示 `unknown`（`V3Predictor` 未声明
   `model_type` 属性），但 `model_name`、识别与日志均正确；这是既有演示脚本的显示问题，
   不在本次收尾修改范围内。
10. 演示台未做移动端深度适配（窄屏下命令按钮会改为单列，功能可用）；未包含截图或录屏。

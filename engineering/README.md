# DialectVoice Engineering

方言语音智能家居转换装置（本科工程导论项目）。

用户说粤语家居命令，端侧完成 VAD 端点检测、特征提取、命令识别与置信度拒识，
再由扬声器播放预录的普通话指令去驱动智能音箱；另备串口 ESP32-S3 与继电器直控两条备用通道。

> **当前状态（2026-11-19）**：默认模型 `v3_cnn_ensemble_v32b.npz`，合成 test 483 条 argmax **0.9110**、val 440 条 0.9523；双信号拒识样本外 FAR **0.0702**（真人非命令负样本）、合成覆盖率 0.8205；全量单测 **347 passed**。硬件实机由硬件负责人负责，真人命令正样本仍为 0（全部准确率为 edge-tts 合成粤语口径）。结果角度审查见 `docs/项目审查_结果角度_2026-11-19.md`。

## 技术路线

部署口径是**两遍识别 + 双信号拒识 + 状态先验三态决策**（v5 S4/S4(b)/S5）：

```
方言语音 → 流式 VAD（1 s 滑窗 / 100 ms 步进）
        ├─ Pass-1 预触发：证据 EMA + 连续两窗一致 → 给用户看的即时反馈
        │                单窗后验过快速端点门槛 → 240 ms 静音挂起即切段
        └─ Pass-2 整句复核：log-mel40+Δ（整句 CMVN，插值到 256 帧）
                          → 1D-CNN 三种子集成（纯 numpy，1.35 MB）
                          → 关键词偏置 + 解析去偏（unbias）
        → 双信号门限：maxprob ≥ 0.51863 AND ood ≥ −0.026796  （v32b 重标，2026-11-19）
                      （ood = 到 500 条 f16 参照池的 kNN 余弦距离取负，k=5；
                        阈值随模型切换重标定，旧值 0.309193/−0.022306 已作废）
        → 设备状态先验 + 期望代价三态决策：execute / clarify / reject
        → 高风险 ALL_OFF 的 OOD 边界复核 + 严格 red-soft 澄清
        → 去抖 + 澄清状态机
        → 语义层映射（命令标签 → 普通话指令 → 预录音频）
        → 执行（日志 / 模拟 ESP32 / 串口 / 继电器 / MQTT）+ 播报
```

**阈值与参照池必须成对下发**：换池必须重标定，池条数不符硬失败。
缺池时降级到单阈值 0.945，并在每条结果的 `rule` 字段标明——降级不是错误，
但看不见的降级等于把真实域 FAR 悄悄退回 0.3110 那一档。

命令集固定 6 条，与训练语料一致：
`LIGHT_ON` / `LIGHT_OFF` / `CURTAIN_OPEN` / `CURTAIN_CLOSE` / `ALL_OFF` / `STATUS_ALL`。

> 早期方案「方言特征 → 普通话特征」的回归映射已被自有实验否决，详见
> [`docs/技术路线修订说明.md`](docs/技术路线修订说明.md)。转换发生在语义层，不在特征层。
>
> 单遍路径（`--single-pass`）保留为历史指标口径与缺件降级路径，
> S0 公布的 FAR 0.3110 就是它测出来的。

## 安装

依赖按用途拆表，**部署到设备只需要第一张**：

```bash
pip install -r requirements.txt            # 端侧运行：numpy / PyYAML / soundfile / sounddevice
pip install -r requirements-hardware.txt   # 可选：串口 ESP32 / MQTT（继电器另需 RPi.GPIO）
pip install -r requirements-dev.txt        # 评估与测试：scipy / pytest / edge-tts
pip install -r requirements-docs.txt       # 可选：答辩 PPT / Word 构建
```

`runtime/` 本身只依赖 numpy + 标准库，可单独拷走部署。
训练依赖在 `DialectVoice_Research/requirements_v3.txt`（torch / numba），不在本表。

## 目录

| 路径 | 作用 |
|---|---|
| `src/vad.py` | 能量 VAD：自适应噪声底、静音挂起、前后裁剪 |
| `src/realtime.py` | 部署管线：两遍识别 → 门限 → 三态决策 → 执行 → 播报 → 日志 |
| `src/commands.py` | 6 条命令定义、普通话文本、设备状态机 |
| `src/control.py` | 执行层：日志 / 模拟 ESP32 / 串口 / 继电器 / MQTT / Home Assistant |
| `src/speaker_out.py` | 普通话预录音频播报 |
| `src/features.py` `src/recognition.py` | 二期 MFCC + DTW 基线，仅用于对比实验 |
| `runtime/` | 自包含推理运行时（由 `scripts/sync_runtime.py` 从 Research 同步，可独立部署） |
| `runtime/streaming.py` | Pass-1 流式预触发：滑窗、证据 EMA、软候选与严格预触发解耦 |
| `runtime/verify.py` | Pass-2 整句复核：关键词偏置与解析去偏 |
| `runtime/selective.py` | 双信号门限 `SelectiveGate`：五信号、参照池、降级留痕 |
| `runtime/decision.py` | 状态先验 + 期望代价三态决策 |
| `runtime/dialog.py` | 去抖窗口 + 澄清状态机 |
| `runtime/state.py` | 状态源：mock / Home Assistant / MQTT |
| `runtime/models/` | **下发 4 件**：`v3_cnn_ensemble_v32b.npz`（当前默认模型，test 0.9110）、`v3_thresholds_vad.json`（单阈值降级路径）、`v3_selective_gate.json`（双信号门限 maxprob 0.51863 / ood −0.026796）、`v3_knn_pool.npz`（500×256 f16 参照池，已随 v32b 重标） |
| `firmware/esp32s3_smart_box/` | ESP32-S3 可编译/可烧录执行固件、默认引脚表、USB/UART 整包与协议验收脚本 |
| `data/audio/mandarin/` | 6 条普通话指令预录音频（edge-tts 合成） |
| `docs/` | 开题报告（Beamer/PDF/PPTX）、项目建议书、修订说明、v5 方案设计、评估结果、项目审查 |

> `runtime/models/` 里另有 `v3_cnn_ensemble.npz` 与 `v3_thresholds.json` 两个
> **旧非-VAD 版本**，部署不使用，仅作 `tests/test_config.py` 的基线 fixture 与
> 消融复现对照。它们仍在 `sync_runtime.py` 的同步清单里，因此不能直接删；
> 打包下发时应排除（见 `docs/项目审查_2026-11-13.md` §3.5）。

## 常用命令

```bash
# 同步自包含运行时与模型（带 sha256 校验，--check 只校验不写入）
python scripts/sync_runtime.py
python scripts/sync_runtime.py --check

# 麦克风实时识别（默认部署口径：两遍 + 状态先验 + 三态决策）
python scripts/run_monitor.py --executor mock_esp32

# 退回单遍路径做对照（S4 之前的历史指标口径）
python scripts/run_monitor.py --single-pass --executor log

# 真实继电器（默认 mock，不碰真 GPIO；仅树莓派）
python scripts/run_monitor.py --executor relay --no-mock-relay

# 离线回放指定音频
python scripts/run_replay.py path/to/audio.wav

# 部署管线评估（已知命令 + 未知指令误接受率）
python scripts/evaluate_pipeline.py \
  --manifest "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" \
  --subset test \
  --unknown "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/unknown_v3_vad"

# 会话级评估（--mode stream 为部署口径）与感知延迟基准
python scripts/eval_dialog.py --mode stream
python scripts/bench_perceived_latency.py

# 端点参数扫描；先守 300 ms 延迟预算，再比较准确率与截断
python scripts/sweep_fast_endpoint.py \
  --manifest ../DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv \
  --values 120 180 240 300 390

# 红区澄清收益 / 误澄清率评估
python scripts/eval_red_soft.py

# 单元测试
python -m pytest tests/ -q
```

启动横幅会打印当前**路径 / 拒识规则 / 状态源**，现场演示与报告是否同一口径一眼可见。

## ESP32-S3 固件

固件采用“PC 识别、ESP32-S3 执行”边界，接收 `CMD,<seq>,<intent>`，
控制灯与窗帘舵机，并回复 `ACK/STATE`。编译、烧录和引脚覆盖说明见
[`firmware/esp32s3_smart_box/README.md`](firmware/esp32s3_smart_box/README.md)。

```powershell
cd firmware/esp32s3_smart_box
pio run -t upload --upload-port COM5
python ..\..\scripts\esp32_serial_smoke.py --port COM5
cd ..\..
python scripts\run_monitor.py --executor serial --port COM5
```

`dist/` 同时提供 `merged-usb.bin`、`merged-uart.bin`，可使用 esptool 直接烧录到
地址 `0x0`。默认引脚仍是临时工程配置，最终必须与硬件负责人接线表对齐。

## 训练与标定（依赖 DialectVoice_Research）

```bash
# 用部署同款 VAD 重生成训练语料，保证训练输入与部署输入同源
python scripts/make_trimmed_corpus.py

# 在 Research 侧重训练
python scripts/train_v3_model.py \
  --manifest data/tts_corpus/manifest_yue_v3_1_vad.csv \
  --split data/splits/track_a_expression_split_v3_1.csv \
  --out data/models/v3_cnn_ensemble_v32b.npz

# 在验证集 + 未知校准集上选拒识阈值（测试集不参与）
python scripts/calibrate_threshold.py
```

## 第一轮优化快照

第一轮安全/端点优化（2026-11-13）已把快速端点从 120 ms 调到 240 ms，并加入
高风险 `ALL_OFF` 的 OOD 边界确认与严格 red-soft 澄清。部署路径会话成功率由
0.7371 提升到 **0.7702**，危险误动作率由 0.0683 降到 **0.0497**，动作延迟 P95
为 **290.5 ms**。这是工程缓解，**尚未**用同一切法重生成语料并重训；完整证据与
口径见 [`docs/优化验证_2026-11-13.md`](docs/优化验证_2026-11-13.md)。

## 尚未解决

合成 TTS 语料与真人粤语之间存在域差距，未见说话人、真实噪声、远场、真实端到端延迟
四项指标均无真人数据支撑，不能作为性能承诺。

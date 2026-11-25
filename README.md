# 方言智控盒 · 进度整理

日期：2026-11-25

一句话：粤语六类家居命令的端侧识别与执行原型。软件链路是通的，SZPI 板卡端的 BSP 和自检也编译通过了；还差实机联调、真人语音和几项收尾材料。

## 目前做到哪

**识别（research/）**

- 六类命令：开灯 / 关灯 / 开窗帘 / 关窗帘 / 全关 / 查询。
- 当前模型 v32b：test 483 条 0.9110，val 440 条 0.9523。
- 口径说明：test 是「未见表达、同音色」的 33 个文本，全部为合成 TTS，不是跨说话人结果。
- 拒识用双信号（maxprob + kNN-OOD）：真实域 FAR 0.0702，合成命令覆盖率 0.8205。覆盖率是合成侧代理，真人命令还没测。

**工程（engineering/）**

- VAD → 特征 → CNN → 拒识 → 三态决策 → 播报/执行，链路能跑。
- 最近一次全量测试 350 passed（本机 11-20，含固件与串口相关测试）。
- ESP32-S3 可烧录执行端固件已产出（USB / UART 两版整包 bin）；C 端侧推理内核与 numpy 在 483 条上 482/483 一致。

**板卡端（szpi/）**

- 立创·实战派 ESP32-S3（N16R8）BSP：I2C / PCA9557、全双工 I2S、ES8311 / ES7210、SD、背光与按键。
- 阶段一硬件自检程序已写；ESP-IDF 5.4.1 全量编译通过，固件 569712 B。
- 还没上真板：PSRAM、四个 I2C 地址、SD、录音通路都没实测。

**还没做**

- 真人命令录音：0 条（`scripts/collect_human.py` 已备，等人录）。
- 硬件实机联调：串口 / GPIO / 舵机 / 麦克风都还没接。
- 3 分钟演示视频、成本表、部分答辩材料。
- 端侧推理要不要上 S3 没定：C 内核在主机上约 107 ms/条，S3 粗估 0.5–1 s，可能超 300 ms 预算，所以现在默认还是「PC 推理 + S3 执行」。

## 仓库里有什么

| 目录 | 是什么 | 状态 |
|---|---|---|
| `engineering/` | 运行时代码、部署模型、v5 方案 / 测试报告 / 交付清单 | **2026-11-06 快照**（199 tests） |
| `research/` | 训练评测脚本、语料清单与划分、模型工件、v2 / v3 报告 | **2026-11-06 快照**（197 tests） |
| `szpi/` | ESP32-S3 板卡端固件与后端占位 | 2026-11-20 起 |

注意：11-13 之后的最新工作（v32b 默认模型、C 内核固件、350 tests、新报告）还在本机 `D:\Run_Tao\DialectVoice_Engineering` 和 `..._Research`，没同步进这个仓库。这是目前最该补的一步。

远程仓库还有两条没并进 `main` 的分支：`pc-dialect-engine`（PC / Web 版 DialectHome，SenseVoice + NLU）和 `esp32-dialect-voice`（另一版 ESP32 端 + 云端适配）。都是独立尝试，想看切过去就行。

## 怎么跑

识别 demo（research，纯 numpy 推理）：

```bash
cd research
python scripts/demo_v3.py --sequence --model data/models/v3_cnn_ensemble.npz
python scripts/run_web_demo.py --model data/models/v3_cnn_ensemble.npz
```

工程回放（engineering）：

```bash
cd engineering
python scripts/run_replay.py --audio ../research/data/tts_corpus/wav16k_v2/edge/LIGHT_ON_02_edge_tts_zh-HK-HiuGaaiNeural_v0_00.wav
python -m pytest tests/ -q
```

SZPI 编译（Windows + ESP-IDF 5.4.1）：

```powershell
cd szpi
powershell -ExecutionPolicy Bypass -File .\flash.ps1 -BuildOnly
```

## 数据说明

仓库只放了清单 / 划分 / 模型 / 少量演示音频（canto、v2 test 111 条、unknown 72 条，约 27 MB）。
完整合成语料（v3 / v3.1 / VAD 切段，约 360 MB）、原始 mp3、FLEURS 真人语料（约 478 MB，CC BY 4.0）都没放。
重建看 `research/docs/data_v3.md`、`data_v3_1.md`；FLEURS 从 HuggingFace 下，再用 `engineering/scripts/build_real_testset.py` 生成测试集。

## 下一步（按先后）

1. 把本机最新的 engineering / research 同步进仓库（v32b 模型、350 tests、固件与新报告），不然仓库和实际进度对不上。
2. 实板烧录，跑阶段一自检，确认 PSRAM / 四个 I2C 地址 / SD / 录音。
3. 真人命令采集（2–3 人 × 30–50 次），先看真人命令的接受率和 FAR，再谈「可用」。
4. 3 分钟演示视频和成本表。
5. 决定端侧推理是否上 S3，或维持「PC 推理 + S3 执行」。

## 详细文档

- 工程方案与评估：`engineering/docs/v5方案设计.md`、`测试报告.md`、`交付物清单.md`
- 识别与数据：`research/docs/v3_report.md`、`v2_report.md`、`data_v3.md`
- 板卡端：`szpi/README.md`、`szpi/PROGRESS.md`
- 最近进度：本机 `DialectVoice_Engineering/PROGRESS.md`（还没进仓库）

## 其他

- 代码 / 文档未声明开源许可证，默认保留所有权利；要用再说。
- 第三方：FLEURS CC BY 4.0 未分发；edge-tts / canto-tts 合成的粤语不是真人录音。
- 作者：Run-Tao，上海交大本科工程导论项目。

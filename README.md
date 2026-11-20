# 方言智控盒（DialectVoice Smart Control Box）

粤语家居语音指令识别与设备控制的工程原型：用户说粤语命令，端侧完成
**VAD 端点检测 → log-mel 特征 → 1D-CNN 六类命令识别 → 置信度/选择性拒识 → 语义层映射 → 播放普通话指令驱动智能音箱**；
另备串口 ESP32-S3 与继电器直控两条执行通道（硬件未接，代码就绪）。

命令集固定 6 类：`LIGHT_ON`（開燈）、`LIGHT_OFF`（關燈）、`CURTAIN_OPEN`（打開窗簾）、
`CURTAIN_CLOSE`（關閉窗簾）、`ALL_OFF`（全屋關閉）、`STATUS_ALL`（查詢全屋狀態）。

> 本仓库是一个**研究 + 工程原型**，不是已量产的产品。所有识别准确率数字均来自
> **合成 TTS 粤语语料**；真人粤语命令正样本尚未采集，真实噪声/远场/未见说话人未验证。
> 真实域误接受率（FAR）已用公开真人粤语语料（FLEURS）测出，是目前最重要的安全短板与改进依据。

---

## 仓库结构

```
dialect-voice-smart-control-box/
├── engineering/     # DialectVoice_Engineering：权威运行时代码、部署模型、工程文档（v5）
│   ├── src/         # VAD、实时管线、命令/状态机、执行层、播报、配置
│   ├── runtime/     # 自包含推理运行时（由 scripts/sync_runtime.py 从 research 同步，可独立部署）
│   ├── scripts/     # 采集/回放/评估/延迟基准/门限构建/报告生成
│   ├── tests/       # pytest（199 项）
│   ├── data/audio/mandarin/   # 6 条普通话播报预录音频
│   └── docs/        # v5 方案设计、测试报告、交付物清单、开题报告/建议书、评估结果 JSON
└── research/        # DialectVoice_Research：训练/评测脚本、语料与实验产物（v2/v3/v3.1）
    ├── src/         # 声学前端、CNN 推理、分类器、选择性分类、流式、决策、对话、状态
    ├── scripts/     # 数据生成/清洗/训练/对照实验/拒识/OOD/报告/演示
    ├── tests/       # unittest（197 项）
    ├── data/        # 清单、固定划分、表达库、模型工件、少量演示音频
    ├── outputs/     # 关键实验结果 JSON/CSV、数据审计、报告表格
    └── docs/        # v2/v3 执行报告、数据文档、演示文档、声学消融（含 docx/pptx 交付件）
```

两个目录的权威关系：

- **engineering/** 是运行时代码与工程文档的唯一权威；`runtime/` 是从 `research/src/` 同步来的自包含副本。
- **research/** 是语料、训练脚本、算法源与模型工件的来源；训练/标定命令以它为准。
- 历史负结果与第三阶段（v3）数据/实验记录保留在 `research/docs` 与 `research/outputs`，便于追溯。
- 未入库：历史归档（`docs/archive_2026-10-30/`、`legacy/`）、中间 smoke/探针产物、特征缓存、
  原始 mp3 与完整音频语料（见下方“数据说明”）；以保证仓库只保留当前有效、可追溯的文件。


---

## 快速开始

### 1. 识别模型演示（research 侧，纯 numpy，无 GPU）

```bash
cd research
python scripts/demo_v3.py --sequence --model data/models/v3_cnn_ensemble.npz
# 六类命令 + 1 条未知指令，输出识别/置信度/拒识/模拟设备状态/JSONL 日志

python scripts/run_web_demo.py --model data/models/v3_cnn_ensemble.npz
# 浏览器打开 http://127.0.0.1:8770/ ，离线演示台（六命令按钮 + 未知指令 + 状态 + 日志）
```

仅需 `numpy` 即可跑推理；`torch` 仅训练需要，`numba` 仅基线 DTW 实验需要。
模型是 3 种子 1D-CNN 集成（1.35 MB，约 35.9 万参数），前向约 16 ms/条（不含特征提取）。

### 2. 工程运行时（engineering 侧）

```bash
cd engineering
python -m pytest tests/ -q          # 199 passed

# 离线回放一条音频（模拟执行器）
python scripts/run_replay.py ../research/data/tts_corpus/wav16k_v2/edge/LIGHT_ON_02_edge_tts_zh-HK-HiuGaaiNeural_v0_00.wav

# 实时麦克风（需要 sounddevice 与麦克风）
python scripts/run_monitor.py --executor mock_esp32

# 部署管线评估（使用仓库内的清单；完整音频需按下方“数据”说明重建）
python scripts/evaluate_pipeline.py \
  --manifest ../research/data/tts_corpus/manifest_yue_v3_1_vad.csv --subset test \
  --unknown ../research/data/tts_corpus/unknown_v3_vad
```

依赖见 `engineering/requirements.txt`（含音频/界面/串口等可选依赖，可按需精简）。

### 3. 测试

```bash
cd engineering  && python -m pytest tests/ -q
# 199 passed（已在本仓库副本实测）

cd research     && python -m unittest discover -s tests -v
# 作者完整工作区：197 项全绿
# 本仓库只含演示音频：其中 6 项“数据完整性 / smoke 数据”测试会因完整语料未随仓库分发而失败（预期）
# 失败项：test_v3_dataset.test_audio_files_exist_and_format、test_sha256_matches_disk_for_sample、
#         test_offline_augmentation_is_deterministic；test_v3_1_dataset.test_audio_quality；
#         test_experiment.test_cosine_result_saved；test_mapping_experiment.test_ols_result_saved
# 完整语料重建后 197 项全绿（重建步骤见下方“数据说明”）。
```

> 注：部分脚本/文档中的示例路径写死为作者本机绝对路径（`D:/Run_Tao/...`），
> 在其它机器上请用仓库内相对路径或命令行参数替换；运行时代码本身不含硬编码路径。

---

## 主要结果（简版）

### 六类命令识别（固定测试集：31 个未见表达组 / 483 条；matched-v2 = 与二期相同的 111 条）

| 模型 | Full test Acc | Macro-F1 | matched-v2 Acc |
|---|---:|---:|---:|
| 二期 MFCC+DTW 基线 | 0.5901 | 0.5921 | 0.5495 |
| 经典判别式（MFCC+Δ+ΔΔ + logreg） | 0.6749 | 0.6706 | 0.7027 |
| 三期 1D-CNN 3 种子集成（部署件） | 0.9048 | 0.9022 | 0.8829 |
| CNN ⊕ 经典 融合（实验最佳） | 0.9151 | 0.9126 | 0.8919 |

- v2→v3 提升：full **+31.47pp**（McNemar p=9.2e-32，按表达组自助 95%CI [+0.178,+0.458]），
  matched-v2 **+33.33pp**（p=3.0e-9）。
- correction（否定/纠正句式）语域仍是最弱项：**0.6429**；窗帘开/关方向互换占剩余错误的大多数。
- 数据消融（v3→v3.1，补 18 条 correction 训练表达）：full +3.52pp（p=2.2e-4）。
- 细节与统计检验：`research/docs/v3_report.md`、`research/outputs/v3_tables_main_v3_1.md`。

### 拒识 / 真实域风险（v5，比准确率更值得看）

| 口径 | 指标 | 值 |
|---|---|---|
| 单 max-prob 阈值 0.945 | 真实域 FAR（FLEURS 真人负样本） | **0.3110**（CI [0.2936,0.3289]），危险误动作率 0.1437 |
| 选择性分类 S3（maxprob + kNN-OOD AND） | 真实域 FAR / 合成覆盖率 | **0.0660**（CI [0.0571,0.0761]）/ 0.8455 |
| 下发门限 S4（500 条 f16 参照池，250 KB） | 样本外 FAR / 合成覆盖率 / 危险误动作率 | **0.0812** / 0.8205 / 0.0212 |
| 两遍流式 S4(b) | 动作延迟 P95 / 预触发 CPU P95 / 对话完成 P95 | **142.7 ms** / 17.1 ms / 942.7 ms |
| 对话补救 S5（部署口径） | 易混对增益 | **0.0000**（离线口径 +0.0709，验收未达成） |

结论：**识别准确率在合成语料上已大幅提升；拒识从真实域 FAR 0.3110 压到 0.0660–0.0812（需成对下发门限与参照池）；
但 S5 对话补救在部署口径下没有收益，且所有覆盖率均是合成侧代理，真人命令覆盖率仍未知。**

---

## 数据说明（重要）

为控制仓库体积与第三方许可风险，**本仓库不包含大音频/原始语料**，只保留：

- 全部**清单、固定划分、粤语表达库、未知/易混指令清单**（CSV）；
- **模型与门限工件**（`data/models/`、`runtime/models/`）；
- **少量演示音频**（约 27 MB）：canto 合成音色、v2 测试集 111 条、unknown_v3 72 条；
- 关键**实验结果 JSON**与数据审计（`outputs/`）、报告文档与 docx/pptx 交付件。

未包含、需另行获取或重建的内容：

| 内容 | 大小 | 获取/重建方式 |
|---|---|---|
| 完整合成语料 v3 / v3.1 / VAD 切段（约 2900–4800 条） | ≈360 MB | `research/docs/data_v3.md`、`data_v3_1.md` 与 `scripts/generate_v3_corpus.py`、`generate_v3_1_corpus.py`、`scripts/make_trimmed_corpus.py`（TTS 步骤需联网；v3 条件变体由 v2 音频离线确定性生成） |
| v2 全量 TTS 语料（600 条） | ≈52 MB | `scripts/generate_tts_corpus.py` 等（edge-tts / canto-tts） |
| FLEURS 真人粤语（`yue_hant_hk` test） | ≈478 MB | HuggingFace `google/fleurs`（CC BY 4.0）；国内可用 `hf-mirror.com`；再跑 `engineering/scripts/build_real_testset.py` |
| 原始 mp3、特征缓存、旧版中间产物 | — | 已剔除，不建议入库 |

第三方与许可提示：

- **FLEURS** 为 CC BY 4.0，本仓库**未分发**其音频；仅保留由它派生的
  `research/data/real_domain_testset/manifest.csv`（含片段元数据）与 `provenance.json`。
  使用/再分发 FLEURS 时请遵守其许可并注明来源。
- 粤语命令音频为 **edge-tts / canto-tts 合成**，不是真人录音；相关文本仅经规则审查（`RULE_REVIEWED`），未经母语者逐条核验。
- 普通话播报音频为 edge-tts 合成的预录音频。
- 本仓库**未声明开源许可证**（默认保留所有权利）。如需以 MIT/Apache 等许可发布，请由作者补充 LICENSE；引用本项目请注明作者与仓库地址。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [`engineering/docs/v5方案设计.md`](engineering/docs/v5方案设计.md) | v5 路线：两遍识别、风险-覆盖拒识、状态先验、真实域评估（S0/S3/S4/S4(b)/S5） |
| [`engineering/docs/测试报告.md`](engineering/docs/测试报告.md) | 自动生成的测试报告：识别、真实域 FAR、选择性分类、延迟、会话级效果 |
| [`engineering/docs/交付物清单.md`](engineering/docs/交付物清单.md) | 逐项交付物与状态（含未完成项与答辩前风险） |
| [`engineering/docs/技术路线修订说明.md`](engineering/docs/技术路线修订说明.md) | 为什么放弃「方言→普通话特征映射」 |
| [`engineering/docs/开源智能音箱借用评估_2026-11-06.md`](engineering/docs/开源智能音箱借用评估_2026-11-06.md) | 执行端硬件借用的许可/能力核实 |
| [`research/docs/v3_report.md`](research/docs/v3_report.md) | 第三阶段完整报告（数据、模型、实验、拒识、原型、局限、复现） |
| [`research/docs/v2_report.md`](research/docs/v2_report.md) | 第二阶段负结果：扩数据不提升，瓶颈是方向词混淆 |
| [`research/docs/data_v3.md`](research/docs/data_v3.md) / [`data_v3_1.md`](research/docs/data_v3_1.md) | 语料生成、质量审计、泄漏检查、条件验证 |
| [`research/docs/demo_v3.md`](research/docs/demo_v3.md) | CLI / Web 演示与实测记录 |
| [`research/docs/acoustic_v3.md`](research/docs/acoustic_v3.md) | 声学前端与经典分类器消融 |

报告交付件：`research/docs/方言智控盒_第三阶段报告.docx`（32 页）、`..._汇报.pptx`（14 页）。

---

## 复现与可追溯性

- 数据与划分文件带 SHA-256，审计结果在 `research/outputs/*_data_audit.json`；实验协议、随机种子与温度标定记录在
  `research/outputs/v3_experiments/train_report.json` 与 `main_v3_1/results.json`。
- 一键脚本：`research/scripts/run_v3_all.ps1`（训练 → 对照实验 → 拒识 → 报告表格 → CLI 演示）。
- 训练：`research/scripts/train_v3_model.py`（仅验证集早停/温度标定，测试集不参与选择）。
- 运行时代码与模型成对下发：`engineering/runtime/models/` 中 `v3_selective_gate.json` 与 `v3_knn_pool.npz`
  必须成对替换；只改阈值不换池会破坏标定。
- 复现实验的完整命令见 `research/docs/v3_report.md` §9 与 `engineering/docs/测试报告.md`。

## 已知限制（必须与结果一起读）

1. **无真人命令正样本**：识别准确率只代表合成 TTS；真实域只测了负样本 FAR。
2. **拒识标定集偏小**：S3 的 0.0660 来自 72 条合成负样本的标定，重采样会摆动；
   S4 改用交叉拟合/500 条池后样本外 FAR 0.0812，但覆盖率仍只是合成侧代理。
3. **窗帘方向仍最弱**（CURTAIN_CLOSE 类 F1 ≈ 0.78；correction 句式 0.64），是下一步数据与建模重点。
4. **S5 对话补救未达成验收**：部署口径易混对增益 0.0000。
5. **硬件未联调**：ESP32-S3 串口、继电器、OLED、真实智能音箱均未接实机；延迟为本机数据。
6. **仓库不含大音频**：按上文“数据说明”重建后才能完整复跑训练/大规模评测。

## 作者

Run-Tao · 上海交通大学 本科工程导论项目（2026）

如发现问题或希望合作，请在本仓库提 Issue。

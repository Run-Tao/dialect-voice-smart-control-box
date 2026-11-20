# DialectVoice Research - Phase 1

方言语音命令识别项目的公共基础框架。当前只实现第一阶段：

- 数据清单读取与校验；
- 按说话人划分训练/验证/测试；
- MFCC特征提取；
- 固定长度处理与DTW对齐接口；
- 配置与结果保存；
- 单元测试和冒烟检查。

## 安装

```bash
python -m pip install -r requirements.txt
```

当前核心只依赖 numpy。后续算法可再安装 scikit-learn、matplotlib 等。

## 生成合成冒烟数据

```bash
python scripts/make_smoke_data.py
```

注意：合成数据只用于验证代码能跑通，不能用于论文实验结论。

## 运行第一阶段检查

```bash
python scripts/run_stage1_check.py --config config.json --manifest data/smoke_manifest.csv
```

## 运行测试

```bash
python -m unittest discover -s tests -v
```

## 目录

```text
src/
  config.py      配置读取与校验
  manifest.py    数据清单
  split.py       按说话人划分
  audio_io.py    WAV读取
  features.py    MFCC
  alignment.py   固定长度与DTW
  results.py     结果保存
scripts/
  make_smoke_data.py
  run_stage1_check.py
tests/
data/
outputs/
```

## 第二阶段：基线实验

```bash
python scripts/run_baseline_experiment.py --method all --output-root outputs/baselines_smoke
```

方法包括：

- `cosine`：固定长度MFCC均值模板 + 逐帧余弦相似度平均；
- `dtw`：多模板DTW，路径长度归一化距离；
- `logreg`：句子级MFCC统计特征 + 纯numpy Softmax回归；
- `svm`：句子级MFCC统计特征 + 纯numpy线性SVM。

输出：

```text
outputs/baselines_smoke/<method>/result.json
outputs/baselines_smoke/<method>/predictions.csv
```

注意：当前 `data/smoke_manifest.csv` 是合成冒烟数据，只用于验证代码路径，不能作为真实识别性能结论。

## 第三阶段：OLS/Ridge映射冒烟测试

```bash
python scripts/run_baseline_experiment.py --method ols --svd-mode none --output-root outputs/mapping_smoke_none
python scripts/run_baseline_experiment.py --method ols --svd-mode R0_literal_rank_k --output-root outputs/mapping_smoke_r0
python scripts/run_baseline_experiment.py --method ols --svd-mode R1_global_projection --output-root outputs/mapping_smoke_r1
python scripts/run_baseline_experiment.py --method ridge --svd-mode none --output-root outputs/mapping_smoke_ridge
```

映射结果保存在 `outputs/<root>/<method>/result.json`，包含：

- 映射方法、是否含截距、SVD模式、配对数量；
- train RMSE/MAE/mean cosine；
- Accuracy、Macro-F1、混淆矩阵；
- 每条测试样本预测；
- 训练/推理耗时。

注意：当前 `smoke_manifest.csv` 是合成数据，source/target语言均为null，配对来自同一训练集合，仅用于验证流程。


## 第一阶段 TTS 与语料验证

隔离环境：

```powershell
python -m venv .venv_tts
.\.venv_tts\Scripts\python.exe -m pip install edge-tts canto-tts soundfile
$env:HF_ENDPOINT="https://hf-mirror.com"
```

生成粤语TTS：

```powershell
.\.venv_tts\Scripts\python.exe scripts/tts_validation_edge.py
.\.venv_tts\Scripts\python.exe scripts/tts_validation_canto_batch.py
```

生成核心六类语料表：

```powershell
python scripts/build_core_corpus.py
```

转换16k WAV并运行算法冒烟：

```powershell
.\.venv_tts\Scripts\python.exe scripts/convert_tts_to_wav16k.py
python scripts/run_tts_algorithm_smoke.py
```

串口协议模拟测试：

```powershell
python scripts/serial_mock_test.py
python -m unittest tests.test_serial_protocol -v
```

注意：TTS音频、合成数据和模板留出实验只用于流程验证，不代表真实粤语识别性能。

## 软件阶段完整运行顺序

```powershell
cd D:\Run_Tao\DialectVoice_Research

# 1) 基础环境与测试
python -m unittest discover -s tests -v

# 2) TTS隔离环境
python -m venv .venv_tts
.\.venv_tts\Scripts\python.exe -m pip install -r requirements_tts.txt
$env:HF_ENDPOINT="https://hf-mirror.com"

# 3) 生成TTS语料（canto-tts + edge-tts粤语/普通话）
.\.venv_tts\Scripts\python.exe scripts/generate_tts_corpus.py

# 4) 语料统计
python scripts/corpus_report.py --manifest data/tts_corpus/manifest.csv

# 5) 合成粤语基线实验（说话人隔离）
python scripts/run_baseline_experiment.py --method cosine --manifest data/tts_corpus/manifest_yue.csv --output-root outputs/tts_yue_baselines
python scripts/run_baseline_experiment.py --method dtw --manifest data/tts_corpus/manifest_yue.csv --output-root outputs/tts_yue_baselines
python scripts/run_baseline_experiment.py --method logreg --manifest data/tts_corpus/manifest_yue.csv --output-root outputs/tts_yue_baselines
python scripts/run_baseline_experiment.py --method svm --manifest data/tts_corpus/manifest_yue.csv --output-root outputs/tts_yue_baselines
python scripts/run_baseline_experiment.py --method ols --manifest data/tts_corpus/manifest_yue.csv --output-root outputs/tts_yue_baselines
python scripts/run_baseline_experiment.py --method ridge --manifest data/tts_corpus/manifest_yue.csv --output-root outputs/tts_yue_baselines

# 6) 评估模型与拒识
python scripts/train_control_model.py
python scripts/evaluate_rejection.py

# 7) 演示模型与端到端模拟控制
python scripts/train_demo_control_model.py
python scripts/run_control_demo_sequence.py

# 8) 单文件音频控制
python scripts/run_audio_control.py --audio data/tts_corpus/wav16k/canto/LIGHT_ON_polite_canto-tts-default.wav --model data/models/tts_control_model_demo.pkl --seq 1
```

### 重要说明

- `tts_control_model.pkl` 是留出表达模板的评估模型。
- `tts_control_model_demo.pkl` 用全部合成粤语TTS训练，仅用于端到端演示，不能作为泛化性能证据。
- `run_control_demo_sequence.py` 使用训练分布内的TTS音频，目的是验证软件闭环，不是识别性能评估。
- 串口模块默认使用模拟设备；真实ESP32-S3未接入。
- 所有粤语文本仍需母语者核验。

## 优化后的DTW控制流程

```powershell
# 验证DTW最大帧数改进（配置中max_frames已改为60）
python scripts/run_baseline_experiment.py --method dtw --manifest data/tts_corpus/manifest_yue.csv --output-root outputs/tts_yue_baselines_max60

# 训练DTW评估模型 + 拒识阈值校准
python scripts/train_dtw_control_model.py

# DTW版六类命令端到端模拟控制
python scripts/run_dtw_control_demo_sequence.py
```

说明：`tts_control_model_dtw.pkl` 为表达模板留出评估模型；`tts_control_model_dtw_demo.pkl` 为全量TTS演示模型，不能用于泛化性能结论。

## 多方言可行性实验

```powershell
python scripts/multidialect_crosslingual.py
python scripts/mixed_training_experiment.py
```

结果文件：
- `outputs/multidialect_crosslingual.json`
- `outputs/mixed_training_experiment.json`
- `docs/multidialect_feasibility_report.md`

## 审计与最终文档

- `docs/project_audit.md`
- `docs/algorithm_experiments.md`
- `docs/dialect_feasibility.md`
- `docs/independent_redesign.md`
- `docs/optimization_report.md`

严格划分实验：`outputs/strict_split_experiment.json`
个性化注册原型：`outputs/enrollment_prototype.json`
数据质量审计：`outputs/data_quality_audit.json`

最终测试命令：

```powershell
python -m unittest discover -s tests -v
```

## 精度优化补充

- 快速特征/DTW变体：`outputs/accuracy_optimization_fast.json`
- 模板选择：`outputs/template_selection_experiment.json`
- 表达留一CV：`outputs/cv_expression_trim.json`
- 音色留一CV：`outputs/cv_voice_trim.json`
- 详细分析：`docs/algorithm_experiments.md`

## 第二阶段：语料扩充到 600 条 + 无泄漏固定测试集

目标：把粤语语料从 120 条扩充到 600 条（六类各 100 条），
建立按「文本表达组」划分的固定测试集，并在其上做 MFCC+DTW 对照实验。

### 完整运行顺序

```powershell
cd D:\Run_Tao\DialectVoice_Research
$env:PYTHONIOENCODING='utf-8'

# 1) 生成扩充语料（480 条，需 TTS 隔离环境）
.\.venv_tts\Scripts\python.exe scripts/generate_yue_corpus_v2.py --plan-only
.\.venv_tts\Scripts\python.exe scripts/generate_yue_corpus_v2.py --execute

# 2) 合并清单 + 质量标记 + 重复/泄漏审计 + 固定划分
python scripts/build_dataset_v2.py

# 3) 对照实验（结果追加合并到同一 run 目录）
python scripts/run_v2_experiments.py --experiments E1 E2 --run-name main_v2
python scripts/run_v2_experiments.py --experiments E3 E4 --run-name main_v2 --seeds 42 7 2026
python scripts/run_v2_experiments.py --experiments B N --run-name main_v2
python scripts/run_v2_experiments.py --experiments F --run-name main_v2

# 4) v1 基线 vs 扩充数据 的配对检验（McNemar + 分组自助法）
python scripts/compare_v1_vs_v2.py

# 5) 拒识评估（阈值只在验证集选择）
python scripts/rejection_v2.py

# 6) 生成报告表格
python scripts/report_v2_tables.py main_v2 > outputs/v2_tables.md
```

### 关键产物

| 路径 | 内容 |
|---|---|
| `data/corpus/yue_expression_library.csv` | 174 条粤语表达库（六类各 29 条，含自然度标注） |
| `data/tts_corpus/manifest_yue_v2.csv` | 600 行统一清单（扩展 13 个字段） |
| `data/splits/track_a_expression_split.csv` | 轨道A 固定划分（按表达组，含哈希） |
| `outputs/v2_generation_log.csv` | 生成日志（480 条，0 失败） |
| `outputs/v2_data_audit.json` | 质量/重复/泄漏/划分审计 |
| `outputs/v2_experiments/main_v2/results.json` | E1/E2/E3/E4/B/N/F 实验结果 |
| `outputs/v2_v1_vs_v2_comparison.json` | v1 vs 扩充 配对比较 |
| `outputs/v2_rejection.json` | 拒识指标 |
| `docs/v2_report.md` | **完整执行报告（含结论与局限）** |

### 重要说明

- 新增 480 条音频全部为合成 TTS（edge-tts 3 个粤语音色），
  **不是真人语音**，不能外推为真实用户场景准确率。
- 全部 174 条粤语文本仅经规则审查（`text_verified_by=RULE_REVIEWED`），
  **仍待母语者核验**；其中 18 条已明确标记为偏书面或含普通话句式。
- 独立粤语音色仅 4 个（canto-tts 只有单一 baked 音色），
  因此「未见音色」实验只能做小规模辅助验证，且无法同时保证「未见文本」。
- 本阶段结论为**负结果**：扩充数据未带来统计显著的准确率提升；
  主要瓶颈是语义对立意图（开/关、拉埋/打開）的方向性混淆。
  详见 `docs/v2_report.md`。
- 新增单元测试：`tests/test_v2_dtw_equivalence.py`、`tests/test_v2_dataset.py`。
  总测试数 65。

# 第三阶段：数据条件扩充 + 方向词/语域瓶颈修复 + CNN 判别模型

二期结论是「数据量不是瓶颈、方向词混淆是瓶颈」。第三阶段因此把重点放在
**声学条件多样性、判别式建模、纠正/否定语域覆盖、拒识与误识别风险**，
并保持与二期相同的固定测试集与统计检验纪律。

## 技术路线（与二期主模型的区别）

| 维度 | 二期 | 三期 |
|---|---|---|
| 声学前端 | MFCC13 + CMVN | log-mel40 + Δ（整句 CMVN） |
| 主模型 | 多模板 DTW 最近邻 | 1D-CNN（6 类，3 种子集成） |
| 数据 | 600 条（3 音色 × 174 表达） | 2904 条（v3 条件增广 + v3.1 correction 表达） |
| 拒识 | DTW 距离 softmax 置信度 | 温度标定 max-prob + 验证集阈值 |
| 评估 | 111 条固定测试集 | 483 条固定测试集（31 组，条件/语域/音色分层） |

固定测试集仍以**表达组**为原子单位（同一表达的所有音色/条件变体整体进入同一子集），
v3/v3.1 的 31 个 test 表达组与二期完全一致，因此可与二期数字直接比较。

## 快速开始（推理 + 演示）

```powershell
cd D:\Run_Tao\DialectVoice_Research
$env:PYTHONIOENCODING='utf-8'
$env:KMP_DUPLICATE_LIB_OK='TRUE'   # 本机 Anaconda 的 MKL/OpenMP 共存需要（仅训练/实验）

# 1) 单条音频推理（v3 CNN 模型，纯 numpy 推理）
python scripts/run_audio_control.py --audio data/tts_corpus/wav16k_v2/edge/LIGHT_ON_02_edge_tts_zh-HK-HiuGaaiNeural_v0_00.wav --model data/models/v3_cnn_ensemble.npz --seq 1

# 2) 六类命令 + 未知指令的 CLI 演示（识别 -> 拒识 -> 模拟设备 -> 状态 -> 日志）
python scripts/demo_v3.py --sequence --model data/models/v3_cnn_ensemble.npz

# 3) Web 演示台（离线，本机）
python scripts/run_web_demo.py --model data/models/v3_cnn_ensemble.npz
# 浏览器打开 http://127.0.0.1:8770/
```

## 训练与正式实验（严格顺序）

```powershell
# 0) 单元测试
python -m unittest discover -s tests

# 1) 训练 3 种子 CNN（仅用 train 训练、val 早停/温度标定；test 不参与）
python scripts/train_v3_model.py --threads 16

# 2) 正式对照实验 E1–E10（v2 基线 / 经典判别式 / CNN / TTA / 融合 / 统计检验 / 错误分析）
python scripts/run_v3_experiments.py --run-name main_v3_1

# 3) 拒识评估（阈值只在 val + 未知校准集选择）
python scripts/evaluate_v3_rejection.py --out outputs/v3_experiments/main_v3_1/rejection.json

# 4) 生成报告表格
python scripts/report_v3_tables.py main_v3_1 > outputs/v3_tables_main_v3_1.md

# 5) 拒识分数对照（max-prob vs 嵌入距离 OOD）
python scripts/evaluate_v3_ood.py

# 6) （可选）数据消融：v3（无 correction 补充）vs v3.1
python scripts/train_v3_model.py --seeds 42 \
  --manifest data/tts_corpus/manifest_yue_v3.csv \
  --split data/splits/track_a_expression_split_v3.csv \
  --out outputs/v3_experiments/data_ablation/v3_only_seed42.npz \
  --report outputs/v3_experiments/data_ablation/train_v3_only.json
python scripts/run_v3_data_ablation.py
```

一键脚本：`powershell -ExecutionPolicy Bypass -File scripts/run_v3_all.ps1`
（`-SkipTrain` 复用已有模型；依赖见 `requirements_v3.txt`）。

## 数据（v3 / v3.1）

- `data/tts_corpus/manifest_yue_v3.csv`（2796 行）+ `data/splits/track_a_expression_split_v3.csv`
- `data/tts_corpus/manifest_yue_v3_1.csv`（2904 行）+ `data/splits/track_a_expression_split_v3_1.csv`
- 条件：`text_morphology / rate_slow / legacy`（二期）+ `rate_slow_18 / rate_fast_18 / pitch_low / pitch_high`（三期）。
  条件变体由 v2 源音频确定性 WSOLA 变速/重采样变调生成，**不是独立 TTS 采样**；
  42 个 WanLung 缺口源与 18 条新表达使用真实 edge-tts 生成。
- v3.1 新增 18 条 correction（否定/纠正）表达（新文本、仅 train、与 test 文本 Jaccard<0.5）。
- 全部为合成语音，文本仅 `RULE_REVIEWED`，未做母语者核验。
- 数据审计：`outputs/v3_data_audit.json`、`outputs/v3_1_data_audit.json`；方法见
  `docs/data_v3.md`、`docs/data_v3_1.md`。

## 关键产物

| 路径 | 内容 |
|---|---|
| `data/models/v3_cnn_ensemble.npz` | v3.1 主模型（3 种子 CNN + JSON meta，纯 numpy 推理） |
| `data/models/v3_thresholds.json` | 拒识阈值（验证集 + 未知校准集选择） |
| `outputs/v3_experiments/main_v3_1/results.json` | E1–E10 全部实验指标 |
| `outputs/v3_experiments/main_v3_1/rejection.json` | 拒识/未知指令指标与逐文本风险 |
| `outputs/v3_experiments/main_v3_1/train_report.json` | 训练协议、数据哈希、每种子 val 曲线 |
| `outputs/v3_tables_main_v3_1.md` | 报告用自动表格 |
| `src/v3_model.py` / `src/v3_cnn_numpy.py` | 运行时推理（纯 numpy，无需 torch） |
| `src/control_server_v3.py` + `scripts/run_web_demo.py` | 控制服务 + Web 演示台 |
| `docs/v3_report.md` / `docs/方言智控盒_第三阶段报告.docx` | **三阶段完整报告与交付文档** |

## 结果摘要

﻿### 识别准确率（固定 test：31 个未见表达组 / 483 条；matched v2 = 与二期相同的 111 条）

| 模型 | Test Acc（full 483） | Macro-F1 | Test Acc（matched v2 111） | 说明 |
|---|---:|---:|---:|---|
| 二期 MFCC+DTW 基线 | 0.5901 | 0.5921 | 0.5495 | 与二期报告 0.5495 逐位一致 |
| 经典判别式（MFCC+Δ+ΔΔ + logreg） | 0.6749 | 0.6706 | 0.7027 | 证明「动态特征+判别式」有效，但仍不足 |
| **CNN 3 种子集成（部署模型）** | **0.9048** | **0.9022** | **0.8829** | 1.35MB，纯 numpy 推理 16–18ms/条 |
| CNN + 经典融合（实验最佳） | **0.9151** | **0.9126** | **0.8919** | val 选 CNN 权重 0.6，未部署 |

配对 McNemar（按表达组自助法 95%CI）：
- v2 基线 → CNN：full ΔAcc +0.3147（p=9.2e-32，CI +0.178~+0.458）；matched ΔAcc +0.3333（p=3.0e-9）
- 经典 → CNN：full ΔAcc +0.2298（p=6.8e-24）；matched ΔAcc +0.1813（p=1.8e-4）

分层（CNN+融合，full test）：语域 short_direct / colloquial / question 均 1.00，
polite 0.9406，**correction 0.6429**；条件 text_morphology 0.942、rate_fast/slow_18 0.936、
pitch_low/high 0.903/0.914、rate_slow 0.833、legacy(canto) 0.792。
错误高度集中：窗帘开/关方向互换（20+13 条）与 correction 句式（35 条）。

数据消融（v3 无 correction 补强 vs v3.1，同种子同配置）：full 0.8778 → 0.9130
（Δ+3.52pp，McNemar p=0.00022）；correction 子集 0.5408 → 0.6429。

### 拒识 / 误识别风险（必须与准确率一起读）

阈值 0.93（仅在验证集 + 未知校准集选择）：已知覆盖率 0.890、接受样本准确率 0.944；
**未知 test 误接受率 0.361（13/36）**，硬负例（開冷氣/關電視/調暗啲燈/開抽氣扇 等）误接受率 0.333。
即：识别准确率大幅提升，但「同动词不同设备」的未知指令仍可能被高置信度接受，
这是当前系统最重要的安全短板。OOD 分数（嵌入余弦）AUC 可达 0.952，但
未知校准集与未知测试集按文本划分存在难度偏移，阈值迁移后 FAR 未改善
（详见 `outputs/v3_experiments/main_v3_1/ood.json`）。下一阶段应引入
拒识类/OOD 训练数据、关键词校验或真实用户负样本。


## 已知限制

- 全部语料为合成 TTS / 合成变换（3 个 edge 粤语音色 + 1 个 canto 遗留音色），
  不是真人粤语，结论不能外推为真实用户准确率。
- rate/pitch 条件变体与源样本高度相关（同一发音的确定性变换），
  不能当作独立样本量；统计上按表达组重采样处理相关性。
- 训练/测试共享全部音色（未做未见音色隔离），音色外推能力未验证。
- 未知指令集 72 条（24 文本 × 3 音色），其中 test 36 条；FAR 的 95% 上界
  约 8.3%（rule of three），观测 0% 不等于可靠安全。
- 设备为模拟 ESP32（串口协议），未接真实硬件；无实时麦克风/VAD。
- v3.1 的 correction 语料补充是在 v3 测试误差分析（2 个窗帘 correction 组全错）
  之后进行的，属于 test-informed 的数据补强；测试集文本未参与训练，
  但报告中已如实标注该改进的发现顺序，不应解读为完全先验的设计。

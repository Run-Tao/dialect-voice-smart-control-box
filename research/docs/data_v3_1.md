# 第三阶段 v3.1：correction 语域训练数据增量说明

本文档说明在已冻结的 v3 数据之上新增的 **v3.1 correction 语域增量**：18 条新 correction
（否定/纠正句式）粤语表达 × 3 音色 × {baseline, rate_slow_18} = 108 条训练样本，全部只进 train。
v3 的三个冻结文件未修改。所有数字由 `scripts/build_dataset_v3_1.py` 从磁盘实际音频重算，
写入 `outputs/v3_1_data_audit.json`；生成日志逐条记录在 `outputs/v3_1_generation_log.csv`。

- 执行日期：2026-10-21（Asia/Shanghai）
- 主环境：`D:\anaconda3\python.exe`；TTS 环境：`.\.venv_tts\Scripts\python.exe`（edge-tts 7.3.1）

## 1. 动机（Lead 的 v3 CNN 误差分析）

v3 固定测试集 31 个表达组中，correction 语域有 6 组（每类 1 组，共 98 条）；
v3 train 132 组中也只有 6 个 correction 组。用 v3 训练的 6 类 CNN 在 test 483 条上总体
准确率 0.913，但 **6 个 correction 测试组只有 0.622**；其中
`CURTAIN_OPEN_14「唔係拉埋窗簾係打開窗簾」`（19 条）与
`CURTAIN_CLOSE_15「唔係拉開係拉埋」`（15 条）**100% 判反**。

这是**训练集语域覆盖不足**造成的明确缺陷，不是测试集噪声。本增量只补充新的 correction
训练表达（每类 3 条），**绝不使用测试集 6 条 correction 文本**；测试集文本/音频未参与训练，
测试集 98 条 correction 样本保持不变。

## 2. 新增的 18 条 correction 表达

写入 `data/corpus/yue_expression_library_v3_1.csv`（= v3 表达库 192 条 + 18 条 = 210 组，
`register=correction`、`text_verified_by=RULE_REVIEWED`、`v3_source=new_v3_1`、
附自然度审查 `yue_naturalness_review` 与 `distractor_terms`（句中反方向干扰词））。

| expression_id | command | 文本 | 干扰词 | 句式 | 自然度 |
|---|---|---|---|---|---|
| LIGHT_ON_NC01 | LIGHT_ON | 唔係熄燈係要開燈 | 熄燈 | 唔係X係Y | HIGH |
| LIGHT_ON_NC02 | LIGHT_ON | 唔好熄燈，開燈先 | 熄燈 | 唔好X，Y先 | HIGH |
| LIGHT_ON_NC03 | LIGHT_ON | 熄咗先唔係，開返燈 | 熄咗 | X咗先唔係，Y | MEDIUM |
| LIGHT_OFF_NC01 | LIGHT_OFF | 唔係開燈係要熄燈 | 開燈 | 唔係X係Y | HIGH |
| LIGHT_OFF_NC02 | LIGHT_OFF | 唔好開燈，熄燈先 | 開燈 | 唔好X，Y先 | HIGH |
| LIGHT_OFF_NC03 | LIGHT_OFF | 開咗先唔係，熄返燈 | 開咗 | X咗先唔係，Y | MEDIUM |
| CURTAIN_OPEN_NC01 | CURTAIN_OPEN | 唔係拉埋係要打開 | 拉埋 | 唔係X係Y | MEDIUM |
| CURTAIN_OPEN_NC02 | CURTAIN_OPEN | 唔好關窗簾，打開窗簾先 | 關窗簾 | 唔好X，Y先 | HIGH |
| CURTAIN_OPEN_NC03 | CURTAIN_OPEN | 拉埋咗先唔係，拉開佢 | 拉埋咗 | X咗先唔係，Y | MEDIUM |
| CURTAIN_CLOSE_NC01 | CURTAIN_CLOSE | 唔係打開係要拉埋 | 打開 | 唔係X係Y | MEDIUM |
| CURTAIN_CLOSE_NC02 | CURTAIN_CLOSE | 唔好開窗簾，關埋窗簾先 | 開窗簾 | 唔好X，Y先 | HIGH |
| CURTAIN_CLOSE_NC03 | CURTAIN_CLOSE | 打開咗先唔係，拉埋佢 | 打開咗 | X咗先唔係，Y | MEDIUM |
| ALL_OFF_NC01 | ALL_OFF | 唔係淨係熄燈，全部都要關 | 淨係熄燈 | 唔係X係Y/全部都要關 | HIGH |
| ALL_OFF_NC02 | ALL_OFF | 唔好只關燈，全部熄晒佢 | 只關燈 | 唔好X，Y | HIGH |
| ALL_OFF_NC03 | ALL_OFF | 熄燈唔夠，全部關晒 | 熄燈 | X唔夠，Y | HIGH |
| STATUS_ALL_NC01 | STATUS_ALL | 唔係開燈，係問下狀態 | 開燈 | 唔係X係Y | HIGH |
| STATUS_ALL_NC02 | STATUS_ALL | 唔好熄燈，講下設備狀態 | 熄燈 | 唔好X，Y | HIGH |
| STATUS_ALL_NC03 | STATUS_ALL | 唔係叫我開燈，係查狀態 | 開燈 | 唔係叫我X，係Y | HIGH |

句式覆盖：`唔係X係Y`（6 条）、`唔好X，Y先`（5 条）、`X咗先唔係…`（4 条）、
`X唔夠，Y` / 变体（3 条）。每条同时包含反方向干扰词，最终意图为 Y。

## 3. 生成方法

- **baseline（54 条）**：真实 edge-tts，3 音色（HiuGaai/HiuMaan/WanLung），rate=0 / pitch=0。
  新文本在 v2/v3 中不存在，无法离线增广，必须真实合成。
  异步并发 6、单条失败重试 3 次（退避 3/6/9s）、幂等（已存在且校验通过则跳过）、可断点续跑。
- **rate_slow_18（54 条）**：对每条 baseline 做离线 WSOLA 时域伸缩（alpha=1/0.82=1.21951，
  保音高），与 v3 的条件增广方法一致；不是独立 TTS 采样。
- 输出统一 16 kHz 单声道 PCM16，存 `data/tts_corpus/wav16k_v3_1/edge/`。
- 生成日志：`outputs/v3_1_generation_log.csv`（108 行，逐条 stage/attempts/status/sha256/error）。

生成统计（本次实测）：baseline 54/54 成功（32.9 s，失败 0，全部 attempts=1）；离线增广 54/54
成功；总耗时 106 s；失败/排除日志 0 条，未静默丢弃任何样本。

## 4. 清单与固定划分

- `data/tts_corpus/manifest_yue_v3_1.csv` = v3 2796 行（**原样保留**，仅在末尾追加 `data_version=v3`
  列）+ 108 新行（`data_version=v3.1`），共 **2904 行 / 210 个表达组**，每命令 484 条。
- `data/splits/track_a_expression_split_v3_1.csv` = v3 192 组（subset 100% 继承）
  + 18 个新组（全部 train）= **210 组：train 150 / val 29 / test 31**。
- 行数：train 1981（v3 1873 + 108）、val 440、test 483（**测试集行数/组数不变**）。
- correction 覆盖：train correction 组 **6 → 24 组**（行 106 → 214）；val 0；test 6 组/98 条不变。
- 新组 `split_source=added_train_v3_1`；`register=correction`。

## 5. 相似度检查（与测试集 correction 文本）

指标：规范化文本（去标点、繁简统一）的 2-gram Jaccard。18 条新文本 vs 6 条测试 correction 文本：

| expression_id | 文本 | 最大 Jaccard | 最相似测试文本 |
|---|---|---|---|
| ALL_OFF_NC01 | 唔係淨係熄燈，全部都要關 | 0.429 | ALL_OFF |
| ALL_OFF_NC02 | 唔好只關燈，全部熄晒佢 | 0.188 | ALL_OFF |
| ALL_OFF_NC03 | 熄燈唔夠，全部關晒 | 0.133 | ALL_OFF |
| CURTAIN_CLOSE_NC01 | 唔係打開係要拉埋 | 0.333 | CURTAIN_CLOSE |
| CURTAIN_CLOSE_NC02 | 唔好開窗簾，關埋窗簾先 | 0.214 | CURTAIN_OPEN |
| CURTAIN_CLOSE_NC03 | 打開咗先唔係，拉埋佢 | 0.308 | CURTAIN_OPEN |
| CURTAIN_OPEN_NC01 | 唔係拉埋係要打開 | 0.333 | CURTAIN_CLOSE |
| CURTAIN_OPEN_NC02 | 唔好關窗簾，打開窗簾先 | 0.214 | CURTAIN_OPEN |
| CURTAIN_OPEN_NC03 | 拉埋咗先唔係，拉開佢 | 0.444 | CURTAIN_CLOSE |
| LIGHT_OFF_NC01 | 唔係開燈係要熄燈 | 0.400 | LIGHT_OFF |
| LIGHT_OFF_NC02 | 唔好開燈，熄燈先 | 0.083 | LIGHT_OFF |
| LIGHT_OFF_NC03 | 開咗先唔係，熄返燈 | 0.400 | LIGHT_OFF |
| LIGHT_ON_NC01 | 唔係熄燈係要開燈 | 0.400 | LIGHT_OFF |
| LIGHT_ON_NC02 | 唔好熄燈，開燈先 | 0.083 | LIGHT_OFF |
| LIGHT_ON_NC03 | 熄咗先唔係，開返燈 | 0.400 | LIGHT_ON |
| STATUS_ALL_NC01 | 唔係開燈，係問下狀態 | 0.364 | LIGHT_OFF |
| STATUS_ALL_NC02 | 唔好熄燈，講下設備狀態 | 0.067 | LIGHT_ON |
| STATUS_ALL_NC03 | 唔係叫我開燈，係查狀態 | 0.231 | LIGHT_OFF |

**最大 0.4444（< 0.5），违反阈值条数 = 0**；新文本与 v3 表达库 192 条无规范化重复。
补充：新文本与 **train** 已有 correction 文本存在同语域近似（如 LIGHT_ON_NC01 与 LIGHT_ON_14
Jaccard≈0.625），这是训练集内部冗余，不构成测试泄漏，已记录于审计 limitations。

## 6. 数据审计（`outputs/v3_1_data_audit.json`）

| 指标 | 结果 |
|---|---|
| 清单总行数 | 2904 = v3 2796 + 新增 108 |
| 表达组 | 210（v2 174 + v3 18 + 新 correction 18） |
| 每命令样本 | 6 类各 484 |
| 划分 | train 150 组/1981 行、val 29/440、test 31/483 |
| 新增样本条件 | baseline 54（真实 TTS）+ rate_slow_18 54（离线增广） |
| 新增样本音色 | 3 音色各 36 条 |
| 跨子集泄漏（组/文本/sha/utterance） | 12 项组合全部 = 0 |
| sha256 重复 | 0 |
| v3.1 vs v3 音频 sha 重叠 | 0 |
| 质量 | 2904 行全部 ok（excluded 0 / flagged 0） |
| v3 冻结文件 | 3 个文件 sha256 复核 unchanged=true |
| Jaccard（vs 测试 correction） | 最大 0.4444，<0.5 违反 0 |
| manifest sha256 | `eb9b60140047f2bfb2eb9fed834f94e55648d645e9e7ee0b5572a0139916aa02` |
| split sha256 | `97a9eef8041ca7272ec26fcfdad33e01729df2947adcb3f405a1baceb7409620` |
| library sha256 | `d755809f847b2ef1b6c379a8eb570de8834c2221795ab31a72a0a13d5ac65851` |

## 7. 局限

1. 全部为合成 TTS / 合成变换音频，非真人粤语；RULE_REVIEWED + 书面自然度审查，未做母语者听测
   （6 条标 MEDIUM：三个方向类 NC01 与 NC03 各 1 条，句式略省略宾语）。
2. `rate_slow_18` 由 baseline 离线 WSOLA 变换得到，与 baseline 高度相关，不是独立 TTS 采样。
3. 本增量只补训练集语域覆盖；**未验证**重训后的 correction 准确率提升幅度（属模型侧实验）。
4. 测试集 6 条 correction 文本未参与训练；新文本与其 Jaccard <0.5，但语义上与测试文本同属
   「先反后正」句式，测试集仍是未见表达评估，结论需在重训实验中对 6 个 correction 测试组单独报告。

## 8. 复现

```powershell
$env:PYTHONIOENCODING='utf-8'; cd D:\Run_Tao\DialectVoice_Research
.\.venv_tts\Scripts\python.exe scripts\generate_v3_1_corpus.py --workers 6 --retries 3   # 54 TTS + 54 增广
D:\anaconda3\python.exe scripts\build_dataset_v3_1.py                                    # 清单/划分/审计
D:\anaconda3\python.exe -m unittest tests.test_v3_1_dataset -v
D:\anaconda3\python.exe -m unittest discover -s tests
```

- 联网：仅 `generate_v3_1_corpus.py` 的 TTS 阶段需要（54 条，本次 32.9 s）；其余离线。
- 预计耗时：生成 ≈ 106 s、build ≈ 2 s、测试 ≈ 13 s（本机实测）。
- 幂等：已存在且校验通过的音频 `skipped_existing`，失败重试 3 次，可断点续跑。

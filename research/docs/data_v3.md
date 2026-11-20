# 第三阶段 v3 语音数据扩充说明（data_v3）

本文档说明「方言智控盒」第三阶段（v3）粤语命令语料扩充的**实际做法、回退原因、数据审计结果与局限**。
所有数字均由 `scripts/build_dataset_v3.py` 从磁盘实际音频重新计算，写入 `outputs/v3_data_audit.json`；
生成日志逐条记录在 `outputs/v3_generation_log.csv`，本文件不手填任何计数。

- 项目根：`D:\Run_Tao\DialectVoice_Research`
- 主环境：`D:\anaconda3\python.exe`（numpy 2.3.5）
- TTS 环境：`.\.venv_tts\Scripts\python.exe`（edge-tts 7.3.1 / soundfile 0.14.0）
- 执行日期：2026-10-21（Asia/Shanghai）

---

## 1. 目标与只读约束

第三阶段的原始需求是「同一批 174 个表达组 × {HiuGaai, HiuMaan, WanLung} 3 个音色 ×
{rate_slow_18, rate_fast_18, pitch_low, pitch_high} 4 个条件 ≈ 2088 条」，用于增加训练条件的
语速/音高多样性（二期结论：数据量不是瓶颈，方向词混淆与条件多样性不足才是）。

本阶段**严格只读**二期产物：

| 只读对象 | 路径 | 状态 |
|---|---|---|
| v2 清单 | `data/tts_corpus/manifest_yue_v2.csv` | 未修改（sha256 `e89f3a00…f3da7`） |
| v2 音频 | `data/tts_corpus/wav16k_v2/`、`data/tts_corpus/wav16k/` | 未修改 |
| v2 固定划分 | `data/splits/track_a_expression_split.csv` | 字节级冻结，sha256 `b9b42403…5c0b34`（与二期审计一致） |
| v2 表达库 | `data/corpus/yue_expression_library.csv` | 未修改（v3 另存为 `_v3.csv`） |
| 既有代码/模型/文档 | `src/` 既有文件、`data/models/`、`config.json`、`README.md`、`docs/` 既有文档 | 未修改 |

v3 只新增：`data/`（新清单/新划分/新音频/新表达库）、`scripts/generate_v3_corpus.py`、
`scripts/generate_unknown_v3.py`、`scripts/build_dataset_v3.py`、`src/dataset_v3.py`、
`tests/test_v3_dataset.py`、`outputs/v3_generation_log.csv`、`outputs/v3_data_audit.json`、本文件。

**冻结文件（字节级，交接给实验侧后不再改写）**：

| 文件 | sha256 |
|---|---|
| `data/tts_corpus/manifest_yue_v3.csv` | `fb2da56e82825e611c30bba5288462006a47969c38f147b39ad52b72b47315e0` |
| `data/splits/track_a_expression_split_v3.csv` | `15c057bfdad69ed8833fb4b06f2a3f027c7d9f3a224a380b13c57d897de882dd` |
| `data/splits/track_a_expression_split.csv`（v2，只读） | `b9b42403525e80b9678b95faedc196a911dc9047b5d2d8cf422f8f7b1f5c0b34` |

---

## 2. 网络实测与回退决策（关键）

### 2.1 实测

在正式生成前用与正式生成完全相同的 edge-tts 调用方式做了三组并发基准（同一天、同一网络，
粤语音色 + rate=−18%），结果如下：

| 并发数 | 请求数 | 失败数 | 失败率 | 墙钟耗时 | 有效吞吐 | 成功请求延迟（mean/median/max） |
|---|---|---|---|---|---|---|
| 6 | 24 | 10 | 41.7% | 115.3 s | 0.121 条/s | 21.7 / 23.2 / 34.8 s |
| 3 | 9 | 2 | 22.2% | 89.6 s | 0.078 条/s | 20.4 / 22.0 / 33.9 s |
| 1 | 9 | 3 | 33.3% | 162.2 s | 0.037 条/s | 9.0 / 7.6 / 22.3 s |

失败形态统一为 `edge_tts.exceptions.NoAudioReceived`（WebSocket 握手成功但服务端不下发音频）。
网络本身可用（HTTP 探测 0.9 s、WebSocket 建连 <1 s），但**语音合成通路被限流/不稳定**。

**正式生成批次的补充观测（如实记录）**：在基准测试之后、正式跑 168 条 baseline 时网络状况变好，
v3 TTS 批次 96 条（42 缺口源 + 54 新表达）实际 95 条成功 + 1 条幂等沿用（0 失败），耗时 202.6 s，
约 2.1 s/条；未知集 72 条全部成功，耗时 59.8 s。即网络是**间歇性不稳定**：基准窗口失败率
22%–42%，正式窗口接近 0%。回退决策依任务书的**预登记规则**（失败率 >5%）在基准窗口作出，
并未因后续窗口变好而回滚；两组数字都保留，避免误导。

### 2.2 决策

任务书第 3 条要求：「若 30 分钟内网络不可用或失败率 >5%，立即改为离线信号级增广方案」。
基准窗口失败率 22.2%–41.7%，远超 5% 阈值；若 2088 条全量走网络，按实测吞吐需 1–16 小时且
仍可能失败。因此**触发回退**：

- **批量条件变体（2088 条）不调用网络**，改为对 v2 同表达、同音色、语速 0%/音高 0 的基准音频
  做**确定性离线信号级增广**（方法见第 3 节）。
- 真实 edge-tts 只用于**没有 v2 音源、无法离线增广**的文本（共 168 条 baseline）：
  1. 18 条新表达 × 3 音色 = 54 条；
  2. 24 条未知/易混文本 × 3 音色 = 72 条（`scripts/generate_unknown_v3.py`）；
  3. 42 个 (表达组, WanLung) 组合在 v2 中缺音源，先生成 baseline 再离线增广 = 42 条。
- 真实 TTS 采用「异步并发 ≤6 + 单条最多重试 3 次（退避 3s/6s/9s）+ 失败不中断整体 + 幂等续跑」。
  逐条状态/重试次数/错误记录在 `outputs/v3_generation_log.csv`，汇总见
  `outputs/v3_data_audit.json` 的 `generation` 段（`by_stage_status`、`failed_samples`）。

> 回退只针对「用 v2 音源做信号变换可得到等价条件」的批量变体。新文本必须合成，
> 无法用 v2 音频变换得到，因此仍需真实 TTS；这不是伪造数据，每条新样本的
> `generation_method=edge_tts`、来源文本与 sha256 均可审计。

---

## 3. 实际生成方法

### 3.1 v3 条件变体（离线信号级增广，2088 条）

**源音频选择**：v2 清单中每个 (表达组, 音色) 组合取一条「语速 0%、音高 0%」的 edge-tts 基准音频
（v2 `text_morphology` 行或 v1 `legacy` 行），共 480 条；WanLung 缺失的 42 组用真实 TTS 补齐 baseline，
合计 522 个源。每个源生成 4 个条件变体 = 2088 条。

| 条件 | 变换 | 目标参数 | 时长变化 | 音高变化 |
|---|---|---|---|---|
| `rate_slow_18` | WSOLA 时域伸缩（保音高） | speed=0.82 | ×1.2195 | 不变 |
| `rate_fast_18` | WSOLA 时域伸缩（保音高） | speed=1.18 | ×0.8475 | 不变 |
| `pitch_low` | 重采样变调 + WSOLA 还原时长 | 基频 −15 Hz | 不变 | −15 Hz |
| `pitch_high` | 重采样变调 + WSOLA 还原时长 | 基频 +15 Hz | 不变 | +15 Hz |

- WSOLA：50 ms Hann 帧、12.5 ms 合成 hop、±12.5 ms 相似度搜索（步长 8 样本）、75% 重叠相加。
- 变调：先按 m=(f0±15)/f0 用 Kaiser 窗 sinc 插值重采样（调 ×m、时长 ÷m），再用 WSOLA 以 alpha=m
  拉伸回原时长；f0 用自相关法逐条估计（60–400 Hz，取浊音帧中位数），估计失败时按音色默认值
  （女声 210–220 Hz / 男声 125 Hz）。实际 f0、倍率写入清单 `condition_detail`。
- 输出统一 **16 kHz、单声道、PCM16**；变换**无随机成分**（同输入必得同字节），便于复现与测试。
- 条件与 v2 不重复：v2 只有 rate=±12%（`rate_slow`）等条件，v3 为 ±18% / ±15 Hz 且经离线变换，
  sha256 与 (文本, 音色, 条件) 键均无交集（审计 `duplicates` 段）。

代码：`src/dataset_v3.py`（`wsola_time_stretch` / `resample_sinc` / `pitch_shift_hz` /
`augment_condition`），落盘：`scripts/generate_v3_corpus.py --mode augment`。

### 3.2 条件参数与实测验证（`outputs/v3_data_audit.json → condition_verification`）

验证方式：对每个增广样本重新读盘计算时长、RMS、sha256，并与 `augment_from` 指向的源音频比对。
时长比 = 输出时长/源时长；f0 用自相关估计（客观信号指标，非人工听测）；"与源 sha 相同"条数应恒为 0。

| 条件 | 条数 | 期望时长比 | 实测时长比 mean（min–max） | 2% 内条数 | 与源 sha 相同 | 格式违规 | RMS mean（min–max） | f0 实测偏移 mean（p05–p95） |
|---|---|---|---|---|---|---|---|---|
| rate_slow_18 | 576 | 1.21951 | 1.2195（1.2195–1.2195） | 576/576 | 0 | 0 | 0.0720（0.0415–0.1096） | — |
| rate_fast_18 | 522 | 0.84746 | 0.8475（0.8474–0.8475） | 522/522 | 0 | 0 | 0.0714（0.0422–0.1094） | — |
| pitch_low | 522 | 1.00000 | 1.0000（1.0000–1.0000） | 522/522 | 0 | 0 | 0.0732（0.0426–0.1094） | −14.61 Hz（−20.43 ~ −8.28） |
| pitch_high | 522 | 1.00000 | 1.0000（1.0000–1.0000） | 522/522 | 0 | 0 | 0.0729（0.0414–0.1089） | +15.13 Hz（+7.19 ~ +22.84） |

> 注：`rate_slow_18` 为 576 条（522 个 v2 组源 + 54 条新表达 baseline 的变慢版本）；
> f0 偏移的 mean 与 p05–p95 接近 ±15 Hz，个别样本因清音/噪声导致 f0 估计失真出现离群值，
> 因此用中位趋势而非单条最小值判断。时长比由清单实测时长计算，故只保留 4 位小数。

### 3.3 新表达（18 条，仅进入 train）

`data/corpus/yue_expression_library_v3.csv` = v2 表达库 174 条 + 新增 18 条（每类 3 条，口语化），
全部标 `text_verified_by=RULE_REVIEWED`，新增条目附 `yue_naturalness_review`（HIGH/MEDIUM 自然度审查）
与 `v3_source=new_v3`。新文本已与 v2 全部 174 条文本做规范化查重（0 重复）。

| 表达组 | 命令 | 文本 | 语域 | 自然度审查 |
|---|---|---|---|---|
| LIGHT_ON_30/31/32 | LIGHT_ON | 快啲開燈 / 開返盞燈先 / 唔該同我開燈 | colloquial / colloquial / polite | HIGH / HIGH / HIGH |
| LIGHT_OFF_30/31/32 | LIGHT_OFF | 快啲熄燈 / 熄返盞燈先 / 唔該同我閂燈 | colloquial / colloquial / polite | HIGH / HIGH / HIGH |
| CURTAIN_OPEN_30/31/32 | CURTAIN_OPEN | 快啲拉開窗簾 / 拉開個窗簾先 / 唔該同我打開窗簾 | colloquial / colloquial / polite | HIGH / HIGH / HIGH |
| CURTAIN_CLOSE_30/31/32 | CURTAIN_CLOSE | 快啲拉埋窗簾 / 拉埋個窗簾先 / 唔該同我閂窗簾 | colloquial / colloquial / polite | HIGH / HIGH / MEDIUM |
| ALL_OFF_30/31/32 | ALL_OFF | 快啲全部熄晒 / 全部關晒先 / 唔該同我全部關晒 | colloquial / colloquial / polite | HIGH / HIGH / HIGH |
| STATUS_ALL_30/31/32 | STATUS_ALL | 快啲講下而家咩狀態 / 睇下狀態先 / 唔該同我查下設備 | colloquial / colloquial / polite | HIGH / HIGH / MEDIUM |

每条新表达生成 3 音色真实 baseline（54 条），并用同一离线增广生成 `rate_slow_18`（54 条），
共 108 条；因文本在 v2 中不存在，不会与 v2 测试集产生文本/音频泄漏。

### 3.4 未知 / 易混指令测试集（24 文本 × 3 音色 = 72 条）

目的：评估拒识与误识别风险。文本均**不属于六类命令**，但包含硬负例（開冷氣 / 關電視 /
調暗啲燈 / 開窗 等），全部为真实 edge-tts baseline（rate 0 / pitch 0），不与主语料文本重复。

清单字段（`data/tts_corpus/unknown_v3.csv`，20 列）：
`utterance_id, audio_path, text, text_norm, text_group_id, text_split, label, negative_type,
voice_id, command, condition, tts_engine, generation_method, sha256, duration, sample_rate,
channels, quality_status, quality_reasons, notes`。

分层字段：

- `label ∈ {hard_negative, general_unknown}`：
  - `hard_negative`（22 文本，66 条）：复用六类命令的「開/關/拉/熄/調」方向词，或对象就在灯光/窗帘/窗
    语义场内，仅设备或动作维度不同，最易被误识别为六类之一；
  - `general_unknown`（2 文本，6 条，較大聲啲/較細聲啲）：不含设备通断方向词、属音量调节维度，
    词面与六类命令距离最远。
- `negative_type`：进一步细分设备类型（aircon/tv/fan/light_brightness/window_vs_curtain/door/
  dehumidifier/water_heater/music/exhaust_fan/fridge/volume_adjust），供拒识实验二次分层。
- `text_split`：按规范化文本 sha256 指纹排序后**按文本对半分**，同一文本 3 个音色整体同侧：
  calibration 12 文本 / 36 条，test 12 文本 / 36 条。hard_negative 在两侧各 33 条，
  general_unknown 各 3 条。
- 质量：72 条全部 `quality_status=ok`（由 `src.dataset_v2.quality_flags_for` 实测重算）。

24 条文本与分层（`label` / `negative_type` / `text_split`）：

| 组 | 文本 | label | negative_type | split |
|---|---|---|---|---|
| UNK_V3_01 | 開冷氣 | hard_negative | device_onoff_aircon | calibration |
| UNK_V3_02 | 關冷氣 | hard_negative | device_onoff_aircon | calibration |
| UNK_V3_03 | 開電視 | hard_negative | device_onoff_tv | calibration |
| UNK_V3_04 | 關電視 | hard_negative | device_onoff_tv | test |
| UNK_V3_05 | 開風扇 | hard_negative | device_onoff_fan | calibration |
| UNK_V3_06 | 關風扇 | hard_negative | device_onoff_fan | test |
| UNK_V3_07 | 調暗啲燈 | hard_negative | light_brightness | test |
| UNK_V3_08 | 調光啲燈 | hard_negative | light_brightness | test |
| UNK_V3_09 | 開窗 | hard_negative | window_vs_curtain | calibration |
| UNK_V3_10 | 關窗 | hard_negative | window_vs_curtain | calibration |
| UNK_V3_11 | 開門 | hard_negative | device_onoff_door | calibration |
| UNK_V3_12 | 關門 | hard_negative | device_onoff_door | calibration |
| UNK_V3_13 | 開抽濕機 | hard_negative | device_onoff_dehumidifier | calibration |
| UNK_V3_14 | 關抽濕機 | hard_negative | device_onoff_dehumidifier | calibration |
| UNK_V3_15 | 開熱水爐 | hard_negative | device_onoff_water_heater | calibration |
| UNK_V3_16 | 關熱水爐 | hard_negative | device_onoff_water_heater | test |
| UNK_V3_17 | 較大聲啲 | general_unknown | volume_adjust | test |
| UNK_V3_18 | 較細聲啲 | general_unknown | volume_adjust | calibration |
| UNK_V3_19 | 開音樂 | hard_negative | device_onoff_music | test |
| UNK_V3_20 | 關音樂 | hard_negative | device_onoff_music | test |
| UNK_V3_21 | 開抽氣扇 | hard_negative | device_onoff_exhaust_fan | test |
| UNK_V3_22 | 關抽氣扇 | hard_negative | device_onoff_exhaust_fan | test |
| UNK_V3_23 | 開雪櫃 | hard_negative | device_onoff_fridge | test |
| UNK_V3_24 | 關雪櫃 | hard_negative | device_onoff_fridge | test |

---

## 4. 清单字段与合并规则

`data/tts_corpus/manifest_yue_v3.csv` 字段 = v2 清单字段 + v2 追加的 `register/split/group_alias`
+ v3 追加：

| 字段 | 说明 |
|---|---|
| `condition_detail` | 条件实现细节（edge-tts 参数或离线增广的 f0/倍率/时域伸缩因子） |
| `generation_method` | `v1_legacy_tts` / `v2_tts` / `edge_tts` / `offline_signal_aug` |
| `augment_from` | 离线增广的源样本 `utterance_id`（可逐条回溯到 v2 音频） |
| `augment_params` | 离线增广参数（JSON 单行） |
| `text_norm` | 规范化文本（去标点、繁简统一），用于泄漏检测 |
| `split_source` | `inherited_v2` / `added_train` |

合并与审计规则（`scripts/build_dataset_v3.py`）：

1. v2 的 600 行原样读入（`generation_batch=v1/v2`），v3 新行来自生成日志中状态成功的条目
   （`generation_batch=v3`），二者 `utterance_id` 拼接后唯一；
2. 每行**重新读取磁盘音频**计算 sha256/时长/采样率/声道/质量（复用
   `src.dataset_v2.analyze_audio`、`quality_flags_for` 阈值），不用清单声明值；
3. sha256 完全重复只保留一条（本次为 0 条）；
4. 按 `(command, normalize_text)` 统一表达组，防止同文本跨子集（本次 0 合并）；
5. 固定划分：v2 的 174 个表达组严格继承 `track_a_expression_split.csv` 的 subset，
   18 个新表达组全部 `train`（不改动 v2 的 val/test）。

---

## 5. 固定划分 v3

`data/splits/track_a_expression_split_v3.csv`（sha256 `15c057bf…de882dd`）：

| subset | 表达组数 | 样本数 |
|---|---|---|
| train | 132（v2 114 + 新 18） | 1873 |
| val | 29 | 440 |
| test | 31 | 483 |
| 合计 | 192 | 2796 |

- 文件另存 `split_source` 列区分 `inherited_v2` / `added_train`；
- v2 的 174 组 subset **100% 继承**（mismatches=0），v2 划分文件 sha256 未变；
- 文件写出后计算 sha256 记录于审计 JSON；实验只读取该文件，不重新划分；
- 划分可由「v2 清单 + v2 划分 + 18 条新表达」确定性重建（`tests/test_v3_dataset.py`
  的 `test_split_reproducible_from_fixed_inputs` 验证）。

---

## 6. 数据审计结果（摘要）

完整数字见 `outputs/v3_data_audit.json`。关键结论（全部由磁盘音频实测重算）：

| 指标 | 数值 |
|---|---|
| 清单总行数 | 2796（v2 600 + v3 新增 2196） |
| 可用 / 排除 / flagged | 2796 / 0 / 0 |
| 表达组 | 192（v2 174 + 新 18） |
| 每命令样本数 | 6 类各 466（均衡） |
| 跨子集泄漏（组/文本/sha256/utterance） | 全部 0 |
| manifest 内 sha256 重复 | 0 |
| v3 与 v2 音频 sha256 重叠 | 0 |
| v3 与 v2 条件键冲突 | 0 |
| 未知集 | 72 条 / 24 文本 / calibration 36、test 36 / sha 重复 0 / 与主语料文本互斥 |
| manifest sha256 | `fb2da56e…15e0` |
| 划分 sha256 | `15c057bf…de882dd` |

生成方式分组（主清单 2796 行）：

| generation_method | 行数 | 说明 |
|---|---|---|
| `v1_legacy_tts` | 120 | v2 清单继承的 v1 粤语 TTS |
| `v2_tts` | 480 | v2 edge-tts |
| `edge_tts` | 54 | 新表达真实 edge-tts baseline（仅这 54 行进主清单） |
| `offline_signal_aug` | 2142 | 离线信号级增广 = 2088 条件变体 + 54 新表达变慢 |
| 合计 | 2796 | — |

另有主清单之外的真实 TTS：42 条 WanLung 缺口源（仅作增广源，不单独作为样本）+ 72 条未知集
（独立清单 `unknown_v3.csv`）。**真实 TTS 共 168 条**（54 + 42 + 72），**离线增广 2142 条**。

### 6.1 新条件与 v2 是否重复

- sha256：v3 新增样本 2196 条的 sha256 与 v2 600 条无交集（0）；
- 条件键 `(text_norm, voice_id, condition)`：v3 与 v2 无冲突（0）；
- v2 只有 `text_morphology / legacy / rate_slow(rate=-12%)`，v3 为 `rate_slow_18(-18%) /
  rate_fast_18(+18%) / pitch_low(-15Hz) / pitch_high(+15Hz) / baseline`，命名与参数均不同。

---

## 7. v2 → v3 数据量对比（按 command × split × condition）

| command | split | v2 text_morph | v2 rate_slow | v2 legacy | v3 rate_slow_18 | v3 rate_fast_18 | v3 pitch_low | v3 pitch_high | v3 baseline | v2 小计 | v3 小计 | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LIGHT_ON | train | 41 | 12 | 16 | 66 | 57 | 57 | 57 | 9 | 69 | 246 | 315 |
| LIGHT_ON | val | 10 | 0 | 0 | 12 | 12 | 12 | 12 | 0 | 10 | 48 | 58 |
| LIGHT_ON | test | 14 | 3 | 4 | 18 | 18 | 18 | 18 | 0 | 21 | 72 | 93 |
| LIGHT_OFF | train | 39 | 12 | 16 | 66 | 57 | 57 | 57 | 9 | 67 | 246 | 313 |
| LIGHT_OFF | val | 12 | 3 | 4 | 15 | 15 | 15 | 15 | 0 | 19 | 60 | 79 |
| LIGHT_OFF | test | 14 | 0 | 0 | 15 | 15 | 15 | 15 | 0 | 14 | 60 | 74 |
| CURTAIN_OPEN | train | 46 | 6 | 8 | 66 | 57 | 57 | 57 | 9 | 60 | 246 | 306 |
| CURTAIN_OPEN | val | 11 | 3 | 4 | 15 | 15 | 15 | 15 | 0 | 18 | 60 | 78 |
| CURTAIN_OPEN | test | 8 | 6 | 8 | 15 | 15 | 15 | 15 | 0 | 22 | 60 | 82 |
| CURTAIN_CLOSE | train | 38 | 15 | 20 | 66 | 57 | 57 | 57 | 9 | 73 | 246 | 319 |
| CURTAIN_CLOSE | val | 14 | 0 | 0 | 15 | 15 | 15 | 15 | 0 | 14 | 60 | 74 |
| CURTAIN_CLOSE | test | 13 | 0 | 0 | 15 | 15 | 15 | 15 | 0 | 13 | 60 | 73 |
| ALL_OFF | train | 45 | 6 | 8 | 66 | 57 | 57 | 57 | 9 | 59 | 246 | 305 |
| ALL_OFF | val | 11 | 3 | 4 | 15 | 15 | 15 | 15 | 0 | 18 | 60 | 78 |
| ALL_OFF | test | 9 | 6 | 8 | 15 | 15 | 15 | 15 | 0 | 23 | 60 | 83 |
| STATUS_ALL | train | 41 | 12 | 16 | 66 | 57 | 57 | 57 | 9 | 69 | 246 | 315 |
| STATUS_ALL | val | 13 | 0 | 0 | 15 | 15 | 15 | 15 | 0 | 13 | 60 | 73 |
| STATUS_ALL | test | 11 | 3 | 4 | 15 | 15 | 15 | 15 | 0 | 18 | 60 | 78 |
| **合计** | 全部 | 390 | 90 | 120 | 576 | 522 | 522 | 522 | 54 | 600 | 2196 | 2796 |

说明：新表达（108 条）只进入 train，故 train 的 v3 条件样本含 54 条 baseline；val/test
无新表达样本，v2 固定测试集不变。

---

## 8. 质量与局限

1. **合成数据**：v3 全部为 TTS 合成 / 对合成音频的信号变换，`quality_flags` 标注
   `SYNTHETIC;UNVERIFIED_BY_NATIVE`，不是真实粤语发音人录音。
2. **离线增广不是独立 TTS 发音（重要）**：4 个条件变体由同一源音频经 WSOLA 变速 / 重采样变调
   得到，与源样本**高度相关**（同一说话人、同一韵律基线的确定性变换），不具备独立 TTS 采样
   的韵律/停顿多样性。训练时若把它们当作完全独立样本，会高估有效数据量与泛化能力；
   建议按 `augment_from` 分组做训练/验证统计，或做「源样本去重」的敏感性分析。
   WSOLA/sinc 还会引入轻微相位与高频失真（质量阈值未触发告警）。
3. **音色覆盖缺口**：v2 中 42 个表达组没有 WanLung 音频，已用真实 TTS 补齐 baseline 后增广；
   补的是 2026-10-21 当次 TTS 结果，重跑可能得到不同音频（日志会更新 sha）。
4. **网络环境依赖**：真实 TTS 只覆盖 168 条 baseline，仍受当时网络影响；失败重试与最终状态见
   `v3_generation_log.csv`，未成功条目不会写入清单（本次 168 条全部成功）。
5. **未知集划分**：24 条文本按哈希对半分为 calibration/test，确定性但样本量小，
   只适合拒识风险的方向性评估；hard_negative/general_unknown 分层不平衡（22:2 文本），
   分层报告时 general_unknown 侧统计功效有限。
6. **新表达自然度**：18 条为规则审查（RULE_REVIEWED）+ 书面自然度评估，未经粤语母语者逐条听测；
   2 条标 MEDIUM（閂窗簾 / 查下設備）。
7. **未验证**：本阶段未做「增广数据训练后的识别精度是否提升」的实验（属模型团队后续工作），
   本文件只对数据本身负责。

---

## 9. 从零重建（一条命令 / 顺序 / 耗时 / 联网）

前提：已存在二期产物（v2 清单/音频/划分/表达库）与本阶段脚本；Python 环境见文首。

**一条命令（PowerShell，在项目根执行）**：

```powershell
$env:PYTHONIOENCODING='utf-8'; cd D:\Run_Tao\DialectVoice_Research; D:\anaconda3\python.exe scripts\generate_v3_corpus.py --mode probe; .\.venv_tts\Scripts\python.exe scripts\generate_v3_corpus.py --mode tts --workers 6 --retries 3; .\.venv_tts\Scripts\python.exe scripts\generate_unknown_v3.py --workers 6 --retries 3; D:\anaconda3\python.exe scripts\generate_v3_corpus.py --mode augment --workers 6; D:\anaconda3\python.exe scripts\build_dataset_v3.py; D:\anaconda3\python.exe -m unittest tests.test_v3_dataset -v; D:\anaconda3\python.exe -m unittest discover -s tests
```

或拆成可观察的分步（推荐先 probe 再全量）：

```powershell
$env:PYTHONIOENCODING='utf-8'; cd D:\Run_Tao\DialectVoice_Research
D:\anaconda3\python.exe scripts\generate_v3_corpus.py --mode plan          # 只打印计划
D:\anaconda3\python.exe scripts\generate_v3_corpus.py --mode probe         # 2 条探针 + 校验
.\.venv_tts\Scripts\python.exe scripts\generate_v3_corpus.py --mode all --workers 6 --retries 3   # TTS 96 条 + 增广 2142 条
.\.venv_tts\Scripts\python.exe scripts\generate_unknown_v3.py --workers 6 --retries 3            # 未知集 72 条
D:\anaconda3\python.exe scripts\build_dataset_v3.py                       # 清单/划分/审计
D:\anaconda3\python.exe -m unittest tests.test_v3_dataset -v
D:\anaconda3\python.exe -m unittest discover -s tests
```

**是否需要联网**：是，仅 `--mode tts` 与 `generate_unknown_v3.py` 需要联网（共 168 条真实 TTS：
54 新表达 baseline + 42 WanLung 缺口源 + 72 未知集；单条约 1.5–5 s，实测批次总耗时 202.6 s + 59.8 s）。
`probe` / `--mode augment` / `build_dataset_v3.py` 完全离线（纯 numpy）。

**预计耗时（本机实测，网络正常时）**：

| 步骤 | 耗时 |
|---|---|
| probe | < 2 s |
| TTS 96 条（并发 6，重试 3） | ≈ 203 s |
| 未知集 72 条 | ≈ 60 s |
| 离线增广 2142 条（6 进程） | ≈ 127 s |
| build（含全量重算 + f0 验证） | ≈ 247 s |
| 测试（22 + 全量 165 项） | ≈ 12 s |
| **合计** | **≈ 11 分钟** |

若网络处于基准窗口那种不稳定状态（失败率 22%–42%），TTS 阶段会因重试明显变长（可能 10–30 分钟）；
生成器幂等可断点续跑——直接重跑同一命令即可，已存在且校验通过的音频会 `skipped_existing`。

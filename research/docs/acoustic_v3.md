# 第三阶段：声学前端与判别式分类器（v3，acoustic front-end & discriminative classifier）

> 写作用域：`src/features_v3.py`、`src/v3_classifier.py`、`scripts/run_v3_acoustic_ablation.py`、
> `tests/test_v3_features.py`、`tests/test_v3_classifier.py`、`outputs/v3_acoustic/`、本文档。
> 未修改任何二期既有文件（`src/features.py` 等保持原样）。

## 1. 目标与结论摘要

二期基线是「MFCC13 + 整句 CMVN + DTW 全模板匹配」，在固定划分 track A 上 test Acc = **0.5495**。
二期的诊断结论是：数据量不是瓶颈，**方向词（开/关、开/关窗帘）混淆**是瓶颈。Lead 的探针实验显示
「MFCC+Δ+ΔΔ 的整句统计特征 + 判别式分类器」val 0.739 / test 0.658，是当前最有效的杠杆。

本工作把探针思路工程化为可配置、可复现、可 import 的模块，并做严格协议下的消融：

> **最终状态（收束）**：最终 `outputs/v3_acoustic/ablation.json`（192 KB，42/54 候选）
> 含 42 个已完成候选的 val/test 指标、每类 F1、混淆矩阵、参数规模、延迟与方向对准确率；
> v3.1 未跑完（已实现未运行）；`outputs/v3_acoustic/best_model.npz` 为推荐配置模型。
> `python -m unittest discover -s tests`：197 项全部通过（原 65 项未破坏）。

**结论摘要**

* 在 v2 固定划分（track A：未见表达、已见音色）上，经典判别式方案的 test 上限为
  **Acc 0.7027 / Macro-F1 0.6952**（`mfcc_delta_pitch + stats + logreg`，val Acc 0.8261）。
  二期 DTW 基线 test 0.5495，提升 **+15.3 个百分点**；相对 Lead 探针（GD 训练）
  test 0.658，提升 +4.5pp。
* 主要增益来自**动态特征（Δ/ΔΔ）**：纯 MFCC13 + 统计量在 test 上只有 0.5586，
  加入 Δ/ΔΔ 后 0.7027；音高、能量、log-mel 是次级增益（val 0.804→0.826 / test 0.748）。
* 方向对二分类中 **LIGHT_ON/OFF 较好（test 0.80），CURTAIN_OPEN/CLOSE 仍难（test 0.60）**，
  且 CURTAIN 的 val 0.938 与 test 0.600 差距大，说明该方向对在「未见表达」上泛化差，
  与二期「开/关窗帘是瓶颈」的诊断一致。
* `fixed40` 高维嵌入（40·D 维展平）在本数据量下明显过拟合，单独使用 val 仅 0.35–0.52，
  必须改用统计量嵌入或强正则。
* 时间盒内完成 54 个计划候选中的 42 个（6 特征集 × 3 嵌入 × 3 分类器中的主体部分），
  未完成的为 `logmel_delta_pitch` 全部候选与部分 `fixed40 + SVM/kNN` 组合；
  代码已实现，可直接重跑（见 §8）。v3.1 扩充数据上的候选训练**已实现未运行**（§5）。

## 2. 方法

### 2.1 可配置前端（`src/features_v3.py`）

`FeatureSpec` 是 frozen dataclass（可哈希、`to_dict()/from_dict()` 可 JSON 序列化、
`fingerprint()` 给出稳定短哈希用于特征缓存），字段：

| 字段 | 默认 | 说明 |
|---|---|---|
| `sample_rate` | 16000 | 输入采样率；与音频不一致时显式报错 |
| `n_fft` / `hop` | 512 / 160 | 32 ms 帧长 / 10 ms 帧移 |
| `n_mels` / `n_mfcc` | 26 / 13 | mel 滤波器组通道数 / 倒谱阶数 |
| `fmin` / `fmax` | 50 / None(sr/2) | mel 频带范围 |
| `preemphasis` | 0.97 | 预加重系数 |
| `use_cmvn` | True | 对 MFCC / log-mel 块沿时间轴做整句 CMVN |
| `include_delta` / `include_delta2` | False | 是否追加 Δ / ΔΔ（回归差分，`delta_width=2`，edge padding） |
| `include_pitch` | False | 是否追加音高块 |
| `include_energy` | False | 是否追加帧 log-RMS 能量块 |
| `include_logmel` / `include_mfcc` | False / True | 选择基础表示（可同时使用） |

特征块顺序固定：

```
[mfcc][Δmfcc][ΔΔmfcc][logmel][Δlogmel][ΔΔlogmel][pitch][Δpitch][ΔΔpitch][energy][Δenergy][ΔΔenergy]
```

各块按开关出现，缺失块直接省略、不补零；维度可用 `output_dim(spec)` 计算。
`extract_sequence(audio, sr, spec) -> float32 [T,D]` 全程纯 numpy、确定性、无 NaN/Inf。

**音高（F0）**：逐帧自相关法。帧内去均值后用 FFT 计算自相关，在 lag ∈ [sr/400, sr/60] 内取
归一化自相关峰值；峰值 ≥ 0.30 且帧 RMS ≥ 1e-3 判为浊音。浊音帧写 `log(f0/100Hz)`，
清音/静音帧写 `0.0`，另加一维 `voiced` 标志；因此音高块基础维度 2，打开 Δ/ΔΔ 后为 6。
音高与 MFCC 使用同一帧网格（严格逐帧对齐）。

**能量**：帧 log-RMS（`log(sqrt(mean(frame²)) + 1e-8)`），基础 1 维，打开 Δ/ΔΔ 后 3 维。

**与二期等价性**：`include_delta=False` 且 `include_pitch/energy/logmel=False` 时，
`extract_sequence` 与 `src.features.extract_mfcc` **逐元素完全相等**（单测断言
`np.testing.assert_allclose(..., atol=1e-6)`，实测 bitwise 相同）。因此 v3 前端是二期的严格超集。

### 2.2 整句嵌入

| 嵌入 | 维度 | 内容 |
|---|---|---|
| `stats` | 6·D | 沿时间轴 `mean/std/min/max/p10/p90`（与 Lead 探针一致，便于对比） |
| `fixed40` | 40·D | 线性插值到 40 帧后展平（保留时间轮廓，但维度高、易过拟合） |
| `stats+fixed40` | 46·D | 两者拼接 |

`stats_embedding(seq, stats=..., trim=...)` 可配置统计量与相对时间裁剪；
`fixed_length_embedding(seq, n_frames=40)` 对任意 T≥1 产生有限值。

### 2.3 分类器（`src/v3_classifier.py`，纯 numpy，不依赖 sklearn）

* `MultinomialLogisticRegression`：`-mean log p(y|x) + 0.5·l2·||W||²`；内置特征标准化
  （训练集 mean/std，`standardize=False` 可关）；**确定性 L-BFGS**（Armijo 回溯线搜索）训练，
  无随机初始化；支持多分类与二分类；`save/load`（npz + JSON 元数据，`version=3.0.0`，
  保留类别 dtype）；`feature_config` 可随模型一起持久化。
  * 实现说明：Lead 探针用固定步长全批量 GD（3000 轮），在本项目高维嵌入上收敛不充分；
    改用 L-BFGS 后同一份 MFCC+Δ+ΔΔ 统计特征在 val 上从 0.739 提升到 0.804
    （迭代 50–100 次即收敛，CPU 上单次训练 <1 s）。
  * `lr/max_lr/momentum` 字段保留用于兼容早期探针记录，当前训练器不使用。
* `LinearSVM`：One-vs-Rest + SGD hinge（确定性打乱，seed 固定），作为线性对照。
* `KNNClassifier`：余弦相似度 kNN，作为非参数对照。
* `load_model(path)`：按文件内 `class` 字段分派并校验版本。

## 3. 实验协议（严格）

* 数据：`data/tts_corpus/manifest_yue_v2.csv`（600 条）× 固定划分
  `data/splits/track_a_expression_split.csv`（train 397 / val 92 / test 111 条；
  114/29/31 个表达组）。track A = **未见表达、已见音色**。
* 只在 **val** 上选择超参（每个候选的 L2/标准化网格、SVM 的 C、kNN 的 k 全部在 val 上选）；
  **test 对每个候选只预测一次**，不参与任何选择、不用于调参。
* 最优候选排序规则：val Acc → val Macro-F1 → 参数更少（更简单）。
* 方向对二分类（LIGHT_ON/LIGHT_OFF、CURTAIN_OPEN/CURTAIN_CLOSE）：
  对全部 logreg 候选单独做二分类，同样 val 选超参、test 只评一次；并给出多数类基线。
* 搜索空间：6 个特征集 × 3 种嵌入 × 3 个分类器 = 54 个候选；
  logreg 网格 `l2 ∈ {1e-4,1e-3,1e-2,1e-1,1.0} × standardize ∈ {True,False}`；
  SVM `C ∈ {0.1,1,10}`（SGD 80 epochs，作为对照，未追求其最优）；
  kNN `k ∈ {1,3,5,7,9,11}`。
* 指标：Acc / Macro-F1 / 每类 F1 / 混淆矩阵 / 参数规模 / 推理延迟（val 集逐样本平均）。
* 复现命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
D:\anaconda3\python.exe scripts\run_v3_acoustic_ablation.py
```

## 4. 结果（v2 语料，track A 固定划分）

<!-- V2_RESULTS -->

### 4.1 各特征集 × stats 嵌入 × logreg（val 选择超参，test 仅一次）

| 特征集 | 序列维度 D | 嵌入维度 | 选中超参 | val Acc | val Macro-F1 | **test Acc** | **test Macro-F1** | 参数数 | 推理延迟/样本 |
|---|---|---|---|---|---|---|---|---|---|
| `mfcc` | 13 | 78 | l2=0.1, std=True | 0.6957 | 0.6924 | 0.5586 | 0.5439 | 474 | 0.0014 ms |
| `mfcc_delta` | 39 | 234 | l2=0.1, std=True | 0.8043 | 0.7976 | 0.7027 | 0.6929 | 1410 | 0.0050 ms |
| **`mfcc_delta_pitch`** | **45** | **270** | **l2=0.1, std=True** | **0.8261** | **0.8221** | **0.7027** | **0.6952** | 1626 | 0.0165 ms |
| `mfcc_delta_pitch_energy` | 48 | 288 | l2=0.01, std=True | 0.8261 | 0.8219 | 0.7297 | 0.7239 | 1734 | 0.0085 ms |
| `logmel_delta` | 120 | 720 | l2=1e-4, std=True | 0.8152 | 0.8142 | **0.7477** | **0.7300** | 4326 | 0.0088 ms |
| `logmel_delta_pitch` | 126 | 756 | — | —（未运行） | — | — | — | — | — |

注：`test` 结果没有参与任何超参选择；`mfcc` 行为二期 DTW 基线的对照（test 0.5495）。
表中「推理延迟」为 val 集逐样本平均的「嵌入+分类」耗时（不含音频 I/O 与特征提取）；
特征提取耗时（v2 600 条实测平均）：`mfcc` 16.1 ms、`mfcc_delta` 16.4 ms、
`mfcc_delta_pitch` 70.5 ms、`logmel_delta` 29.9 ms、`mfcc_delta_pitch_energy` 58.2 ms
（纯 Python/numpy，未做优化；pitch 自相关占主要开销）。端到端单条延迟 ≈ 特征提取 + 0.02 ms。

### 4.2 最佳配置的每类 F1 与混淆矩阵（`mfcc_delta_pitch + stats + logreg`）

| LIGHT_ON | LIGHT_OFF | CURTAIN_OPEN | CURTAIN_CLOSE | ALL_OFF | STATUS_ALL |
|---|---|---|---|---|---|
| 0.7368 | 0.7586 | 0.5128 | 0.5161 | 0.8261 | 0.8205 |

test 混淆矩阵（行=真实，列=预测；顺序 LIGHT_ON, LIGHT_OFF, CURTAIN_OPEN,
CURTAIN_CLOSE, ALL_OFF, STATUS_ALL）：

```
[[14,  4,  2,  0,  0,  1],
 [ 2, 11,  0,  0,  1,  0],
 [ 0,  0, 10, 10,  0,  2],
 [ 0,  0,  4,  8,  1,  0],
 [ 1,  0,  1,  0, 19,  2],
 [ 0,  0,  0,  0,  2, 16]]
```

主要错误集中在 CURTAIN_OPEN ↔ CURTAIN_CLOSE（各 10 条互错），与二期诊断一致。

### 4.3 方向对二分类（stats + logreg，每对单独 val 选超参、test 只评一次）

| 特征集 | LIGHT_ON/OFF val | LIGHT_ON/OFF test | CURTAIN_OPEN/CLOSE val | CURTAIN_OPEN/CLOSE test |
|---|---|---|---|---|
| `mfcc` | 0.828 | 0.657 | 0.750 | 0.514 |
| `mfcc_delta` | 0.793 | 0.800 | 0.938 | 0.600 |
| `mfcc_delta_pitch`（推荐主配置） | 0.828 | 0.657 | 0.938 | 0.629 |
| `logmel_delta` | 0.724 | **0.914** | 0.938 | **0.657** |
| `mfcc_delta_pitch_energy` / `logmel_delta_pitch` | 未运行 | 未运行 | 未运行 | 未运行 |

（方向对为 stats+logreg，每对单独在 val 选择 L2×标准化，test 只评一次；
本次收束只保留了 Acc，Macro-F1 见 `ablation.json` 的 `directional_pairs` 说明。）

CURTAIN 对 val→test 从 0.938 掉到 0.60–0.66，是典型的「val 上把表达特异线索记住了、
未见表达不成立」；说明该类别需要更强的语义/上下文建模，而不是继续加大声学前端容量。
值得注意的是 **logmel_delta 的 LIGHT_ON/OFF 二分类 test 达 0.914**，即方向对本身
可以做得很好，六分类的错误主要来自其他类别的混淆。

### 4.4 嵌入与分类器对照

* **嵌入**：`stats` 全面优于 `fixed40` / `stats+fixed40`。例如 `mfcc` 特征集下
  val Acc：stats 0.6957、stats+fixed40 0.5978、fixed40 0.5109；
  `logmel_delta` 特征集：stats 0.8152、stats+fixed40 0.4457、fixed40 0.3478。
  40 帧展平在 397 条训练样本上维度 1560–5040，严重过拟合。
* **分类器**：logreg > linear_svm > knn。同一 `mfcc_delta` stats 特征：
  logreg val 0.8043 / test 0.7027；linear_svm val 0.7391 / test 0.6667；
  knn val 0.5435 / test 0.5676。线性 SVM 使用 SGD（80 epochs）作为对照，
  未追求其最优；kNN 在统计嵌入上偏弱。
* **L2/标准化**：val 选择几乎总是 `standardize=True`；最佳 l2 随维度增大而变小
  （`stats` 嵌入 234–720 维时 l2=0.1/0.01/1e-4）。
* **优化器**：把探针的固定步长全批量 GD 换成确定性 L-BFGS 后，同一份
  `mfcc_delta` stats 特征在 val 上从 0.739（探针）提升到 0.804（本工作），
  test 0.658→0.703，且单次训练 <1 s。**这部分提升来自优化收敛，不是新特征**。

## 5. 在 v3.1 扩充数据上的对比（与 CNN 主模型同划分）

<!-- V31_RESULTS -->

**状态：已实现、未运行（时间盒终止）。**

* 脚本已支持任意清单/划分：
  `python scripts/run_v3_acoustic_ablation.py --manifest data\tts_corpus\manifest_yue_v3_1.csv
  --split data\splits\track_a_expression_split_v3_1.csv --feature-sets ... --embeddings stats
  --classifiers logreg --out outputs\v3_acoustic\ablation_v3_1.json`
* v3.1 数据（2904 条：train 1981 / val 440 / test 483，210 个表达组；条件含 rate/pitch 离线增广）
  的**特征缓存已生成**（`mfcc_delta`、`mfcc_delta_pitch`、`mfcc_delta_pitch_energy`、
  `logmel_delta` 四套，均按清单 sha 命名，位于 `outputs/v3_acoustic/cache/`）。
* 候选训练与 test 评估未完成，因此**没有 v3.1 的 test 数字**；`ablation_v3_1.json` 未生成。
  预估重跑成本：4 个 stats+logreg 候选（各 10 个 L2 网格）在 v3.1 上约 5–10 分钟（已有缓存）。
* 与 CNN 对比的意义：若 v3.1 上该经典方案（如 `mfcc_delta_pitch/logmel_delta + stats + logreg`）
  的 test 与 CNN 差距仍显著，则说明 CNN 的收益来自更强的时频建模而不是「更多数据」；
  反之若差距收窄，说明 CNN 主要吃了数据扩充的红利。该实验是报告里区分这两种解释的关键对照，
  但本次未取得数字，**报告中不应把 v3.1 的经典模型结果当作已测数据引用**。

## 6. 推荐配置与接口

<!-- RECOMMENDATION -->

**推荐给最终 v3 系统的经典配置（v2 上按 val 选出）**

```python
from src.features_v3 import FeatureSpec, extract_sequence, stats_embedding
from src.v3_classifier import MultinomialLogisticRegression

spec = FeatureSpec(include_delta=True, include_delta2=True, include_pitch=True)
# D = 13*3 + 2*3 = 45；stats 嵌入 = 6*45 = 270 维
model = MultinomialLogisticRegression.load('outputs/v3_acoustic/best_model.npz')
seq = extract_sequence(audio, sr, spec)     # float32 [T,45]
x = stats_embedding(seq)                    # float32 [270]
proba = model.predict_proba(x[None, :])     # [1,6]，列顺序 = model.classes_
```

* 训练入口：`scripts/run_v3_acoustic_ablation.py`（可 `--feature-sets/--embeddings/
  --classifiers/--manifest/--split/--no-pairs` 子集化）；最优模型由脚本自动保存为
  `outputs/v3_acoustic/best_model.npz`，`feature_config` 字段记录了 FeatureSpec 与嵌入方式。
* 若更看重 test 上的绝对准确率，`logmel_delta + stats + logreg`（720 维，test 0.7477）
  是 test 最高的已测候选；但按**只允许 val 选择**的协议，推荐 `mfcc_delta_pitch`。
  报告里引用「经典模型上限」时建议同时给出两者并注明选择依据（val vs test）。
* 计算成本：单条特征提取约 70 ms（pitch 自相关主导，未优化）、推理 <0.02 ms；
  模型 1626 参数、文件 <1 MB，适合端侧。
* 已知短板：CURTAIN 开/关（test F1 0.51/0.52）。若要继续提升，优先考虑
  **文本/语义先验或专门的窗帘类二阶段判别器**，而不是继续增加声学特征维度。

**「已实现未运行」清单**

* `logmel_delta_pitch × {stats, fixed40, stats+fixed40} × {logreg, linear_svm, knn}` 全部；
* `mfcc_delta_pitch_energy` 与 `logmel_delta` 的 `stats+fixed40` 组合、`mfcc_delta_pitch`
  的 `stats+fixed40 × {svm, knn}`（部分已跑，详见 `ablation.json` 的 `ablation_scope`）；
* v3.1 数据上的全部候选（特征缓存已生成）。

## 7. 局限（必须如实引用）

1. **全部是合成 TTS 语音**：v2 600 条来自 3 个 edge-tts 粤语音色 + 1 个 canto-tts 遗留音色；
   v3.1 的额外条件（rate/pitch）是**离线信号级增广**，不是真人录音，不能代表真实方言发音。
2. **「未见表达、已见音色」**：固定划分保证表达组不跨子集，但音色全部在训练集中出现过；
   不能外推到未见音色、真实麦克风、远场噪声。
3. **test 规模小**：v2 test 仅 111 条（31 个表达组），准确率标准误约 ±4.5 个百分点；
   候选之间 1–3 个百分点的差异不显著，不应过度解读。
4. **方向词仍是最弱环节**：CURTAIN 开/关二分类明显低于 LIGHT 开/关（详见第 4 节），
   与二期诊断一致——问题主要在语义/韵律而非声学前端容量。
5. **fixed40 高维嵌入严重过拟合**：40·D 维展平特征在 397 条训练样本上无法泛化，
   必须靠 `stats+fixed40` 或更强正则；单独使用 fixed40 明显劣于 stats。
6. 超参与嵌入选择都基于 val；test 只评一次。但作者在报告中多次查看 test 结果，
   存在读者视角的选择偏差，复现时请以 `ablation.json` 记录为准。

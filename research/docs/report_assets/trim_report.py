# -*- coding: utf-8 -*-
"""收尾：压缩冗余 prose，把正文控制在约 9000–10000 汉字（含表格）。不改数字/结论。"""
from pathlib import Path

P = Path(__file__).resolve().parents[2] / 'docs' / 'v3_report.md'
s = P.read_text(encoding='utf-8')


def section(start, end_marker, new):
    global s
    a = s.find(start)
    b = s.find(end_marker, a + 5)
    if a < 0 or b < 0:
        raise SystemExit(f'MISS {start}')
    s = s[:a] + new + s[b:]


section('### 2.3 预注册与选择纪律', '### 2.4 本报告的可信度约定', '''### 2.3 预注册与选择纪律

本阶段在**跑任何 test 指标之前**把规则写死：

1. 所有超参（E2 的 l2、E4 的 TTA 平移量、E5 的融合权重、拒识阈值）只在 **val** 上选择；
   拒识阈值额外使用未知校准集（calibration 12 文本 / 36 条），test 未知样本只在最终评估用一次；
2. 每个候选项的 test 数字都列出以便审计，但**结论以 val 选择为准**；若出现 test 更高的
   非选中项，照实写出且不据此改选（二期已有先例）；
3. 集成种子 `{42, 7, 2026}`、最大 90 epoch、patience=12 预先写入 `config_v3.json`，
   训练脚本不读 test。

''')

section('### 8.3 数据与伦理', '---\n\n## 九、复现指南', '''### 8.3 数据与伦理

1. **合成语音披露**：全部语料由 edge-tts / canto-tts 合成或对合成音频变换，报告与交付物
   必须披露，不得表述为「粤语使用者实测」。
2. **无真人数据**：未采集真人语音，不涉及声纹/生物特征与隐私授权；代价是结论无法覆盖
   真人口音、噪声、远场。
3. **文本未母语者核验**：v3 新增 18 条、v3.1 新增 18 条表达均为 `RULE_REVIEWED`，
   正式发布前需粤语母语者复核；**不使用声音克隆**，TTS 音色为公开服务默认音色。

''')

section('### 10.1 新增/修改文件清单', '### 10.2 关键文件哈希', '''### 10.1 新增/修改文件清单

本阶段新增模块/脚本/数据/测试/文档的完整清单，分别见 `docs/data_v3.md` §1、
`docs/data_v3_1.md` §4、`docs/demo_v3.md` §2 与 `scripts/run_v3_all.ps1` 的步骤注释。
核心新增：`src/features_v3.py`、`src/v3_classifier.py`、`src/v3_cnn_numpy.py`、
`src/v3_model.py`、`src/dataset_v3.py`、`scripts/train_v3_model.py`、
`scripts/run_v3_experiments.py`、`scripts/evaluate_v3_rejection.py`、
`scripts/report_v3_tables.py`、`scripts/run_v3_acoustic_ablation.py`、
`src/control_server_v3.py`、`scripts/demo_v3.py`、`scripts/run_web_demo.py`、
`config_v3.json`、v3/v3.1 清单·划分·音频·表达库、`tests/test_v3_*.py`
（数据/特征/模型/分类器/实验/控制）、`docs/data_v3.md`、`docs/data_v3_1.md`、
`docs/demo_v3.md`、`docs/v3_report.md`（本报告）。**未修改** `src/` 既有文件、
`config.json`、`README.md`、`docs/v2_report.md`。

''')

section('### 6.4 风险解读（必须与数字一起写）', '---\n\n## 七、工程原型与演示', '''### 6.4 风险解读（必须与数字一起写）

1. **本期拒识是失败结果**：未知误接受 **36.11%**（13/36），Wilson 95%CI [22.5%, 52.4%]；
   在覆盖率 ≥85% 约束下，验证集可达的最低 FAR 也只有约 36%–39%（阈值 0.95 时 FAR 22.2%，
   但覆盖率掉到 83.2%）。「可用性」与「抗误接受」**无法同时满足**目标。
2. **误接受是「高置信度错映射」而非「低置信度猜错」**：最危险未知文本置信度达 0.99，
   「開抽氣扇」被稳定映射到 CURTAIN_OPEN——CNN 学到的是方向词而非对象词，
   与 §5.4 的 correction 失败是同一根因。
3. **与二期不可直接比较**：二期阈值 0.62 时未知误接受 0%（12 条）、误拒识 47.8%；
   三期闭集准确率大幅提高却破坏了低置信度拒识路径。样本量、模型、阈值都不同，
   不能写成「拒识变差了几个百分点」。
4. **0% ≠ 安全，36% 更不安全**：未知 test 仅 36 条，rule-of-three 上界 8.3%；
   实际观测 36% 已远超该上界 → **当前模型不能用于需要拒识未知指令的部署场景**。
5. **换 OOD 分数救不了**：4 种分数 AUC 0.85–0.95，但阈值跨 val→test 全部失效
   （test FAR 16.7%–44.4%），问题在未知数据太少、校准/测试分布不同（§5.8）。
6. **不能外推到真人**：未知样本与已知命令同源（同 3 个 TTS 音色），真人分布外程度未知。

''')

P.write_text(s, encoding='utf-8', newline='')
import re
print('CJK now:', len(re.findall(r'[\u4e00-\u9fff]', s)))

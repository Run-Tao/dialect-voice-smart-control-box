# 方言智控盒 v3 实验结果表（自动生成）



运行：`main_v3_1`；生成时间：2026-10-21T20:04:05



## 1. 数据规模与条件

| 数据版本 | 总样本 | train | val | test | 表达组 |
|---|---|---|---|---|---|
| v2 | 600 | - | - | - | 174 |
| v3 | 2796 | 1873 | 440 | 483 | 192 |
| v3.1 | 2904 | 1981 | 440 | 483 | 210 |

**v3 条件分布**

| 条件 | 条数 |
|---|---|
| baseline | 54 |
| legacy | 120 |
| pitch_high | 522 |
| pitch_low | 522 |
| rate_fast_18 | 522 |
| rate_slow | 90 |
| rate_slow_18 | 576 |
| text_morphology | 390 |

**v3.1 增量**：新增 108 行全部 train；correction 组 train 24 个、test 6 个；test 总行数 483（未改动）。

## 2. 模型对比（同一固定 test）

| 模型 | Test Acc（full 483） | Test Macro-F1 | Test Acc（matched v2 111） | Val Acc | n |
|---|---|---|---|---|---|
| E1 v2 基线 MFCC+DTW | 0.5901 | 0.5921 | 0.5495 | - | 483 |
| E2 经典判别式 logreg | 0.6749 | 0.6706 | 0.7027 | - | 483 |
| E3 CNN 集成（3 seeds） | 0.9048 | 0.9022 | 0.8829 | 0.9455 | 483 |
| E4 CNN + TTA（val 选） | 0.9006 | 0.8985 | - | - | 483 |
| E5 CNN+经典 融合（val 选） | 0.9151 | 0.9126 | 0.8919 | - | 483 |

**模型体积与延迟**：体积 1.351 MB，推理 16.03 ms/条（仅 CNN 前向，不含特征提取），种子数 3。


## 3. 统计检验（配对 McNemar + 按表达组自助法）

| 比较 | McNemar p | ΔAcc(B−A) | ΔAcc 95%CI | 判定 |
|---|---|---|---|---|
| E1_vs_E3_full | 0.0000 | +0.3180 | [+0.1780, +0.4581] | 差异显著（McNemar p<0.05） |
| E1_vs_E3_matched_v2 | 0.0000 | +0.3378 | [+0.1786, +0.5041] | 差异显著（McNemar p<0.05） |
| E2_vs_E3_full | 0.0000 | +0.2304 | [+0.1419, +0.3271] | 差异显著（McNemar p<0.05） |
| E2_vs_E3_matched_v2 | 0.0002 | +0.1813 | [+0.0866, +0.2830] | 差异显著（McNemar p<0.05） |

## 4. 条件与语域分层（E5_fusion）

**按声学条件**

| 条件 | n | Accuracy |
|---|---|---|
| legacy | 24 | 79.17% |
| pitch_high | 93 | 91.40% |
| pitch_low | 93 | 90.32% |
| rate_fast_18 | 93 | 93.55% |
| rate_slow | 18 | 83.33% |
| rate_slow_18 | 93 | 93.55% |
| text_morphology | 69 | 94.20% |

**按语域**

| 语域 | n | Accuracy |
|---|---|---|
| colloquial | 93 | 100.00% |
| correction | 98 | 64.29% |
| polite | 101 | 94.06% |
| question | 90 | 100.00% |
| short_direct | 101 | 100.00% |

**按音色**

| 音色 | n | Accuracy |
|---|---|---|
| canto-tts-default | 6 | 66.67% |
| edge-zh-HK-HiuGaaiNeural | 161 | 91.93% |
| edge-zh-HK-HiuMaanNeural | 161 | 92.55% |
| edge-zh-HK-WanLungNeural | 155 | 90.97% |


## 5. 每类 Precision/Recall/F1（E5_fusion）

| 意图 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LIGHT_ON | 100.00% | 96.77% | 98.36% | 93 |
| LIGHT_OFF | 90.24% | 100.00% | 94.87% | 74 |
| CURTAIN_OPEN | 82.67% | 75.61% | 78.98% | 82 |
| CURTAIN_CLOSE | 75.00% | 82.19% | 78.43% | 73 |
| ALL_OFF | 100.00% | 93.98% | 96.89% | 83 |
| STATUS_ALL | 100.00% | 100.00% | 100.00% | 78 |

## 6. 错误分析（E5_fusion）

```text
confusion matrix（行=真实，列=预测；类顺序 LIGHT_ON, LIGHT_OFF, CURTAIN_OPEN, CURTAIN_CLOSE, ALL_OFF, STATUS_ALL）
   90   3   0   0   0   0
    0  74   0   0   0   0
    0   0  62  20   0   0
    0   0  13  60   0   0
    0   5   0   0  78   0
    0   0   0   0   0  78
```

**主要错误对**

| 真实 → 预测 | 次数 |
|---|---|
| CURTAIN_OPEN → CURTAIN_CLOSE | 20 |
| CURTAIN_CLOSE → CURTAIN_OPEN | 13 |
| ALL_OFF → LIGHT_OFF | 5 |
| LIGHT_ON → LIGHT_OFF | 3 |

**错误按语域**：correction=35，polite=6

## 7. 拒识评估（阈值仅验证集 + 未知校准集选择）

选中阈值 **0.93**；验证集覆盖率 85.91%，验证集未知误接受率 36.11%。

| 指标 | 值 |
|---|---|
| 已知命令覆盖率 | 89.03% |
| 已知命令误拒识率 | 10.97% |
| 被接受样本准确率 | 94.42% |
| 无阈值时准确率 | 90.48% |
| 未知命令误接受率（test） | 36.11% |
| 未知误接受 95% Wilson CI | [0.225, 0.524] |
| 硬负例误接受率 | 33.33% |
| 未知测试样本数 | 36 |

**按硬负例类型（未知 test 36 条）**

| 类型 | n | 被接受数 |
|---|---|---|
| device_onoff_exhaust_fan | 6 | 5 |
| device_onoff_fan | 3 | 1 |
| device_onoff_fridge | 6 | 0 |
| device_onoff_music | 6 | 2 |
| device_onoff_tv | 3 | 0 |
| device_onoff_water_heater | 3 | 0 |
| light_brightness | 6 | 3 |
| volume_adjust | 3 | 2 |

**最高置信度的未知文本（最危险）**

| 文本 | n | 被接受 | 最高置信度 | 主要预测 |
|---|---|---|---|---|
| 開抽氣扇 | 3 | 3 | 0.9982 | CURTAIN_OPEN |
| 開音樂 | 3 | 2 | 0.9971 | LIGHT_ON |
| 關抽氣扇 | 3 | 2 | 0.9969 | CURTAIN_CLOSE |
| 調光啲燈 | 3 | 1 | 0.9534 | LIGHT_ON |
| 較大聲啲 | 3 | 2 | 0.9446 | LIGHT_OFF |
| 關風扇 | 3 | 1 | 0.9382 | CURTAIN_CLOSE |
| 調暗啲燈 | 3 | 2 | 0.9347 | LIGHT_ON |
| 關雪櫃 | 3 | 0 | 0.8884 | ALL_OFF |

限制：未知测试集仅 36 条 / 12 个文本，FAR 的 95% 置信上界约 8.3%（rule of three），观测 0% 不等于可靠安全。；未知样本为合成 TTS（3 个 edge 音色），部分与已知命令共享粤语词汇，不能外推到真人语音或任意未登记命令。

## 8. 训练协议与验证集结果

特征：logmel+Δ，n_frames=None；数据哈希：manifest eb9b60140047…，split 97a9eef8041c…。

| seed | Val Acc | best epoch | epochs | 训练秒 |
|---|---|---|---|---|
| 42 | 0.9409 | 38 | 51 | 15905.5 |
| 7 | 0.9500 | 20 | 33 | 88.6 |
| 2026 | 0.9500 | 13 | 26 | 334.1 |

集成验证集准确率 **94.55%**（Macro-F1 0.9444）；温度 T=0.500；总训练 16352.4s。

# DialectVoice ESP32-S3 固件

把粤语家居命令识别搬到 ESP32-S3 上跑：**端侧特征 + int8 推理**，不联网、不依赖云端 ASR。

本目录只负责固件；麦克风选型、PCB、外壳由硬件负责人提供。

## 现状

| 项 | 状态 |
|---|---|
| 特征前端（C） | 已完成，与 numpy 逐样本对拍通过（最大绝对误差 2.8e-4） |
| int8 推理（C） | 已完成，三种子 logits 平均；**483 条测试集与 numpy 决策 483/483 一致，准确率 0.8820 与 float 版相同** |
| 串口协议 | 已完成（`PING` / `INFO` / `RUNRAW`） |
| I2S 麦克风 | **未接**，`MIC` 命令返回 `mic_not_implemented` |
| 硬件实测 | **未做**（本机无板；耗时是主机 gcc 口径） |

模型：`v3_cnn_ensemble_vad` 三种子，6 类，int8 权重 339 KB，输入 80×256。

## 目录

```
firmware/
├── CMakeLists.txt          ESP-IDF 顶层
├── sdkconfig.defaults      S3 / 16MB flash / 八线 PSRAM / 240MHz
├── partitions.csv          app 2 MB
├── main/
│   ├── app_main.c          串口协议入口
│   ├── dv_core.h           对外接口 + DV_BIG（PSRAM 标记）
│   ├── dv_features.c       log-mel40 + Δ（复刻 features_v3.py）
│   ├── dv_infer.c          int8 卷积 + 池化 + FC
│   └── model/              由 tools/export_c_model.py 生成，勿手改
├── tools/
│   ├── export_c_model.py   npz -> C 头文件（含 int8 量化与激活校准）
│   ├── dump_reference.py   生成 numpy 参考数据
│   ├── host_verify.c       主机逐项对拍
│   └── serial_client.py    PC 侧串口客户端
└── README.md
```

## 编译与烧录

需要 **ESP-IDF v5.4.x**（S3 支持完整）。

```powershell
# 1. 激活 IDF 环境（按自己的安装路径）
C:\esp\esp-idf\export.bat

# 2. 编译
cd D:\Run_Tao\DialectVoice_Engineering\firmware
idf.py set-target esp32s3
idf.py build

# 3. 烧录并看日志（-p 换成实际串口）
idf.py -p COM5 flash monitor
```

产物：`build/dialectvoice.bin`（约 700 KB），分区表 app 给 2 MB。

### 模组版本必须对上

`sdkconfig.defaults` 按 **N16R8**（16 MB flash + 8 MB 八线 PSRAM）配置：

- N16R2 / N8R2（四线 PSRAM）：把 `CONFIG_SPIRAM_MODE_OCT=y` 改成 `CONFIG_SPIRAM_MODE_QUAD=y`
- 无 PSRAM 的模组：先清空 `dv_core.h` 里的 `DV_BIG`，并把 `DV_MAX_T` 降到 256，否则启动即崩（特征/推理工作区约 350 KB，内部 SRAM 放不下）

## 烧完怎么验证（不用麦克风）

```powershell
pip install pyserial numpy
python firmware\tools\serial_client.py --port COM5 --info
python firmware\tools\serial_client.py --port COM5 --wav 某条粤语.wav
```

期望：

```
INFO -> {"model":"v3_cnn_ensemble_vad(int8)","seeds":3,"classes":6,"frames":256,"psram":8388608,...}
[1] LIGHT_ON conf=0.9946 dev=?? ms wall=?? ms T=96
```

`dev` 是设备端特征+推理耗时，`wall` 含串口传输，别拿 wall 当推理延迟。

WAV 必须是 **16 kHz 单声道 PCM16**。`RUNRAW` 的二进制帧是 `int32 小端样本数 + float32 小端 PCM`。

## 主机对拍（无硬件验证数值正确性）

```powershell
# 生成参考数据（任选几条）
python firmware\tools\dump_reference.py --wav <a.wav> --out firmware\tools\ref\light_on

# 用 gcc 编译同一份 C 代码
cd firmware
gcc -O2 -std=c99 -w -I main main/dv_features.c main/dv_infer.c tools/host_verify.c -lm -o tools/host_verify.exe

# 逐项比较
tools\host_verify.exe tools\ref\light_on
```

判定线：特征 `max|abs| < 1e-3`、logits `< 5e-2`、argmax 一致。三条样本实测：

| 样本 | 特征 max\|abs\| | logits max\|abs\| | argmax |
|---|---:|---:|---|
| LIGHT_ON | 2.02e-4 | 3.26e-2 | 一致 |
| ALL_OFF | 7.78e-5 | 1.29e-2 | 一致 |
| CURTAIN_CLOSE | 2.85e-4 | 3.35e-2 | 一致 |

logits 的 1e-2 量级差异来自 int8 量化本身（不是实现错误）：float→w8a8 在 483 条上 argmax 只翻 1 条，见 `docs/开源智能音箱借用评估_2026-11-06.md` §8.8。

## 模型更新后必须重跑

模型一换（重训、重标定、换语料），`model/*.h` 全部作废：

```powershell
python firmware\tools\export_c_model.py --calib 300     # 重新校准激活 scale
python firmware\tools\dump_reference.py --wav <样本> --out firmware\tools\ref\x
tools\host_verify.exe tools\ref\x                        # 必须 PASS
idf.py build
```

`export_c_model.py` 会把激活 max 缓存到 `tools/.act_scale_cache.json`；换模型后要删掉缓存或改 `--calib`。

## 性能预期（未在硬件实测）

单种子约 1500 万 MAC，三种子约 4500 万。纯 C 标量（没有用 ESP-DL 的 PIE 汇编）在 240 MHz 上估计：

- 特征前端：30–60 ms
- 推理（3 种子）：400–800 ms
- 合计：**约 0.5–0.9 s**

这**不满足**项目 300 ms 的动作延迟指标。想达标有两条路：

1. `DV_ACTIVE_SEEDS=1`（改 `dv_core.h` 或编译宏）：快 3 倍，但丢掉集成分歧信号，拒识阈值要重标；
2. 接 ESP-DL：用 ESP-PPQ 导出 `.espdl`，走 PIE 的 int8 向量指令，Conv 可快 20 倍以上。代价是多了 ONNX/量化工具链，且 esp-ppq 对新版 Python 兼容性需先验证。

先用串口链路把"能跑、跑得对"钉死，再决定走哪条。

## 已知限制

- **没有硬件实测**：以上数字是主机 gcc 的结果，S3 上的真实耗时要烧板后测。
- **PSRAM 带宽**：激活缓冲在 PSRAM，随机访问比 SRAM 慢，实测可能比估计更慢。
- **无 VAD**：串口模式假定输入已经是单条命令。接麦克风后要补能量 VAD + 端点检测，否则整段静音也会进网络。
- **无拒识**：固件只输出 argmax 与置信度，`maxprob + kNN-OOD` 双信号门限还没搬过来。PC 侧 `runtime/selective.py` 的参照池 250 KB 需要一起导出。

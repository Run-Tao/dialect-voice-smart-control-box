# DialectVoice Engineering

方言语音智能家居转换装置（本科工程导论项目）。

用户说粤语家居命令，端侧完成 VAD 端点检测、特征提取、命令识别与置信度拒识，
再由扬声器播放预录的普通话指令去驱动智能音箱；另备串口 ESP32-S3 与继电器直控两条备用通道。

## 技术路线

```
方言语音 → VAD 端点检测 → log-mel40+Δ（整句 CMVN，插值到 256 帧）
        → 1D-CNN 三种子集成（纯 numpy，1.35 MB）
        → 温度标定置信度 → 阈值拒识
        → 语义层映射（命令标签 → 普通话指令 → 预录音频）
        → 扬声器播放 → 智能音箱执行
```

命令集固定 6 条，与训练语料一致：
`LIGHT_ON` / `LIGHT_OFF` / `CURTAIN_OPEN` / `CURTAIN_CLOSE` / `ALL_OFF` / `STATUS_ALL`。

> 早期方案「方言特征 → 普通话特征」的回归映射已被自有实验否决，详见
> [`docs/技术路线修订说明.md`](docs/技术路线修订说明.md)。转换发生在语义层，不在特征层。

## 目录

| 路径 | 作用 |
|---|---|
| `src/vad.py` | 能量 VAD：自适应噪声底、静音挂起、前后裁剪 |
| `src/realtime.py` | 部署管线：VAD → 特征 → 分类 → 拒识 → 执行 → 播报 → 日志 |
| `src/commands.py` | 6 条命令定义、普通话文本、设备状态机 |
| `src/control.py` | 执行层：日志 / 模拟 ESP32 / 串口 / 继电器 |
| `src/speaker_out.py` | 普通话预录音频播报 |
| `src/features.py` `src/recognition.py` | 二期 MFCC + DTW 基线，仅用于对比实验 |
| `runtime/` | 自包含推理运行时（由 `scripts/sync_runtime.py` 从 Research 同步，可独立部署） |
| `runtime/models/` | 部署模型 `v3_cnn_ensemble_vad.npz` 与阈值 `v3_thresholds_vad.json` |
| `data/audio/mandarin/` | 6 条普通话指令预录音频（edge-tts 合成） |
| `docs/` | 开题报告（Beamer/PDF/PPTX）、项目建议书、修订说明、评估结果 |

## 常用命令

```bash
# 同步自包含运行时与模型（带 sha256 校验，--check 只校验不写入）
python scripts/sync_runtime.py
python scripts/sync_runtime.py --check

# 麦克风实时识别
python scripts/run_monitor.py --executor mock_esp32

# 离线回放指定音频
python scripts/run_replay.py path/to/audio.wav

# 部署管线评估（已知命令 + 未知指令误接受率）
python scripts/evaluate_pipeline.py \
  --manifest "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" \
  --subset test \
  --unknown "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/unknown_v3_vad"

# 单元测试
python -m pytest tests/ -q
```

## 训练与标定（依赖 DialectVoice_Research）

```bash
# 用部署同款 VAD 重生成训练语料，保证训练输入与部署输入同源
python scripts/make_trimmed_corpus.py

# 在 Research 侧重训练
python scripts/train_v3_model.py \
  --manifest data/tts_corpus/manifest_yue_v3_1_vad.csv \
  --split data/splits/track_a_expression_split_v3_1.csv \
  --out data/models/v3_cnn_ensemble_vad.npz

# 在验证集 + 未知校准集上选拒识阈值（测试集不参与）
python scripts/calibrate_threshold.py
```

## 尚未解决

合成 TTS 语料与真人粤语之间存在域差距，未见说话人、真实噪声、远场、真实端到端延迟
四项指标均无真人数据支撑，不能作为性能承诺。

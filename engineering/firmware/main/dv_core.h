/* DialectVoice 端侧推理内核（ESP32-S3 / 主机通用，纯 C99，无外部依赖）。
 *
 * 与 Python 侧严格对齐：
 *   runtime/features_v3.py  log-mel40 + Δ，整句 CMVN，线性插值到 256 帧
 *   runtime/v3_cnn_numpy.py Conv1d×3 + global mean/max + FC，3 种子 logits 平均
 *
 * 量化口径见 docs/开源智能音箱借用评估_2026-11-06.md §8.8：
 * 权重 per-output-channel 对称 int8，激活 per-tensor 对称 int8，int32 累加。
 *
 * 线程安全：内部使用静态缓冲区，**不可重入**（单线程固件够用）。
 */
#ifndef DV_CORE_H
#define DV_CORE_H

#include <stdint.h>

#include "model/dv_model_config.h"

/* 大缓冲区放在 PSRAM：N16R8 有 8 MB，内部 SRAM 只有 512 KB，
 * 特征+推理的静态工作区约 350 KB，全放 SRAM 会把 heap 挤空。
 * 主机对拍时没有 ESP_PLATFORM，退化为普通静态数组。 */
#ifdef ESP_PLATFORM
#include "esp_attr.h"
#define DV_BIG EXT_RAM_BSS_ATTR
#else
#define DV_BIG
#endif

#define DV_OK           0
#define DV_ERR_AUDIO   -1
#define DV_ERR_FRAMES  -2   /* 音频过长，超过 DV_MAX_T */
#define DV_ERR_NOMEM   -3

/* 特征帧数上限：512 帧 ≈ 5.1 s @ hop 160 / 16 kHz。
 * 一条家居命令通常 1–3 s，留足余量；不够时调大并同步 seq 缓冲区。 */
#define DV_MAX_T 512

typedef struct {
    int   n_frames;                  /* 实际特征帧数 T */
    int   command;                   /* argmax 类别下标 */
    float confidence;                /* softmax(logits / T) 的最大值 */
    float logits[DV_N_CLASSES];      /* 种子平均后的 logits */
} dv_result_t;

/* 提取帧级特征：[T, 80]，行优先（seq[t*80 + d]）。
 * seq_out 容量必须 >= DV_MAX_T * 80 个 float。
 * 返回 DV_OK；*out_T 写入实际帧数。 */
int dv_extract_features(const float *pcm, int n_samples,
                        float *seq_out, int *out_T);

/* 插值到 256 帧 + int8 量化 + 三种子推理。
 * seq 为 dv_extract_features 的输出，T 为其帧数。 */
int dv_predict(const float *seq, int T, dv_result_t *out);

/* 便捷入口：PCM -> 结果。 */
int dv_run(const float *pcm, int n_samples, dv_result_t *out);

const char *dv_class_name(int idx);

#endif /* DV_CORE_H */

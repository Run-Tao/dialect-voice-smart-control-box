/* int8 推理：Conv1d×3（BN 已折叠）+ ReLU + MaxPool + global mean/max + FC。
 *
 * 与 runtime/v3_cnn_numpy.py 的对应关系：
 *   Conv1d(80->64,k5,pad2) -> pool2
 *   Conv1d(64->128,k5,pad2) -> pool2
 *   Conv1d(128->128,k3,pad1)
 *   concat(mean(T), max(T)) -> FC(256->6)
 *
 * 量化：权重 per-output-channel 对称 int8；激活 per-tensor 对称 int8；
 * 卷积 int32 累加后反量化，再 ReLU、再量化。MaxPool 直接作用在 int8 上
 * （round 单调，先 pool 后量化与先量化后 pool 的最大值一致）。
 */
#include "dv_core.h"
#include "model/dv_model_data.h"

#include <math.h>
#include <string.h>

#ifndef DV_ACTIVE_SEEDS
#define DV_ACTIVE_SEEDS DV_N_SEEDS
#endif

#define DV_C0 64
#define DV_C1 128
#define DV_C2 128

typedef struct {
    const int8_t *w0, *w1, *w2, *wf;
    const float  *s0, *s1, *s2, *sf;
    const float  *b0, *b1, *b2, *bf;
} dv_seed_t;

static const dv_seed_t DV_SEEDS[DV_N_SEEDS] = {
    {DV_S0_CONV0_W, DV_S0_CONV1_W, DV_S0_CONV2_W, DV_S0_FC_W,
     DV_S0_CONV0_SCALE, DV_S0_CONV1_SCALE, DV_S0_CONV2_SCALE, DV_S0_FC_SCALE,
     DV_S0_CONV0_BIAS, DV_S0_CONV1_BIAS, DV_S0_CONV2_BIAS, DV_S0_FC_BIAS},
    {DV_S1_CONV0_W, DV_S1_CONV1_W, DV_S1_CONV2_W, DV_S1_FC_W,
     DV_S1_CONV0_SCALE, DV_S1_CONV1_SCALE, DV_S1_CONV2_SCALE, DV_S1_FC_SCALE,
     DV_S1_CONV0_BIAS, DV_S1_CONV1_BIAS, DV_S1_CONV2_BIAS, DV_S1_FC_BIAS},
    {DV_S2_CONV0_W, DV_S2_CONV1_W, DV_S2_CONV2_W, DV_S2_FC_W,
     DV_S2_CONV0_SCALE, DV_S2_CONV1_SCALE, DV_S2_CONV2_SCALE, DV_S2_FC_SCALE,
     DV_S2_CONV0_BIAS, DV_S2_CONV1_BIAS, DV_S2_CONV2_BIAS, DV_S2_FC_BIAS},
};

/* 工作区：约 101 KB。若 SRAM 紧张，可整体移到 PSRAM（N16R8）。 */
static DV_BIG int8_t g_in[DV_IN_DIM * DV_N_FRAMES];   /* 20 KB */
static DV_BIG int8_t g_a[DV_C1 * DV_N_FRAMES];        /* 32 KB，容纳 64x256 */
static DV_BIG int8_t g_b[DV_C1 * DV_C1];              /* 16 KB，容纳 128x128 */
static DV_BIG float  g_c[DV_C2 * 64];           /* 32 KB，conv2 的 float 输出 */
static float  g_pool[256];

static inline int8_t quant_clamp(float v, float inv_scale)
{
    /* 传 1/scale 而不是 scale：卷积内层每个输出都要量化，除法要换成乘法 */
    const float q = v * inv_scale;
    int qi = (int)(q + (q >= 0.0f ? 0.5f : -0.5f));
    if (qi > 127) {
        return 127;
    }
    if (qi < -127) {
        return -127;
    }
    return (int8_t)qi;
}

/* 线性插值到 DV_N_FRAMES，并量化成 [D][T] 布局（卷积按通道优先）。 */
static void resample_quant(const float *seq, int T, int8_t *out)
{
    const float last = (float)(DV_N_FRAMES - 1);
    const float inv_in = 1.0f / DV_ACT_SCALE[0];
    for (int i = 0; i < DV_N_FRAMES; i++) {
        float pos = 0.0f;
        if (T > 1) {
            pos = (float)i * (float)(T - 1) / last;
        }
        int i0 = (int)pos;
        if (i0 > T - 1) {
            i0 = T - 1;
        }
        int i1 = (i0 + 1 < T) ? i0 + 1 : T - 1;
        const float frac = pos - (float)i0;
        const float *r0 = seq + (size_t)i0 * DV_IN_DIM;
        const float *r1 = seq + (size_t)i1 * DV_IN_DIM;
        for (int d = 0; d < DV_IN_DIM; d++) {
            const float v = r0[d] * (1.0f - frac) + r1[d] * frac;
            out[(size_t)d * DV_N_FRAMES + i] = quant_clamp(v, inv_in);
        }
    }
}

/* 量化卷积：输出 int8，尺寸不变（same padding）。
 * 边界单独处理，内部帧走无分支快路径——这是整个推理的热点。 */
static void conv_i8(const int8_t *x, int C, int T,
                    const int8_t *w, const float *wscale, const float *bias,
                    int O, int K, float in_scale, float out_scale, int8_t *y)
{
    const int pad = K / 2;
    const float inv_out = 1.0f / out_scale;
    for (int o = 0; o < O; o++) {
        const int8_t *wo = w + (size_t)o * C * K;
        const float gain = in_scale * wscale[o];
        const float b = bias[o];
        int8_t *yo = y + (size_t)o * T;
        for (int t = 0; t < T; t++) {
            int32_t acc = 0;
            const int lo = t - pad;
            if (lo >= 0 && lo + K <= T) {
                for (int c = 0; c < C; c++) {
                    const int8_t *xc = x + (size_t)c * T + lo;
                    const int8_t *wc = wo + (size_t)c * K;
                    for (int k = 0; k < K; k++) {
                        acc += (int32_t)xc[k] * (int32_t)wc[k];
                    }
                }
            } else {
                for (int c = 0; c < C; c++) {
                    const int8_t *xc = x + (size_t)c * T;
                    const int8_t *wc = wo + (size_t)c * K;
                    for (int k = 0; k < K; k++) {
                        const int idx = lo + k;
                        if (idx >= 0 && idx < T) {
                            acc += (int32_t)xc[idx] * (int32_t)wc[k];
                        }
                    }
                }
            }
            float v = (float)acc * gain + b;
            if (v < 0.0f) {
                v = 0.0f;
            }
            yo[t] = quant_clamp(v, inv_out);
        }
    }
}

/* 最后一层卷积输出 float（后面是 mean/max，不能量化）。 */
static void conv_f32(const int8_t *x, int C, int T,
                     const int8_t *w, const float *wscale, const float *bias,
                     int O, int K, float in_scale, float *y)
{
    const int pad = K / 2;
    for (int o = 0; o < O; o++) {
        const int8_t *wo = w + (size_t)o * C * K;
        const float gain = in_scale * wscale[o];
        const float b = bias[o];
        float *yo = y + (size_t)o * T;
        for (int t = 0; t < T; t++) {
            int32_t acc = 0;
            const int lo = t - pad;
            if (lo >= 0 && lo + K <= T) {
                for (int c = 0; c < C; c++) {
                    const int8_t *xc = x + (size_t)c * T + lo;
                    const int8_t *wc = wo + (size_t)c * K;
                    for (int k = 0; k < K; k++) {
                        acc += (int32_t)xc[k] * (int32_t)wc[k];
                    }
                }
            } else {
                for (int c = 0; c < C; c++) {
                    const int8_t *xc = x + (size_t)c * T;
                    const int8_t *wc = wo + (size_t)c * K;
                    for (int k = 0; k < K; k++) {
                        const int idx = lo + k;
                        if (idx >= 0 && idx < T) {
                            acc += (int32_t)xc[idx] * (int32_t)wc[k];
                        }
                    }
                }
            }
            float v = (float)acc * gain + b;
            yo[t] = v > 0.0f ? v : 0.0f;
        }
    }
}

static void maxpool2_i8(const int8_t *x, int C, int T, int8_t *y, int *out_T)
{
    const int t2 = T / 2;
    for (int c = 0; c < C; c++) {
        const int8_t *xc = x + (size_t)c * T;
        int8_t *yc = y + (size_t)c * t2;
        for (int t = 0; t < t2; t++) {
            const int8_t a = xc[2 * t];
            const int8_t b = xc[2 * t + 1];
            yc[t] = a > b ? a : b;
        }
    }
    *out_T = t2;
}

static void run_seed(const dv_seed_t *sd, float *logits)
{
    int t0 = DV_N_FRAMES;

    /* g_a / g_b 交替做输入输出：原地 pool 会覆盖前一通道的结果，禁止。 */
    conv_i8(g_in, DV_IN_DIM, t0, sd->w0, sd->s0, sd->b0,
            DV_C0, 5, DV_ACT_SCALE[0], DV_ACT_SCALE[1], g_a);
    int t1;
    maxpool2_i8(g_a, DV_C0, t0, g_b, &t1);

    conv_i8(g_b, DV_C0, t1, sd->w1, sd->s1, sd->b1,
            DV_C1, 5, DV_ACT_SCALE[1], DV_ACT_SCALE[2], g_a);
    int t2;
    maxpool2_i8(g_a, DV_C1, t1, g_b, &t2);

    conv_f32(g_b, DV_C1, t2, sd->w2, sd->s2, sd->b2,
             DV_C2, 3, DV_ACT_SCALE[2], g_c);

    for (int c = 0; c < DV_C2; c++) {
        const float *xc = g_c + (size_t)c * t2;
        float sum = 0.0f;
        float mx = xc[0];
        for (int t = 0; t < t2; t++) {
            sum += xc[t];
            if (xc[t] > mx) {
                mx = xc[t];
            }
        }
        g_pool[c] = sum / (float)t2;
        g_pool[DV_C2 + c] = mx;
    }

    const float inv_pool = 1.0f / DV_ACT_SCALE[3];
    for (int o = 0; o < DV_N_CLASSES; o++) {
        int32_t acc = 0;
        const int8_t *wo = sd->wf + (size_t)o * 256;
        for (int i = 0; i < 256; i++) {
            const int8_t q = quant_clamp(g_pool[i], inv_pool);
            acc += (int32_t)q * (int32_t)wo[i];
        }
        logits[o] = (float)acc * (DV_ACT_SCALE[3] * sd->sf[o]) + sd->bf[o];
    }
}

int dv_predict(const float *seq, int T, dv_result_t *out)
{
    if (seq == NULL || out == NULL || T <= 0) {
        return DV_ERR_AUDIO;
    }

    resample_quant(seq, T, g_in);

    float acc[DV_N_CLASSES] = {0};
    for (int si = 0; si < DV_ACTIVE_SEEDS; si++) {
        float logits[DV_N_CLASSES];
        run_seed(&DV_SEEDS[si], logits);
        for (int c = 0; c < DV_N_CLASSES; c++) {
            acc[c] += logits[c];
        }
    }

    const float inv_seeds = 1.0f / (float)DV_ACTIVE_SEEDS;
    float mx = -INFINITY;
    for (int c = 0; c < DV_N_CLASSES; c++) {
        out->logits[c] = acc[c] * inv_seeds;
        if (out->logits[c] > mx) {
            mx = out->logits[c];
        }
    }

    /* softmax(logits / temperature) */
    float sum = 0.0f;
    float probs[DV_N_CLASSES];
    for (int c = 0; c < DV_N_CLASSES; c++) {
        probs[c] = expf((out->logits[c] - mx) / DV_TEMPERATURE);
        sum += probs[c];
    }
    int best = 0;
    float best_p = 0.0f;
    for (int c = 0; c < DV_N_CLASSES; c++) {
        const float p = probs[c] / sum;
        if (p > best_p) {
            best_p = p;
            best = c;
        }
    }
    out->command = best;
    out->confidence = best_p;
    return DV_OK;
}

int dv_run(const float *pcm, int n_samples, dv_result_t *out)
{
    static DV_BIG float seq[DV_MAX_T * DV_IN_DIM];   /* 164 KB @ DV_MAX_T=512 */
    int T = 0;
    const int rc = dv_extract_features(pcm, n_samples, seq, &T);
    if (rc != DV_OK) {
        return rc;
    }
    const int rc2 = dv_predict(seq, T, out);
    if (rc2 == DV_OK) {
        out->n_frames = T;
    }
    return rc2;
}

const char *dv_class_name(int idx)
{
    if (idx < 0 || idx >= DV_N_CLASSES) {
        return "UNKNOWN";
    }
    return DV_CLASSES[idx];
}

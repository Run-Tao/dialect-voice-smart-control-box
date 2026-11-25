/* 特征前端：严格复刻 runtime/features_v3.py 的 log-mel40 + Δ 路径。
 *
 * 数值约定（与 numpy 实现逐项对应）：
 *   pre-emphasis   x[0] 保持，x[i] -= 0.97 * x[i-1]（补零发生在之后）
 *   分帧           512 点 / 160 跳，n_frames = 1 + (n - 512) / 160
 *   窗             numpy.hamming(512) = 0.54 - 0.46 cos(2πn/511)，已导出为表
 *   功率谱         |rfft512|²，取 0..256 共 257 个 bin
 *   mel            log(max(power · fbᵀ, 1e-10))
 *   CMVN           沿时间轴 (x - mean) / (std + 1e-8)，std 用总体标准差
 *   Δ              width=2，edge padding，den = 2·(1²+2²) = 10
 */
#include "dv_core.h"

#include <math.h>
#include <string.h>

#define DV_NFFT   512
#define DV_HOP    160
#define DV_NMEL   40
#define DV_NBINS  257
#define DV_PI     3.14159265358979323846f

/* 静态工作区：单线程固件，避免在 8 KB 任务栈上开大数组 */
static DV_BIG float g_mel[DV_MAX_T][DV_NMEL];   /* 82 KB @ DV_MAX_T=512 */
static float g_re[DV_NFFT];
static float g_im[DV_NFFT];
static float g_frame[DV_NFFT];
static float g_tw_re[DV_NFFT / 2];
static float g_tw_im[DV_NFFT / 2];
static int   g_tw_ready = 0;

static void twiddle_init(void)
{
    if (g_tw_ready) {
        return;
    }
    for (int i = 0; i < DV_NFFT / 2; i++) {
        float ang = -2.0f * DV_PI * (float)i / (float)DV_NFFT;
        g_tw_re[i] = cosf(ang);
        g_tw_im[i] = sinf(ang);
    }
    g_tw_ready = 1;
}

/* 原地 radix-2 复数 FFT，长度固定 512（实数输入时虚部置 0）。 */
static void fft512(float *re, float *im)
{
    const int n = DV_NFFT;

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int step = n / len;
        for (int base = 0; base < n; base += len) {
            for (int k = 0; k < half; k++) {
                const float wr = g_tw_re[k * step];
                const float wi = g_tw_im[k * step];
                const int a = base + k;
                const int b = a + half;
                const float vr = re[b] * wr - im[b] * wi;
                const float vi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - vr;
                im[b] = im[a] - vi;
                re[a] += vr;
                im[a] += vi;
            }
        }
    }
}

static int frame_count(int n_samples)
{
    if (n_samples < DV_NFFT) {
        return 1;
    }
    return 1 + (n_samples - DV_NFFT) / DV_HOP;
}

int dv_extract_features(const float *pcm, int n_samples,
                        float *seq_out, int *out_T)
{
    if (pcm == NULL || seq_out == NULL || out_T == NULL || n_samples <= 0) {
        return DV_ERR_AUDIO;
    }
    const int T = frame_count(n_samples);
    if (T > DV_MAX_T) {
        return DV_ERR_FRAMES;
    }
    twiddle_init();

    /* ---- 逐帧：预加重 + 加窗 + rfft + 功率 + mel + log ---- */
    for (int t = 0; t < T; t++) {
        const int base = t * DV_HOP;
        for (int i = 0; i < DV_NFFT; i++) {
            const int idx = base + i;
            float v = 0.0f;
            if (idx < n_samples) {
                /* pre_emphasis 在补零之前完成，所以越界样本直接是 0 */
                v = (idx == 0) ? pcm[0]
                               : pcm[idx] - 0.97f * pcm[idx - 1];
            }
            g_frame[i] = v * DV_HAMMING[i];
        }

        memcpy(g_re, g_frame, sizeof(g_frame));
        memset(g_im, 0, sizeof(g_im));
        fft512(g_re, g_im);

        for (int m = 0; m < DV_NMEL; m++) {
            float acc = 0.0f;
            const float *row = DV_MEL_FB[m];
            for (int k = 0; k < DV_NBINS; k++) {
                const float w = row[k];
                if (w != 0.0f) {
                    acc += (g_re[k] * g_re[k] + g_im[k] * g_im[k]) * w;
                }
            }
            g_mel[t][m] = logf(acc > 1e-10f ? acc : 1e-10f);
        }
    }

    /* ---- 整句 CMVN（沿时间轴，按维归一化） ---- */
    for (int m = 0; m < DV_NMEL; m++) {
        float mean = 0.0f;
        for (int t = 0; t < T; t++) {
            mean += g_mel[t][m];
        }
        mean /= (float)T;
        float var = 0.0f;
        for (int t = 0; t < T; t++) {
            const float d = g_mel[t][m] - mean;
            var += d * d;
        }
        var /= (float)T;
        const float std = sqrtf(var);
        const float inv = 1.0f / (std + 1e-8f);
        for (int t = 0; t < T; t++) {
            g_mel[t][m] = (g_mel[t][m] - mean) * inv;
        }
    }

    /* ---- 拼 [logmel_cmvn(40) | Δ(40)] ---- */
    const float den = 10.0f;   /* 2 * (1^2 + 2^2) */
    for (int t = 0; t < T; t++) {
        const int im2 = t - 2 < 0 ? 0 : (t - 2 >= T ? T - 1 : t - 2);
        const int im1 = t - 1 < 0 ? 0 : (t - 1 >= T ? T - 1 : t - 1);
        const int ip1 = t + 1 >= T ? T - 1 : t + 1;
        const int ip2 = t + 2 >= T ? T - 1 : t + 2;
        float *row = seq_out + (size_t)t * DV_IN_DIM;
        for (int m = 0; m < DV_NMEL; m++) {
            row[m] = g_mel[t][m];
            const float num = (g_mel[ip1][m] - g_mel[im1][m])
                            + 2.0f * (g_mel[ip2][m] - g_mel[im2][m]);
            row[DV_NMEL + m] = num / den;
        }
    }

    *out_T = T;
    return DV_OK;
}

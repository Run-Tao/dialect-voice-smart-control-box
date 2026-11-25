/* 主机对拍：把 firmware/main 下的同一份 C 代码用 gcc 编译，
 * 读 dump_reference.py 产出的参考数据，逐项比较。
 *
 * 比较三段，便于定位误差来源：
 *   1) 特征：C 的 seq vs numpy 的 seq
 *   2) 推理：用 numpy 的 seq 喂 C 的 dv_predict，比 logits
 *   3) 端到端：C 的 seq 直接喂 C 的 predict，比 argmax
 *
 * 用法：host_verify <ref_dir>
 * 退出码：0 全部通过；1 有指标超限。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dv_core.h"

#define TOL_SEQ_ABS   1e-3f
#define TOL_LOGIT_ABS 5e-2f

static float *read_f32(const char *path, size_t *count)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "打不开 %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *buf = (float *)malloc((size_t)bytes);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)bytes, f) != (size_t)bytes) {
        fprintf(stderr, "读取不足 %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *count = (size_t)bytes / sizeof(float);
    return buf;
}

static void report(const char *label, const float *a, const float *b, size_t n)
{
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > max_abs) {
            max_abs = d;
            worst = i;
        }
        const float denom = fabsf(b[i]) > 1e-6f ? fabsf(b[i]) : 1e-6f;
        const float r = d / denom;
        if (r > max_rel) {
            max_rel = r;
        }
    }
    printf("%-22s max|abs|=%.6g  max|rel|=%.3g  (worst idx %zu: %.6f vs %.6f)\n",
           label, max_abs, max_rel, worst, a[worst], b[worst]);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "用法: %s <ref_dir>\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    char path[512];
    size_t n_pcm = 0, n_seq = 0, n_in = 0, n_logits = 0;

    snprintf(path, sizeof(path), "%s/pcm.f32", dir);
    float *pcm = read_f32(path, &n_pcm);
    snprintf(path, sizeof(path), "%s/seq.f32", dir);
    float *seq_ref = read_f32(path, &n_seq);
    snprintf(path, sizeof(path), "%s/input.f32", dir);
    float *input_ref = read_f32(path, &n_in);
    snprintf(path, sizeof(path), "%s/logits.f32", dir);
    float *logits_ref = read_f32(path, &n_logits);

    if (!pcm || !seq_ref || !input_ref || !logits_ref) {
        return 2;
    }
    if (n_seq % DV_IN_DIM != 0) {
        fprintf(stderr, "seq.f32 大小不是 80 的整数倍\n");
        return 2;
    }
    const int T_ref = (int)(n_seq / DV_IN_DIM);
    printf("pcm=%zu 点   T_ref=%d   input=%zu   logits=%zu\n\n",
           n_pcm, T_ref, n_in, n_logits);

    /* ---- 1) 特征对拍 ---- */
    static float seq_c[DV_MAX_T * DV_IN_DIM];
    int T_c = 0;
    const int rc = dv_extract_features(pcm, (int)n_pcm, seq_c, &T_c);
    if (rc != DV_OK) {
        fprintf(stderr, "dv_extract_features 失败: %d\n", rc);
        return 2;
    }
    printf("帧数：C=%d  numpy=%d  %s\n", T_c, T_ref, T_c == T_ref ? "一致" : "不一致");
    if (T_c != T_ref) {
        return 1;
    }
    report("seq (C vs numpy)", seq_c, seq_ref, n_seq);

    /* ---- 2) 推理对拍：用 numpy 的 seq 喂 C ---- */
    dv_result_t r1;
    if (dv_predict(seq_ref, T_ref, &r1) != DV_OK) {
        fprintf(stderr, "dv_predict 失败\n");
        return 2;
    }
    report("logits (C vs numpy)", r1.logits, logits_ref, (size_t)DV_N_CLASSES);
    int best_ref = 0;
    for (int i = 1; i < DV_N_CLASSES; i++) {
        if (logits_ref[i] > logits_ref[best_ref]) {
            best_ref = i;
        }
    }
    printf("argmax：C=%s(%.4f)  numpy=%s  %s\n",
           dv_class_name(r1.command), r1.confidence,
           DV_CLASSES[best_ref],
           r1.command == best_ref ? "一致" : "不一致");

    /* ---- 3) 端到端：C 的 seq 再跑一次 ---- */
    dv_result_t r2;
    if (dv_predict(seq_c, T_c, &r2) != DV_OK) {
        return 2;
    }
    printf("端到端：C=%s(%.4f)  argmax %s\n",
           dv_class_name(r2.command), r2.confidence,
           r2.command == best_ref ? "一致" : "不一致");

    /* ---- 判定 ---- */
    float max_seq = 0.0f;
    for (size_t i = 0; i < n_seq; i++) {
        const float d = fabsf(seq_c[i] - seq_ref[i]);
        if (d > max_seq) {
            max_seq = d;
        }
    }
    float max_logit = 0.0f;
    for (int i = 0; i < DV_N_CLASSES; i++) {
        const float d = fabsf(r1.logits[i] - logits_ref[i]);
        if (d > max_logit) {
            max_logit = d;
        }
    }
    const int ok = (max_seq < TOL_SEQ_ABS)
                && (max_logit < TOL_LOGIT_ABS)
                && (r1.command == best_ref)
                && (r2.command == best_ref);
    printf("\n判定：seq %.2e (<%.0e)  logits %.2e (<%.0e)  -> %s\n",
           max_seq, TOL_SEQ_ABS, max_logit, TOL_LOGIT_ABS,
           ok ? "PASS" : "FAIL");

    free(pcm); free(seq_ref); free(input_ref); free(logits_ref);
    return ok ? 0 : 1;
}

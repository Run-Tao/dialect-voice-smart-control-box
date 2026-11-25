/* 批量对拍：读 batch_reference.py 打好的 PCM 包，逐条跑 C 版内核，
 * 统计「与标签一致率」和「与 numpy 决策一致率」。
 *
 * 用法：host_batch <batch_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dv_core.h"

static float *read_all(const char *path, size_t *count)
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
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *count = (size_t)bytes / sizeof(float);
    return buf;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "用法: %s <batch_dir>\n", argv[0]);
        return 2;
    }
    char path[512];
    size_t n_pcm = 0;

    snprintf(path, sizeof(path), "%s/pcm.f32", argv[1]);
    float *pcm = read_all(path, &n_pcm);
    if (!pcm) {
        return 2;
    }
    snprintf(path, sizeof(path), "%s/index.txt", argv[1]);
    FILE *index = fopen(path, "r");
    if (!index) {
        fprintf(stderr, "打不开 %s\n", path);
        free(pcm);
        return 2;
    }

    int total = 0;
    int hit_label = 0;
    int agree_numpy = 0;
    int failed = 0;
    char line[512];

    while (fgets(line, sizeof(line), index)) {
        char name[192], label[64], npy[64];
        long offset = 0;
        int count = 0;
        if (sscanf(line, "%191[^,],%63[^,],%63[^,],%ld,%d",
                   name, label, npy, &offset, &count) != 5) {
            continue;
        }
        if (count <= 0 || (size_t)(offset + count) > n_pcm) {
            failed++;
            continue;
        }

        dv_result_t r;
        memset(&r, 0, sizeof(r));
        if (dv_run(pcm + offset, count, &r) != DV_OK) {
            failed++;
            continue;
        }
        total++;
        const char *pred = dv_class_name(r.command);
        if (strcmp(pred, label) == 0) {
            hit_label++;
        }
        if (strcmp(pred, npy) == 0) {
            agree_numpy++;
        }
    }
    fclose(index);

    printf("样本 %d 条（失败 %d）\n", total, failed);
    if (total > 0) {
        printf("与标签一致      %d/%d = %.4f\n", hit_label, total, (double)hit_label / total);
        printf("与 numpy 一致   %d/%d = %.4f\n", agree_numpy, total,
               (double)agree_numpy / total);
    }
    free(pcm);
    return 0;
}

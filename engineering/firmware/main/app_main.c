/* 固件入口：串口输入 → 特征 → int8 推理 → JSON 结果。
 *
 * 为什么先做串口而不是 I2S 麦克风：麦克风与引脚由硬件负责人提供，固件这边
 * 必须能在一个「只有 USB 线」的条件下验证整条链路。串口跑通后再把 I2S 采集
 * 接到同一个 dv_run() 上即可，推理内核不需要改。
 *
 * 协议（UART0，921600 8N1，行/二进制混合）：
 *   PING\n                  -> PONG\n
 *   INFO\n                  -> {"model":...,"psram":...,"heap":...}\n
 *   RUNRAW\n                -> 随后 4 字节小端样本数 n + n*4 字节 float32 PCM
 *                              -> {"ok":1,"cmd":"LIGHT_ON","conf":0.99,"ms":23,...}\n
 *   MIC\n                   -> {"error":"mic_not_implemented"}（等 I2S 接入）
 *
 * 日志以 I/W/E 开头，PC 侧只解析以 { 或 PONG 开头的行。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#if __has_include("esp_psram.h")
#include "esp_psram.h"
#endif

#include "dv_core.h"

static const char *TAG = "dv";

#define DV_UART        UART_NUM_0
#define DV_UART_BAUD   921600
#define DV_UART_BUF    4096
#define DV_MAX_SAMPLES (16000 * 10)   /* 10 s 上限 */

static void reply(const char *text)
{
    uart_write_bytes(DV_UART, text, strlen(text));
}

static int read_exact(uint8_t *buf, int n, int timeout_ms)
{
    int got = 0;
    const int64_t deadline = esp_timer_get_time() + (int64_t)timeout_ms * 1000;
    while (got < n) {
        const int r = uart_read_bytes(DV_UART, buf + got, n - got, pdMS_TO_TICKS(50));
        if (r > 0) {
            got += r;
        } else if (esp_timer_get_time() > deadline) {
            break;
        }
    }
    return got;
}

static void handle_info(void)
{
    size_t psram = 0;
#if __has_include("esp_psram.h")
    psram = esp_psram_get_size();
#endif
    char out[320];
    snprintf(out, sizeof(out),
             "{\"model\":\"%s\",\"seeds\":%d,\"classes\":%d,\"frames\":%d,"
             "\"psram\":%u,\"heap_free\":%u,\"heap_min\":%u}\n",
             "v3_cnn_ensemble_vad(int8)", (int)DV_ACTIVE_SEEDS, (int)DV_N_CLASSES,
             (int)DV_N_FRAMES, (unsigned)psram,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT));
    reply(out);
}

static void handle_runraw(void)
{
    uint8_t hdr[4];
    if (read_exact(hdr, 4, 2000) != 4) {
        reply("{\"ok\":0,\"error\":\"hdr_timeout\"}\n");
        return;
    }
    const int n = (int)((uint32_t)hdr[0] | ((uint32_t)hdr[1] << 8)
                      | ((uint32_t)hdr[2] << 16) | ((uint32_t)hdr[3] << 24));
    if (n <= 0 || n > DV_MAX_SAMPLES) {
        reply("{\"ok\":0,\"error\":\"bad_len\"}\n");
        return;
    }

    float *pcm = (float *)heap_caps_malloc((size_t)n * sizeof(float), MALLOC_CAP_SPIRAM);
    if (pcm == NULL) {
        pcm = (float *)malloc((size_t)n * sizeof(float));
    }
    if (pcm == NULL) {
        reply("{\"ok\":0,\"error\":\"oom\"}\n");
        return;
    }

    const int want = n * 4;
    if (read_exact((uint8_t *)pcm, want, 8000) != want) {
        free(pcm);
        reply("{\"ok\":0,\"error\":\"pcm_timeout\"}\n");
        return;
    }

    dv_result_t r;
    memset(&r, 0, sizeof(r));
    const int64_t t0 = esp_timer_get_time();
    const int rc = dv_run(pcm, n, &r);
    const int64_t us = esp_timer_get_time() - t0;
    free(pcm);

    if (rc != DV_OK) {
        char out[96];
        snprintf(out, sizeof(out), "{\"ok\":0,\"error\":\"infer\",\"rc\":%d}\n", rc);
        reply(out);
        return;
    }

    char out[384];
    int len = snprintf(out, sizeof(out),
                       "{\"ok\":1,\"cmd\":\"%s\",\"idx\":%d,\"conf\":%.4f,"
                       "\"ms\":%.1f,\"T\":%d,\"logits\":[",
                       dv_class_name(r.command), r.command, r.confidence,
                       (double)us / 1000.0, r.n_frames);
    for (int c = 0; c < DV_N_CLASSES && len < (int)sizeof(out) - 16; c++) {
        len += snprintf(out + len, sizeof(out) - len, "%.3f%s",
                        (double)r.logits[c], c + 1 < DV_N_CLASSES ? "," : "");
    }
    snprintf(out + len, sizeof(out) - len, "]}\n");
    reply(out);
}

static void dispatch(const char *line)
{
    if (strcmp(line, "PING") == 0) {
        reply("PONG\n");
    } else if (strcmp(line, "INFO") == 0) {
        handle_info();
    } else if (strcmp(line, "RUNRAW") == 0) {
        handle_runraw();
    } else if (strcmp(line, "MIC") == 0) {
        reply("{\"ok\":0,\"error\":\"mic_not_implemented\"}\n");
    } else {
        reply("{\"ok\":0,\"error\":\"unknown_cmd\"}\n");
    }
}

void app_main(void)
{
    const uart_config_t cfg = {
        .baud_rate = DV_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(DV_UART, DV_UART_BUF * 2, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(DV_UART, &cfg));

    ESP_LOGI(TAG, "DialectVoice firmware up: %d seeds, %d classes, %d frames",
             (int)DV_ACTIVE_SEEDS, (int)DV_N_CLASSES, (int)DV_N_FRAMES);
    reply("READY\n");

    char line[64];
    int li = 0;
    for (;;) {
        uint8_t ch;
        const int r = uart_read_bytes(DV_UART, &ch, 1, pdMS_TO_TICKS(100));
        if (r <= 0) {
            continue;
        }
        if (ch == '\n' || ch == '\r') {
            if (li > 0) {
                line[li] = '\0';
                dispatch(line);
                li = 0;
            }
        } else if (li < (int)sizeof(line) - 1) {
            line[li++] = (char)ch;
        }
    }
}

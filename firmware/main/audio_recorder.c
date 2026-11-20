/* audio_recorder：按住说话录音 + SD 落盘，双任务解耦。
 *
 * 任务划分（硬约束：SD 写不得与 I2S 采集同任务，防 DMA 溢出丢帧）：
 *   capture_task  prio 10 / core1：只读 I2S（esp_codec_dev 主麦通道），写 PSRAM 环形线性缓冲
 *   sd_writer_task prio 5 / core0：拼 44 字节 WAV 头 + fwrite 到 /sdcard/dialect/
 * 音频缓冲 320000B（10s@16k/16bit/mono）显式 heap_caps_malloc(MALLOC_CAP_SPIRAM)。
 */
#include "audio_recorder.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "board_bsp.h"
#include "board_pins.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"

static const char *TAG = "recorder";

#define REC_CHUNK_SAMPLES 240                 /* 15ms 一帧，与 BSP DMA 帧长一致 */
#define REC_SAMPLE_RATE   BSP_AUDIO_SAMPLE_RATE
#define BIT_START         (1 << 0)
#define BIT_STOP          (1 << 1)

typedef struct {
    uint32_t samples;
    uint32_t duration_ms;
} save_job_t;

/* 44 字节标准 PCM WAV 头 */
typedef struct __attribute__((packed)) {
    char     riff[4];
    uint32_t riff_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char     data[4];
    uint32_t data_size;
} wav_header_t;

static int16_t           *s_buf;             /* PSRAM */
static uint32_t           s_cap_samples;
static volatile uint32_t  s_len_samples;
static volatile bool      s_recording;
static volatile bool      s_saving;
static volatile uint8_t   s_level;
static TickType_t         s_start_tick;
static uint32_t           s_seq;
static EventGroupHandle_t s_events;
static QueueHandle_t      s_save_queue;
static audio_recorder_saved_cb s_cb;

static void make_sample_id(char *out, size_t out_size)
{
    time_t now = time(NULL);
    if (now > 1700000000) {                  /* SNTP 已同步 */
        struct tm tmv;
        localtime_r(&now, &tmv);
        char ts[24];
        strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tmv);
        snprintf(out, out_size, "%s_%03u", ts, (unsigned)(s_seq++ % 1000));
    } else {                                 /* 未联网校时：开机秒数兜底，仍保证唯一 */
        snprintf(out, out_size, "boot%06u_%03u",
                 (unsigned)(esp_timer_get_time() / 1000000), (unsigned)(s_seq++ % 1000));
    }
}

static void capture_task(void *arg)
{
    (void)arg;
    for (;;) {
        xEventGroupWaitBits(s_events, BIT_START, pdTRUE, pdTRUE, portMAX_DELAY);
        if (board_bsp_audio_open_input() != ESP_OK) {
            ESP_LOGE(TAG, "open input failed");
            s_recording = false;
            if (s_cb) s_cb("", "", 0, ESP_FAIL);
            continue;
        }
        s_len_samples = 0;
        s_level = 0;
        s_start_tick = xTaskGetTickCount();
        s_recording = true;
        ESP_LOGI(TAG, "capture start");

        while (!(xEventGroupGetBits(s_events) & BIT_STOP)) {
            if (s_len_samples + REC_CHUNK_SAMPLES > s_cap_samples) {
                ESP_LOGW(TAG, "reach %ds cap, auto stop", AUDIO_RECORDER_MAX_SECONDS);
                break;
            }
            esp_err_t err = board_bsp_audio_read_mono(s_buf + s_len_samples,
                                                      REC_CHUNK_SAMPLES);
            size_t got = (err == ESP_OK) ? REC_CHUNK_SAMPLES : 0;
            if (err != ESP_OK || got == 0) {
                ESP_LOGE(TAG, "i2s read err=%s got=%u", esp_err_to_name(err), (unsigned)got);
                break;
            }
            int64_t sum = 0;
            for (size_t i = 0; i < got; i++) {
                int32_t v = s_buf[s_len_samples + i];
                sum += v * v;
            }
            uint32_t rms = (uint32_t)sqrtf((float)(sum / (int64_t)got));
            uint32_t lv = rms * 100 / 3000;
            s_level = (uint8_t)(lv > 100 ? 100 : lv);
            s_len_samples += (uint32_t)got;
        }

        s_recording = false;
        uint32_t ms = (uint32_t)((xTaskGetTickCount() - s_start_tick) * portTICK_PERIOD_MS);
        board_bsp_audio_close_input();
        xEventGroupClearBits(s_events, BIT_STOP);
        save_job_t job = { .samples = s_len_samples, .duration_ms = ms };
        if (xQueueSend(s_save_queue, &job, pdMS_TO_TICKS(1000)) != pdTRUE) {
            ESP_LOGE(TAG, "save queue full, drop recording");
            if (s_cb) s_cb("", "", ms, ESP_ERR_NO_MEM);
        }
    }
}

static void sd_writer_task(void *arg)
{
    (void)arg;
    save_job_t job;
    for (;;) {
        if (xQueueReceive(s_save_queue, &job, portMAX_DELAY) != pdTRUE) continue;
        s_saving = true;
        mkdir(BSP_SD_MOUNT_POINT "/dialect", 0777);   /* 已存在则忽略 EEXIST */

        char sample_id[40];
        make_sample_id(sample_id, sizeof(sample_id));
        char path[96];
        snprintf(path, sizeof(path), BSP_SD_MOUNT_POINT "/dialect/%s.wav", sample_id);

        esp_err_t err = ESP_OK;
        FILE *fp = fopen(path, "wb");
        if (!fp) {
            err = ESP_FAIL;
            ESP_LOGE(TAG, "fopen %s failed", path);
        } else {
            uint32_t data_bytes = job.samples * sizeof(int16_t);
            wav_header_t h = {
                .riff = {'R', 'I', 'F', 'F'},
                .riff_size = 36 + data_bytes,
                .wave = {'W', 'A', 'V', 'E'},
                .fmt = {'f', 'm', 't', ' '},
                .fmt_size = 16,
                .audio_format = 1,
                .channels = 1,
                .sample_rate = REC_SAMPLE_RATE,
                .byte_rate = REC_SAMPLE_RATE * 2,
                .block_align = 2,
                .bits_per_sample = 16,
                .data = {'d', 'a', 't', 'a'},
                .data_size = data_bytes,
            };
            if (fwrite(&h, 1, sizeof(h), fp) != sizeof(h) ||
                fwrite(s_buf, sizeof(int16_t), job.samples, fp) != job.samples) {
                err = ESP_FAIL;
                ESP_LOGE(TAG, "fwrite %s failed", path);
            }
            if (fclose(fp) != 0 && err == ESP_OK) err = ESP_FAIL;
        }
        s_saving = false;
        ESP_LOGI(TAG, "saved %s (%u ms, %u samples, err=%s)",
                 path, (unsigned)job.duration_ms, (unsigned)job.samples, esp_err_to_name(err));
        if (s_cb) s_cb(path, sample_id, job.duration_ms, err);
    }
}

esp_err_t audio_recorder_init(audio_recorder_saved_cb cb)
{
    s_cb = cb;
    s_cap_samples = REC_SAMPLE_RATE * AUDIO_RECORDER_MAX_SECONDS;
    s_buf = heap_caps_malloc(s_cap_samples * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!s_buf) {
        ESP_LOGE(TAG, "PSRAM alloc %u B failed", (unsigned)(s_cap_samples * 2));
        return ESP_ERR_NO_MEM;
    }
    s_events = xEventGroupCreate();
    s_save_queue = xQueueCreate(2, sizeof(save_job_t));
    if (!s_events || !s_save_queue) return ESP_ERR_NO_MEM;
    if (xTaskCreatePinnedToCore(capture_task, "rec_cap", 4096, NULL, 10, NULL, 1) != pdPASS ||
        xTaskCreatePinnedToCore(sd_writer_task, "rec_sd", 4096, NULL, 5, NULL, 0) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "init ok, buffer %u B in PSRAM", (unsigned)(s_cap_samples * 2));
    return ESP_OK;
}

esp_err_t audio_recorder_start(void)
{
    if (s_recording || s_saving) return ESP_ERR_INVALID_STATE;
    if (!board_bsp_sd_ready()) {
        ESP_LOGE(TAG, "SD not ready, refuse record");
        return ESP_ERR_INVALID_STATE;
    }
    xEventGroupClearBits(s_events, BIT_STOP);
    xEventGroupSetBits(s_events, BIT_START);
    return ESP_OK;
}

esp_err_t audio_recorder_stop(void)
{
    xEventGroupSetBits(s_events, BIT_STOP);
    return ESP_OK;
}

bool audio_recorder_is_recording(void) { return s_recording; }
bool audio_recorder_is_saving(void) { return s_saving; }

uint32_t audio_recorder_elapsed_ms(void)
{
    if (!s_recording) return 0;
    return (uint32_t)((xTaskGetTickCount() - s_start_tick) * portTICK_PERIOD_MS);
}

uint8_t audio_recorder_level(void) { return s_level; }
#pragma once

#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 云端 /api/v1/speech/understand 的 JSON 契约（与 backend/main.py 一一对应） */
typedef struct {
    char  sample_id[40];
    char  dialect[16];          /* cantonese / mandarin / ... */
    char  raw_text[256];
    char  normalized_text[256];
    char  intent[24];           /* device_control / unknown */
    char  location[24];         /* living_room / bedroom / ... */
    char  device[24];           /* light / ac / tv / curtain / fan / socket */
    char  action[24];           /* on / off / brighter / set_temp / ... */
    char  slots_json[192];      /* 完整 slots 对象序列化（含 delta/temperature） */
    float confidence;           /* 0~1 */
    char  reply_text[256];
} understand_result_t;

/* 上传完成回调：在 upload 任务上下文调用，res 为 NULL 表示 err != ESP_OK */
typedef void (*understand_cb)(const understand_result_t *res, esp_err_t err);
typedef void (*wifi_state_cb)(bool connected);

esp_err_t network_client_init(const char *base_url, const char *device_id,
                              understand_cb cb, wifi_state_cb wifi_cb);
esp_err_t network_client_wifi_start(const char *ssid, const char *password);
bool network_client_is_connected(void);
bool network_client_time_synced(void);

/* 异步上传：multipart(file/device_id/timestamp/sample_id)，独立 upload 任务执行，
 * 请求体整体在 PSRAM 拼装（WAV <= 320KB），不阻塞调用者。 */
esp_err_t network_client_upload_async(const char *wav_path, const char *sample_id,
                                      const char *timestamp);

/* 同步 POST /api/v1/speech/feedback（body 极小，内部 SRAM 即可） */
esp_err_t network_client_feedback(const char *sample_id, bool is_correct);

#ifdef __cplusplus
}
#endif
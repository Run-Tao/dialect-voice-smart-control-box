/* network_client：Wi-Fi STA + SNTP 校时 + 云端 HTTP 协议适配。
 *
 * 硬约束落地：
 *   * multipart 请求体（WAV 全量 <= 320KB + 边界串）显式 heap_caps_malloc(MALLOC_CAP_SPIRAM)
 *   * 上传在独立 upload 任务执行，与 I2S 采集任务、SD 写任务互不阻塞
 *   * esp_http_client 内部收发缓冲 4KB/2KB（< 16KB 内部 SRAM 上限）
 */
#include "network_client.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "cJSON.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_sntp.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"

static const char *TAG = "net";

#define BOUNDARY "----DialectVoiceBoundary7MA4YWxk"
#define BIT_CONNECTED   (1 << 0)
#define BIT_TIME_SYNCED (1 << 1)
#define MAX_WAV_BYTES   (400 * 1024)

typedef struct {
    char path[96];
    char sample_id[40];
    char ts[24];
} upload_job_t;

static char s_base_url[128];
static char s_device_id[32];
static understand_cb s_cb;
static wifi_state_cb s_wifi_cb;
static EventGroupHandle_t s_events;
static QueueHandle_t s_upload_queue;
static bool s_sntp_started;

static void upload_task(void *arg);

static void sntp_sync_cb(struct timeval *tv)
{
    (void)tv;
    xEventGroupSetBits(s_events, BIT_TIME_SYNCED);
    ESP_LOGI(TAG, "SNTP time synced");
}

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_events, BIT_CONNECTED);
        if (s_wifi_cb) s_wifi_cb(false);
        ESP_LOGW(TAG, "wifi disconnected, reconnecting");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *evt = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&evt->ip_info.ip));
        xEventGroupSetBits(s_events, BIT_CONNECTED);
        if (s_wifi_cb) s_wifi_cb(true);
        if (!s_sntp_started) {
            s_sntp_started = true;
            esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
            esp_sntp_setservername(0, "ntp.aliyun.com");
            esp_sntp_setservername(1, "pool.ntp.org");
            esp_sntp_set_time_sync_notification_cb(sntp_sync_cb);
            esp_sntp_init();
        }
    }
}

esp_err_t network_client_init(const char *base_url, const char *device_id,
                              understand_cb cb, wifi_state_cb wifi_cb)
{
    snprintf(s_base_url, sizeof(s_base_url), "%s", base_url);
    snprintf(s_device_id, sizeof(s_device_id), "%s", device_id);
    s_cb = cb;
    s_wifi_cb = wifi_cb;
    s_events = xEventGroupCreate();
    s_upload_queue = xQueueCreate(2, sizeof(upload_job_t));
    if (!s_events || !s_upload_queue) return ESP_ERR_NO_MEM;
    if (xTaskCreatePinnedToCore(upload_task, "net_up", 6144, NULL, 5, NULL, 0) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t network_client_wifi_start(const char *ssid, const char *password)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler, NULL, NULL));
    wifi_config_t wc = {0};
    snprintf((char *)wc.sta.ssid, sizeof(wc.sta.ssid), "%s", ssid);
    snprintf((char *)wc.sta.password, sizeof(wc.sta.password), "%s", password);
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "wifi sta starting, ssid=%s", ssid);
    return ESP_OK;
}

bool network_client_is_connected(void)
{
    return s_events && (xEventGroupGetBits(s_events) & BIT_CONNECTED);
}

bool network_client_time_synced(void)
{
    return s_events && (xEventGroupGetBits(s_events) & BIT_TIME_SYNCED);
}

static void fill_result_from_json(const char *json, understand_result_t *res)
{
    memset(res, 0, sizeof(*res));
    cJSON *root = cJSON_Parse(json);
    if (!root) return;
    cJSON *it;
#define STR_FIELD(key, dst) do { \
        it = cJSON_GetObjectItem(root, key); \
        if (cJSON_IsString(it)) snprintf((dst), sizeof(res->dst), "%s", it->valuestring); \
    } while (0)
    STR_FIELD("sample_id", sample_id);
    STR_FIELD("dialect", dialect);
    STR_FIELD("raw_text", raw_text);
    STR_FIELD("normalized_text", normalized_text);
    STR_FIELD("intent", intent);
    STR_FIELD("reply_text", reply_text);
#undef STR_FIELD
    it = cJSON_GetObjectItem(root, "confidence");
    if (cJSON_IsNumber(it)) res->confidence = (float)it->valuedouble;
    cJSON *slots = cJSON_GetObjectItem(root, "slots");
    if (cJSON_IsObject(slots)) {
        it = cJSON_GetObjectItem(slots, "location");
        if (cJSON_IsString(it)) snprintf(res->location, sizeof(res->location), "%s", it->valuestring);
        it = cJSON_GetObjectItem(slots, "device");
        if (cJSON_IsString(it)) snprintf(res->device, sizeof(res->device), "%s", it->valuestring);
        it = cJSON_GetObjectItem(slots, "action");
        if (cJSON_IsString(it)) snprintf(res->action, sizeof(res->action), "%s", it->valuestring);
        char *printed = cJSON_PrintUnformatted(slots);
        if (printed) {
            snprintf(res->slots_json, sizeof(res->slots_json), "%s", printed);
            cJSON_free(printed);
        }
    }
    cJSON_Delete(root);
}

static void upload_task(void *arg)
{
    (void)arg;
    upload_job_t job;
    for (;;) {
        if (xQueueReceive(s_upload_queue, &job, portMAX_DELAY) != pdTRUE) continue;

        struct stat st;
        if (stat(job.path, &st) != 0 || st.st_size <= 44 || st.st_size > MAX_WAV_BYTES) {
            ESP_LOGE(TAG, "bad wav %s", job.path);
            if (s_cb) s_cb(NULL, ESP_ERR_INVALID_SIZE);
            continue;
        }
        uint32_t wav_bytes = (uint32_t)st.st_size;

        char pre[512];
        int pre_len = snprintf(pre, sizeof(pre),
            "--" BOUNDARY "\r\nContent-Disposition: form-data; name=\"device_id\"\r\n\r\n%s\r\n"
            "--" BOUNDARY "\r\nContent-Disposition: form-data; name=\"timestamp\"\r\n\r\n%s\r\n"
            "--" BOUNDARY "\r\nContent-Disposition: form-data; name=\"sample_id\"\r\n\r\n%s\r\n"
            "--" BOUNDARY "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n",
            s_device_id, job.ts, job.sample_id, job.sample_id);
        char post[64];
        int post_len = snprintf(post, sizeof(post), "\r\n--" BOUNDARY "--\r\n");
        uint32_t body_len = (uint32_t)pre_len + wav_bytes + (uint32_t)post_len;

        /* 大块 HTTP 请求体：显式 PSRAM */
        char *body = heap_caps_malloc(body_len + 1, MALLOC_CAP_SPIRAM);
        if (!body) {
            ESP_LOGE(TAG, "PSRAM body alloc %u B failed", (unsigned)body_len);
            if (s_cb) s_cb(NULL, ESP_ERR_NO_MEM);
            continue;
        }
        memcpy(body, pre, pre_len);
        FILE *fp = fopen(job.path, "rb");
        size_t got = fp ? fread(body + pre_len, 1, wav_bytes, fp) : 0;
        if (fp) fclose(fp);
        if (got != wav_bytes) {
            heap_caps_free(body);
            if (s_cb) s_cb(NULL, ESP_FAIL);
            continue;
        }
        memcpy(body + pre_len + wav_bytes, post, post_len);

        char url[160];
        snprintf(url, sizeof(url), "%s/api/v1/speech/understand", s_base_url);
        esp_http_client_config_t hcfg = {
            .url = url,
            .method = HTTP_METHOD_POST,
            .timeout_ms = 30000,
            .buffer_size = 4096,
            .buffer_size_tx = 2048,
        };
        esp_http_client_handle_t client = esp_http_client_init(&hcfg);
        esp_http_client_set_header(client, "Content-Type",
                                   "multipart/form-data; boundary=" BOUNDARY);
        esp_http_client_set_post_field(client, body, (int)body_len);
        esp_err_t err = esp_http_client_perform(client);
        int status = esp_http_client_get_status_code(client);
        ESP_LOGI(TAG, "upload %s -> http %d, err=%s", job.sample_id, status,
                 esp_err_to_name(err));

        understand_result_t res;
        esp_err_t cb_err = ESP_OK;
        if (err == ESP_OK && status == 200) {
            int rlen = esp_http_client_get_content_length(client);
            if (rlen <= 0 || rlen > 64 * 1024) {
                cb_err = ESP_FAIL;
            } else {
                char *resp = (rlen > 8192)
                    ? heap_caps_malloc(rlen + 1, MALLOC_CAP_SPIRAM)
                    : malloc(rlen + 1);
                if (!resp) {
                    cb_err = ESP_ERR_NO_MEM;
                } else {
                    int n = esp_http_client_read_response(client, resp, rlen);
                    resp[n > 0 ? n : 0] = '\0';
                    fill_result_from_json(resp, &res);
                    heap_caps_free(resp);
                    if (res.sample_id[0] == '\0') cb_err = ESP_FAIL;
                }
            }
        } else {
            cb_err = (err != ESP_OK) ? err : ESP_FAIL;
        }
        esp_http_client_cleanup(client);
        heap_caps_free(body);
        if (s_cb) s_cb(cb_err == ESP_OK ? &res : NULL, cb_err);
    }
}

esp_err_t network_client_upload_async(const char *wav_path, const char *sample_id,
                                      const char *timestamp)
{
    upload_job_t job = {0};
    snprintf(job.path, sizeof(job.path), "%s", wav_path);
    snprintf(job.sample_id, sizeof(job.sample_id), "%s", sample_id);
    if (timestamp && timestamp[0]) {
        snprintf(job.ts, sizeof(job.ts), "%s", timestamp);
    } else {
        time_t now = time(NULL);
        struct tm tmv;
        gmtime_r(&now, &tmv);
        strftime(job.ts, sizeof(job.ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    }
    if (xQueueSend(s_upload_queue, &job, pdMS_TO_TICKS(500)) != pdTRUE) {
        return ESP_ERR_INVALID_STATE;
    }
    return ESP_OK;
}

esp_err_t network_client_feedback(const char *sample_id, bool is_correct)
{
    char url[160];
    snprintf(url, sizeof(url), "%s/api/v1/speech/feedback", s_base_url);
    char body[160];
    snprintf(body, sizeof(body), "{\"sample_id\":\"%s\",\"is_correct\":%s}",
             sample_id, is_correct ? "true" : "false");
    esp_http_client_config_t hcfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 10000,
        .buffer_size = 2048,
        .buffer_size_tx = 1024,
    };
    esp_http_client_handle_t client = esp_http_client_init(&hcfg);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, body, (int)strlen(body));
    esp_err_t err = esp_http_client_perform(client);
    int status = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);
    ESP_LOGI(TAG, "feedback %s correct=%d -> http %d err=%s",
             sample_id, (int)is_correct, status, esp_err_to_name(err));
    return (err == ESP_OK && status == 200) ? ESP_OK : ESP_FAIL;
}
/* main：端云协同方言语音终端状态机。
 *
 * IDLE → RECORDING → UPLOADING → DISPLAY_AND_EXECUTE → (FEEDBACK) → IDLE
 * 触发源双通道：BOOT 键(GPIO0) ISR 边沿 + 屏幕「按住说话」触摸事件，汇入同一事件队列。
 * 10s 上限由 audio_recorder 采集任务自动收尾；落盘/上传完成经回调回到本任务。
 * 执行门槛：confidence >= 0.7 且 intent == device_control（与云端契约一致）。
 */
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "audio_recorder.h"
#include "board_bsp.h"
#include "board_pins.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "network_client.h"
#include "nvs_flash.h"
#include "ui_manager.h"

static const char *TAG = "app";

/* ---- 部署配置：烧录前修改，或 idf.py -DWIFI_SSID=... 覆盖 ---- */
#ifndef WIFI_SSID
#define WIFI_SSID     "YOUR_WIFI_SSID"
#endif
#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#endif
#ifndef CLOUD_BASE_URL
#define CLOUD_BASE_URL "http://192.168.1.100:8000"   /* 运行 backend 的 PC 局域网地址 */
#endif
#ifndef DEVICE_ID
#define DEVICE_ID     "esp32-s3-001"
#endif

#define CONF_GATE 0.7f

typedef enum {
    ST_IDLE = 0,
    ST_RECORDING,
    ST_UPLOADING,
    ST_DISPLAY,
} app_state_t;

typedef enum {
    EVT_TALK_PRESS = 0,
    EVT_TALK_RELEASE,
    EVT_REC_SAVED,
    EVT_UPLOAD_DONE,
    EVT_FEEDBACK,
    EVT_RETRY,
    EVT_WIFI,
} app_evt_type_t;

typedef struct {
    app_evt_type_t type;
    union {
        struct {
            char path[96];
            char sample_id[40];
            uint32_t duration_ms;
            esp_err_t err;
        } rec;
        struct {
            understand_result_t res;
            esp_err_t err;
        } up;
        bool correct;
        bool connected;
    };
} app_event_t;

static QueueHandle_t s_queue;
static app_state_t s_state = ST_IDLE;
static char s_last_sample_id[40];
static int64_t s_last_press_us, s_last_release_us;

static void post_event(const app_event_t *evt)
{
    xQueueSend(s_queue, evt, pdMS_TO_TICKS(100));
}

/* ---- 各任务上下文回调 → 事件队列 ---- */

static void recorder_saved_cb(const char *wav_path, const char *sample_id,
                              uint32_t duration_ms, esp_err_t err)
{
    app_event_t evt = { .type = EVT_REC_SAVED };
    snprintf(evt.rec.path, sizeof(evt.rec.path), "%s", wav_path);
    snprintf(evt.rec.sample_id, sizeof(evt.rec.sample_id), "%s", sample_id);
    evt.rec.duration_ms = duration_ms;
    evt.rec.err = err;
    post_event(&evt);
}

static void understand_done_cb(const understand_result_t *res, esp_err_t err)
{
    app_event_t evt = { .type = EVT_UPLOAD_DONE };
    evt.up.err = err;
    if (res) evt.up.res = *res;
    post_event(&evt);
}

static void wifi_state_cb(bool connected)
{
    app_event_t evt = { .type = EVT_WIFI, .connected = connected };
    post_event(&evt);
}

static void ui_talk_press(void)
{
    app_event_t evt = { .type = EVT_TALK_PRESS };
    post_event(&evt);
}

static void ui_talk_release(void)
{
    app_event_t evt = { .type = EVT_TALK_RELEASE };
    post_event(&evt);
}

static void ui_feedback(bool correct)
{
    app_event_t evt = { .type = EVT_FEEDBACK, .correct = correct };
    post_event(&evt);
}

static void ui_retry(void)
{
    app_event_t evt = { .type = EVT_RETRY };
    post_event(&evt);
}

static void IRAM_ATTR key_isr_handler(void *arg)
{
    (void)arg;
    app_event_t evt = {
        .type = (gpio_get_level(BSP_KEY_USER_GPIO) == BSP_KEY_ACTIVE_LEVEL)
                    ? EVT_TALK_PRESS : EVT_TALK_RELEASE,
    };
    BaseType_t yielded = pdFALSE;
    xQueueSendFromISR(s_queue, &evt, &yielded);
    portYIELD_FROM_ISR(yielded);
}

/* ---- 提示音与外设执行 ---- */

static void beep(uint32_t freq_hz, uint32_t ms)
{
    if (board_bsp_audio_open_output() != ESP_OK) return;
    int16_t pcm[1600];                       /* 100ms @16k，栈上 3.2KB */
    uint32_t n = ms * BSP_AUDIO_SAMPLE_RATE / 1000;
    if (n > 1600) n = 1600;
    for (uint32_t i = 0; i < n; i++) {
        pcm[i] = (int16_t)(12000.0f * sinf(2.0f * 3.14159265f * freq_hz * i
                                           / BSP_AUDIO_SAMPLE_RATE));
    }
    board_bsp_audio_write_mono(pcm, n);
    board_bsp_audio_close_output();
}

static void execute_action(const understand_result_t *res)
{
    if (strcmp(res->intent, "device_control") != 0 || res->confidence < CONF_GATE) {
        ESP_LOGW(TAG, "gate not passed (intent=%s conf=%.2f), display only",
                 res->intent, res->confidence);
        return;
    }
    /* 演示绑定：扩展口1=灯，扩展口2=风扇；其余设备云端状态已更新，仅日志 */
    if (strcmp(res->device, "light") == 0) {
        board_bsp_exp_set(0, strcmp(res->action, "on") == 0);
    } else if (strcmp(res->device, "fan") == 0) {
        board_bsp_exp_set(1, strcmp(res->action, "on") == 0);
    }
    ESP_LOGI(TAG, "execute %s/%s @%s conf=%.2f slots=%s",
             res->device, res->action, res->location, res->confidence, res->slots_json);
}

/* ---- 状态机 ---- */

static void handle_event(const app_event_t *evt)
{
    int64_t now = esp_timer_get_time();
    switch (evt->type) {
    case EVT_TALK_PRESS:
        if (now - s_last_press_us < 30000) break;      /* 30ms 消抖 */
        s_last_press_us = now;
        if (s_state != ST_IDLE) break;
        if (!board_bsp_sd_ready()) {
            ui_manager_show_error("SD 卡未就绪，无法录音存档");
            break;
        }
        if (audio_recorder_start() != ESP_OK) {
            ui_manager_show_error("录音启动失败（上次尚未落盘）");
            break;
        }
        s_state = ST_RECORDING;
        ui_manager_show_recording();
        beep(880, 80);
        break;

    case EVT_TALK_RELEASE:
        if (now - s_last_release_us < 30000) break;
        s_last_release_us = now;
        if (s_state != ST_RECORDING) break;
        audio_recorder_stop();
        beep(660, 80);
        break;

    case EVT_REC_SAVED:
        if (s_state != ST_RECORDING) break;
        if (evt->rec.err != ESP_OK || evt->rec.path[0] == '\0') {
            ui_manager_show_error("录音保存失败，请检查 SD 卡");
            s_state = ST_IDLE;
            break;
        }
        snprintf(s_last_sample_id, sizeof(s_last_sample_id), "%s", evt->rec.sample_id);
        char ts[24];
        time_t t = time(NULL);
        struct tm tmv;
        gmtime_r(&t, &tmv);
        strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);
        if (network_client_upload_async(evt->rec.path, evt->rec.sample_id, ts) != ESP_OK) {
            ui_manager_show_error("上传队列满，请重试");
            s_state = ST_IDLE;
            break;
        }
        s_state = ST_UPLOADING;
        ui_manager_show_uploading();
        break;

    case EVT_UPLOAD_DONE:
        if (s_state != ST_UPLOADING) break;
        if (evt->up.err != ESP_OK) {
            ui_manager_show_error("云端无响应或返回解析失败");
            s_state = ST_IDLE;
            break;
        }
        ui_manager_show_result(&evt->up.res);
        execute_action(&evt->up.res);
        s_state = ST_DISPLAY;
        break;

    case EVT_FEEDBACK:
        if (s_state != ST_DISPLAY) break;
        if (network_client_feedback(s_last_sample_id, evt->correct) != ESP_OK) {
            ESP_LOGW(TAG, "feedback post failed, sample=%s", s_last_sample_id);
        }
        ui_manager_show_idle();
        s_state = ST_IDLE;
        break;

    case EVT_RETRY:
        ui_manager_show_idle();
        s_state = ST_IDLE;
        break;

    case EVT_WIFI:
        ui_manager_set_wifi(evt->connected);
        break;
    }
}

static void app_task(void *arg)
{
    (void)arg;
    app_event_t evt;
    for (;;) {
        if (xQueueReceive(s_queue, &evt, pdMS_TO_TICKS(100)) == pdTRUE) {
            handle_event(&evt);
        } else if (s_state == ST_RECORDING) {
            ui_manager_update_recording(audio_recorder_level(), audio_recorder_elapsed_ms());
        }
    }
}

void app_main(void)
{
    esp_err_t nvs = nvs_flash_init();
    if (nvs == ESP_ERR_NVS_NO_FREE_PAGES || nvs == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs);

    ESP_ERROR_CHECK(board_bsp_init());

    s_queue = xQueueCreate(6, sizeof(app_event_t));
    assert(s_queue);

    const ui_callbacks_t uicb = {
        .talk_press = ui_talk_press,
        .talk_release = ui_talk_release,
        .feedback = ui_feedback,
        .retry = ui_retry,
    };
    ESP_ERROR_CHECK(ui_manager_init(&uicb));
    ESP_ERROR_CHECK(audio_recorder_init(recorder_saved_cb));
    ESP_ERROR_CHECK(network_client_init(CLOUD_BASE_URL, DEVICE_ID,
                                        understand_done_cb, wifi_state_cb));
    ESP_ERROR_CHECK(network_client_wifi_start(WIFI_SSID, WIFI_PASSWORD));

    const gpio_config_t key_cfg = {
        .pin_bit_mask = 1ULL << BSP_KEY_USER_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_ANYEDGE,
    };
    ESP_ERROR_CHECK(gpio_config(&key_cfg));
    ESP_ERROR_CHECK(gpio_install_isr_service(0));
    ESP_ERROR_CHECK(gpio_isr_handler_add(BSP_KEY_USER_GPIO, key_isr_handler, NULL));

    xTaskCreatePinnedToCore(app_task, "app_fsm", 8192, NULL, 6, NULL, 0);
    ESP_LOGI(TAG, "dialect voice terminal up: cloud=%s device=%s", CLOUD_BASE_URL, DEVICE_ID);
    ESP_LOGW(TAG, "烧录前请对照立创实战派 V1.0 原理图核实 board_pins.h 扩展口引脚(GPIO10/11)");
}
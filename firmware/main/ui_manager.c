/* ui_manager：ST7789 + FT6336 + LVGL8 五态界面（手写 LVGL 端口层）。
 *
 * 端口层构成：
 *   flush_cb  -> esp_lcd_panel_draw_bitmap（SPI DMA 异步），完成中断里 lv_disp_flush_ready
 *   tick      -> esp_timer 5ms 周期 lv_tick_inc
 *   touch     -> esp_lcd_touch_ft5x06 读坐标，lv_indev read_cb 上报
 *   并发      -> 所有 LVGL 调用（含 lv_timer_handler）持同一 mutex
 * 显存双缓冲 2 x 320x40x2B = 50KB，显式 PSRAM（>16KB 不进内部 SRAM）。
 * 中文：fonts/ui_font_16.c（lv_font_conv 子集），ASCII/符号回退 montserrat_14。
 */
#include "ui_manager.h"

#include <stdio.h>
#include <string.h>

#include "board_bsp.h"
#include "board_pins.h"
#include "driver/spi_master.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_st7789.h"
#include "esp_lcd_touch.h"
#include "esp_lcd_touch_ft5x06.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lvgl.h"

static const char *TAG = "ui";

LV_FONT_DECLARE(ui_font_16);

#define LV_BUF_PIXELS (BSP_LCD_H_RES * 40)

enum { P_IDLE = 0, P_REC, P_UP, P_RES, P_ERR, P_COUNT };

static esp_lcd_panel_io_handle_t s_io;
static esp_lcd_panel_handle_t    s_panel;
static esp_lcd_touch_handle_t    s_tp;
static lv_disp_t                *s_disp;
static SemaphoreHandle_t         s_lv_mutex;
static lv_disp_draw_buf_t        s_draw_buf;
static ui_callbacks_t            s_cb;

static lv_obj_t *s_lbl_wifi;
static lv_obj_t *s_panel[P_COUNT];
static lv_obj_t *s_bar_level;
static lv_obj_t *s_lbl_elapsed;
static lv_obj_t *s_lbl_reply;
static lv_obj_t *s_lbl_raw;
static lv_obj_t *s_lbl_norm;
static lv_obj_t *s_lbl_act;
static lv_obj_t *s_lbl_conf;
static lv_obj_t *s_lbl_err;

static bool ui_lock(void)
{
    return xSemaphoreTake(s_lv_mutex, pdMS_TO_TICKS(1000)) == pdTRUE;
}

static void ui_unlock(void)
{
    xSemaphoreGive(s_lv_mutex);
}

static bool lcd_flush_ready_cb(esp_lcd_panel_io_handle_t io, esp_lcd_panel_io_event_data_t *edata,
                               void *user_ctx)
{
    (void)io; (void)edata; (void)user_ctx;
    lv_disp_flush_ready(s_disp);
    return false;
}

static void lcd_flush_cb(lv_disp_drv_t *drv, const lv_area_t *area, lv_color_t *color_p)
{
    (void)drv;
    esp_lcd_panel_draw_bitmap(s_panel, area->x1, area->y1, area->x2 + 1, area->y2 + 1, color_p);
}

static void touch_read_cb(lv_indev_drv_t *drv, lv_indev_data_t *data)
{
    (void)drv;
    uint16_t x[1] = {0}, y[1] = {0};
    uint8_t cnt = 0;
    esp_lcd_touch_read_data(s_tp);
    esp_lcd_touch_get_coordinates(s_tp, x, y, NULL, &cnt, 1);
    if (cnt > 0) {
        data->point.x = x[0];
        data->point.y = y[0];
        data->state = LV_INDEV_STATE_PRESSED;
    } else {
        data->state = LV_INDEV_STATE_RELEASED;
    }
}

static void lv_tick_cb(void *arg)
{
    (void)arg;
    lv_tick_inc(5);
}

static void ui_task(void *arg)
{
    (void)arg;
    for (;;) {
        if (ui_lock()) {
            lv_timer_handler();
            ui_unlock();
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

/* ---------- 文案映射 ---------- */

static const char *cn_map(const char *key, const char *const map[][2], size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (strcmp(map[i][0], key) == 0) return map[i][1];
    }
    return key;
}

static const char *room_cn(const char *k)
{
    static const char *const m[][2] = {
        {"living_room", "客厅"}, {"bedroom", "卧室"}, {"kitchen", "厨房"},
        {"bathroom", "浴室"}, {"study", "书房"},
    };
    return cn_map(k, m, sizeof(m) / sizeof(m[0]));
}

static const char *device_cn(const char *k)
{
    static const char *const m[][2] = {
        {"light", "灯"}, {"ac", "空调"}, {"tv", "电视"},
        {"curtain", "窗帘"}, {"fan", "风扇"}, {"socket", "插座"},
    };
    return cn_map(k, m, sizeof(m) / sizeof(m[0]));
}

static const char *action_cn(const char *k)
{
    static const char *const m[][2] = {
        {"on", "开"}, {"off", "关"}, {"brighter", "调亮"}, {"dimmer", "调暗"},
        {"temp_up", "升温"}, {"temp_down", "降温"}, {"set_temp", "设定温度"},
        {"open", "打开"}, {"close", "关闭"}, {"speed_up", "加速"}, {"speed_down", "减速"},
        {"channel_up", "换台"}, {"channel_down", "回台"},
        {"volume_up", "音量加"}, {"volume_down", "音量减"},
    };
    return cn_map(k, m, sizeof(m) / sizeof(m[0]));
}

/* ---------- 事件 ---------- */

static void talk_btn_event_cb(lv_event_t *e)
{
    lv_event_code_t code = lv_event_get_code(e);
    if (code == LV_EVENT_PRESSED) {
        if (s_cb.talk_press) s_cb.talk_press();
    } else if (code == LV_EVENT_RELEASED || code == LV_EVENT_PRESS_LOST) {
        if (s_cb.talk_release) s_cb.talk_release();
    }
}

static void ok_btn_event_cb(lv_event_t *e)
{
    (void)e;
    if (s_cb.feedback) s_cb.feedback(true);
}

static void bad_btn_event_cb(lv_event_t *e)
{
    (void)e;
    if (s_cb.feedback) s_cb.feedback(false);
}

static void retry_btn_event_cb(lv_event_t *e)
{
    (void)e;
    if (s_cb.retry) s_cb.retry();
}

/* ---------- 界面搭建 ---------- */

static lv_obj_t *make_panel(lv_obj_t *parent)
{
    lv_obj_t *p = lv_obj_create(parent);
    lv_obj_set_size(p, BSP_LCD_H_RES, BSP_LCD_V_RES - 28);
    lv_obj_align(p, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_set_style_bg_opa(p, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(p, 0, 0);
    lv_obj_set_style_pad_all(p, 8, 0);
    lv_obj_clear_flag(p, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_flag(p, LV_OBJ_FLAG_HIDDEN);
    return p;
}

static lv_obj_t *make_label(lv_obj_t *parent, const char *text, lv_align_t align,
                            lv_coord_t y, lv_coord_t width)
{
    lv_obj_t *l = lv_label_create(parent);
    lv_label_set_text(l, text);
    if (width > 0) {
        lv_obj_set_width(l, width);
        lv_label_set_long_mode(l, LV_LABEL_LONG_WRAP);
    }
    lv_obj_align(l, align, 0, y);
    return l;
}

static void build_ui(void)
{
    lv_obj_t *scr = lv_scr_act();
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x101418), 0);
    lv_obj_set_style_text_color(scr, lv_color_hex(0xe8ecf1), 0);
    lv_obj_set_style_text_font(scr, &ui_font_16, 0);
    lv_obj_clear_flag(scr, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t *top = lv_obj_create(scr);
    lv_obj_set_size(top, BSP_LCD_H_RES, 28);
    lv_obj_align(top, LV_ALIGN_TOP_MID, 0, 0);
    lv_obj_set_style_bg_color(top, lv_color_hex(0x1f2630), 0);
    lv_obj_set_style_border_width(top, 0, 0);
    lv_obj_set_style_radius(top, 0, 0);
    lv_obj_set_style_pad_all(top, 4, 0);
    lv_obj_clear_flag(top, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_t *title = lv_label_create(top);
    lv_label_set_text(title, "方言语音终端");
    lv_obj_align(title, LV_ALIGN_LEFT_MID, 0, 0);
    s_lbl_wifi = lv_label_create(top);
    lv_label_set_text(s_lbl_wifi, "Wi-Fi: 未连接");
    lv_obj_set_style_text_color(s_lbl_wifi, lv_color_hex(0xe0a030), 0);
    lv_obj_align(s_lbl_wifi, LV_ALIGN_RIGHT_MID, 0, 0);

    for (int i = 0; i < P_COUNT; i++) s_panel[i] = make_panel(scr);

    /* IDLE */
    lv_obj_t *btn = lv_btn_create(s_panel[P_IDLE]);
    lv_obj_set_size(btn, 220, 84);
    lv_obj_align(btn, LV_ALIGN_CENTER, 0, -20);
    lv_obj_set_style_bg_color(btn, lv_color_hex(0x2d6cdf), 0);
    lv_obj_t *bl = lv_label_create(btn);
    lv_label_set_text(bl, "按住说话");
    lv_obj_center(bl);
    lv_obj_add_event_cb(btn, talk_btn_event_cb, LV_EVENT_ALL, NULL);
    lv_obj_t *hint = make_label(s_panel[P_IDLE],
                                "按住 BOOT 键或屏幕按钮说话，松开结束（最长 10 秒）",
                                LV_ALIGN_BOTTOM_MID, -6, 300);
    lv_obj_set_style_text_align(hint, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(hint, lv_color_hex(0x9aa4b2), 0);

    /* RECORDING */
    make_label(s_panel[P_REC], "正在聆听方言...", LV_ALIGN_TOP_MID, 16, 0);
    s_bar_level = lv_bar_create(s_panel[P_REC]);
    lv_obj_set_size(s_bar_level, 260, 16);
    lv_obj_align(s_bar_level, LV_ALIGN_CENTER, 0, 0);
    lv_bar_set_range(s_bar_level, 0, 100);
    lv_bar_set_value(s_bar_level, 0, LV_ANIM_OFF);
    s_lbl_elapsed = make_label(s_panel[P_REC], "0.0 秒", LV_ALIGN_BOTTOM_MID, -24, 0);

    /* UPLOADING */
    make_label(s_panel[P_UP], "上传并识别中...", LV_ALIGN_TOP_MID, 24, 0);
    lv_obj_t *sp = lv_spinner_create(s_panel[P_UP], 1000, 60);
    lv_obj_set_size(sp, 64, 64);
    lv_obj_align(sp, LV_ALIGN_CENTER, 0, 10);

    /* RESULT */
    s_lbl_reply = make_label(s_panel[P_RES], "", LV_ALIGN_TOP_MID, 0, 300);
    lv_obj_set_style_text_color(s_lbl_reply, lv_color_hex(0x7fd08a), 0);
    s_lbl_raw = make_label(s_panel[P_RES], "", LV_ALIGN_TOP_MID, 40, 300);
    s_lbl_norm = make_label(s_panel[P_RES], "", LV_ALIGN_TOP_MID, 64, 300);
    s_lbl_act = make_label(s_panel[P_RES], "", LV_ALIGN_TOP_MID, 88, 300);
    s_lbl_conf = make_label(s_panel[P_RES], "", LV_ALIGN_TOP_MID, 112, 300);
    lv_obj_t *bok = lv_btn_create(s_panel[P_RES]);
    lv_obj_set_size(bok, 140, 44);
    lv_obj_align(bok, LV_ALIGN_BOTTOM_MID, -75, -4);
    lv_obj_set_style_bg_color(bok, lv_color_hex(0x2e7d32), 0);
    lv_obj_t *bokl = lv_label_create(bok);
    lv_label_set_text(bokl, LV_SYMBOL_OK " 识别准确");
    lv_obj_center(bokl);
    lv_obj_add_event_cb(bok, ok_btn_event_cb, LV_EVENT_CLICKED, NULL);
    lv_obj_t *bbad = lv_btn_create(s_panel[P_RES]);
    lv_obj_set_size(bbad, 140, 44);
    lv_obj_align(bbad, LV_ALIGN_BOTTOM_MID, 75, -4);
    lv_obj_set_style_bg_color(bbad, lv_color_hex(0xc62828), 0);
    lv_obj_t *bbadl = lv_label_create(bbad);
    lv_label_set_text(bbadl, LV_SYMBOL_CLOSE " 识别有误");
    lv_obj_center(bbadl);
    lv_obj_add_event_cb(bbad, bad_btn_event_cb, LV_EVENT_CLICKED, NULL);

    /* ERROR */
    s_lbl_err = make_label(s_panel[P_ERR], "", LV_ALIGN_CENTER, -20, 300);
    lv_obj_set_style_text_align(s_lbl_err, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(s_lbl_err, lv_color_hex(0xe07050), 0);
    lv_obj_t *br = lv_btn_create(s_panel[P_ERR]);
    lv_obj_set_size(br, 160, 44);
    lv_obj_align(br, LV_ALIGN_BOTTOM_MID, 0, -8);
    lv_obj_t *brl = lv_label_create(br);
    lv_label_set_text(brl, "返回重试");
    lv_obj_center(brl);
    lv_obj_add_event_cb(br, retry_btn_event_cb, LV_EVENT_CLICKED, NULL);
}

static void show_panel(int idx)
{
    for (int i = 0; i < P_COUNT; i++) {
        if (i == idx) lv_obj_clear_flag(s_panel[i], LV_OBJ_FLAG_HIDDEN);
        else lv_obj_add_flag(s_panel[i], LV_OBJ_FLAG_HIDDEN);
    }
}

/* ---------- 对外接口 ---------- */

esp_err_t ui_manager_init(const ui_callbacks_t *cb)
{
    s_cb = *cb;
    s_lv_mutex = xSemaphoreCreateMutex();
    if (!s_lv_mutex) return ESP_ERR_NO_MEM;

    const spi_bus_config_t buscfg = {
        .mosi_io_num = BSP_LCD_MOSI_GPIO,
        .miso_io_num = -1,
        .sclk_io_num = BSP_LCD_SCLK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = LV_BUF_PIXELS * sizeof(lv_color_t),
    };
    ESP_RETURN_ON_ERROR(spi_bus_initialize(BSP_LCD_SPI_HOST, &buscfg, SPI_DMA_CH_AUTO),
                        TAG, "spi bus");
    const esp_lcd_panel_io_spi_config_t iocfg = {
        .cs_gpio_num = BSP_LCD_CS_GPIO,
        .dc_gpio_num = BSP_LCD_DC_GPIO,
        .spi_mode = BSP_LCD_SPI_MODE,
        .pclk_hz = BSP_LCD_PCLK_HZ,
        .trans_queue_depth = 10,
        .on_color_trans_done = lcd_flush_ready_cb,
        .user_ctx = NULL,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)BSP_LCD_SPI_HOST,
                                                 &iocfg, &s_io), TAG, "panel io");
    const esp_lcd_panel_dev_config_t pcfg = {
        .reset_gpio_num = -1,                  /* 复位由 TCA9557 P0 完成（board_bsp） */
        .color_space = ESP_LCD_COLOR_SPACE_RGB,
        .bits_per_pixel = 16,
        .flags = { .reset_active_high = 0 },
        .vendor_config = NULL,
    };
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_st7789(s_io, &pcfg, &s_panel), TAG, "st7789");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_reset(s_panel), TAG, "panel reset");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_init(s_panel), TAG, "panel init");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_invert_color(s_panel, true), TAG, "invert");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_swap_xy(s_panel, BSP_LCD_SWAP_XY), TAG, "swap");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_mirror(s_panel, BSP_LCD_MIRROR_X, BSP_LCD_MIRROR_Y),
                        TAG, "mirror");
    ESP_RETURN_ON_ERROR(esp_lcd_panel_disp_on_off(s_panel, true), TAG, "disp on");

    const esp_lcd_panel_io_i2c_config_t tiocfg = {
        .dev_addr = BSP_TOUCH_ADDR,
        .control_phase_bytes = 1,
        .dc_bit_offset = 0,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
        .flags = { .disable_control_phase = 1 },
        .scl_speed_hz = 400000,
    };
    esp_lcd_panel_io_handle_t tio = NULL;
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_io_i2c(board_bsp_i2c_bus(), &tiocfg, &tio),
                        TAG, "touch io");
    const esp_lcd_touch_config_t tpcfg = {
        .x_max = BSP_LCD_V_RES,                /* 触摸原生坐标系 240x320 */
        .y_max = BSP_LCD_H_RES,
        .rst_gpio_num = -1,                    /* 与 LCD 共用 TCA9557 复位 */
        .int_gpio_num = -1,
        .levels = { .reset = 0, .interrupt = 0 },
        .flags = { .swap_xy = 1, .mirror_x = 1, .mirror_y = 0 },
    };
    ESP_RETURN_ON_ERROR(esp_lcd_touch_new_i2c_ft5x06(tio, &tpcfg, &s_tp), TAG, "ft5x06");

    lv_init();
    lv_color_t *buf1 = heap_caps_malloc(LV_BUF_PIXELS * sizeof(lv_color_t), MALLOC_CAP_SPIRAM);
    lv_color_t *buf2 = heap_caps_malloc(LV_BUF_PIXELS * sizeof(lv_color_t), MALLOC_CAP_SPIRAM);
    if (!buf1 || !buf2) return ESP_ERR_NO_MEM;
    lv_disp_draw_buf_init(&s_draw_buf, buf1, buf2, LV_BUF_PIXELS);
    static lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);
    disp_drv.hor_res = BSP_LCD_H_RES;
    disp_drv.ver_res = BSP_LCD_V_RES;
    disp_drv.flush_cb = lcd_flush_cb;
    disp_drv.draw_buf = &s_draw_buf;
    s_disp = lv_disp_drv_register(&disp_drv);
    static lv_indev_drv_t indev_drv;
    lv_indev_drv_init(&indev_drv);
    indev_drv.type = LV_INDEV_TYPE_POINTER;
    indev_drv.read_cb = touch_read_cb;
    lv_indev_drv_register(&indev_drv);

    const esp_timer_create_args_t tick_args = {
        .callback = lv_tick_cb,
        .name = "lv_tick",
    };
    esp_timer_handle_t tick_timer = NULL;
    ESP_RETURN_ON_ERROR(esp_timer_create(&tick_args, &tick_timer), TAG, "tick timer");
    ESP_RETURN_ON_ERROR(esp_timer_start_periodic(tick_timer, 5000), TAG, "tick start");

    ui_font_16.fallback = &lv_font_montserrat_14;   /* ASCII 之外的内置符号 */
    if (ui_lock()) {
        build_ui();
        show_panel(P_IDLE);
        ui_unlock();
    }
    if (xTaskCreatePinnedToCore(ui_task, "ui_lvgl", 8192, NULL, 4, NULL, 0) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "ui ready %dx%d", BSP_LCD_H_RES, BSP_LCD_V_RES);
    return ESP_OK;
}

void ui_manager_show_idle(void)
{
    if (ui_lock()) { show_panel(P_IDLE); ui_unlock(); }
}

void ui_manager_show_recording(void)
{
    if (ui_lock()) {
        lv_bar_set_value(s_bar_level, 0, LV_ANIM_OFF);
        lv_label_set_text(s_lbl_elapsed, "0.0 秒");
        show_panel(P_REC);
        ui_unlock();
    }
}

void ui_manager_show_uploading(void)
{
    if (ui_lock()) { show_panel(P_UP); ui_unlock(); }
}

void ui_manager_show_result(const understand_result_t *res)
{
    if (!ui_lock()) return;
    bool executed = (strcmp(res->intent, "device_control") == 0) && res->confidence >= 0.7f;
    lv_label_set_text(s_lbl_reply, res->reply_text);
    lv_label_set_text_fmt(s_lbl_raw, "原文：%s", res->raw_text);
    lv_label_set_text_fmt(s_lbl_norm, "归一化：%s", res->normalized_text);
    lv_label_set_text_fmt(s_lbl_act, "动作：%s · %s · %s",
                          room_cn(res->location), device_cn(res->device), action_cn(res->action));
    lv_label_set_text_fmt(s_lbl_conf, "置信度：%.2f（%s）",
                          res->confidence, executed ? "已执行" : "仅展示");
    show_panel(P_RES);
    ui_unlock();
}

void ui_manager_show_error(const char *msg)
{
    if (ui_lock()) {
        lv_label_set_text(s_lbl_err, msg);
        show_panel(P_ERR);
        ui_unlock();
    }
}

void ui_manager_update_recording(uint8_t level, uint32_t elapsed_ms)
{
    if (ui_lock()) {
        lv_bar_set_value(s_bar_level, level, LV_ANIM_OFF);
        lv_label_set_text_fmt(s_lbl_elapsed, "%.1f 秒", elapsed_ms / 1000.0);
        ui_unlock();
    }
}

void ui_manager_set_wifi(bool connected)
{
    if (ui_lock()) {
        lv_label_set_text(s_lbl_wifi, connected ? "Wi-Fi: 已连接" : "Wi-Fi: 未连接");
        lv_obj_set_style_text_color(s_lbl_wifi,
                                    lv_color_hex(connected ? 0x7fd08a : 0xe0a030), 0);
        ui_unlock();
    }
}
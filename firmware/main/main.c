/* -*- coding: utf-8 -*-
 * main.c — 第一阶段「硬件自检」入口
 *
 * 本文件只做一件事：把板子点亮并证明每个器件都通了。
 * 它会在串口打印：
 *   1) Flash / PSRAM 实际容量（验证 N16R8 与 Octal PSRAM 是否真的生效）
 *   2) I2C 总线上扫到的全部地址（PCA9557=0x19 / ES8311=0x18 / ES7210=0x41 / FT6336=0x38）
 *   3) SD 卡挂载结果与容量
 *   4) 背光闪烁 2 次（目视确认 GPIO42 极性没搞反）
 *   5) 进入 200 ms 周期循环，打印用户按键状态与 PSRAM 剩余
 *
 * 第二阶段会被状态机版 main.c 整体替换（IDLE→RECORDING→UPLOADING→…），
 * 届时保留本文件为 tools/self_test.c 以便回归验板。
 */
#include <inttypes.h>
#include <stdio.h>

#include "board_bsp.h"
#include "board_pins.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

static const char *TAG = "selftest";

/* ---------------------------------------------------------------- I2C 扫描 */
static void i2c_scan(void)
{
    const i2c_port_t port = bsp_i2c_port();
    ESP_LOGI(TAG, "I2C scan on port %d (SDA=%d SCL=%d) ...",
             port, BSP_I2C_SDA_GPIO, BSP_I2C_SCL_GPIO);

    int found = 0;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
        i2c_master_stop(cmd);
        esp_err_t err = i2c_master_cmd_begin(port, cmd, pdMS_TO_TICKS(20));
        i2c_cmd_link_delete(cmd);
        if (err == ESP_OK) {
            const char *who = "unknown";
            switch (addr) {
            case BSP_PCA9557_ADDR: who = "PCA9557 (IO expander)"; break;
            case BSP_ES8311_ADDR:  who = "ES8311 (DAC/speaker)"; break;
            case BSP_ES7210_ADDR:  who = "ES7210 (ADC/mic)"; break;
            case BSP_TOUCH_ADDR:   who = "FT6336 (touch)"; break;
            default: break;
            }
            ESP_LOGI(TAG, "  found 0x%02X  %s", addr, who);
            found++;
        }
    }
    if (found == 0) {
        ESP_LOGE(TAG, "  no device ACK! 检查 SDA/SCL 是否接反、上拉是否缺失");
    } else {
        ESP_LOGI(TAG, "  total %d device(s)", found);
    }

    /* 逐项给出「应有但没扫到」的明确提示，避免用户误判 */
    const struct { uint8_t addr; const char *name; } expect[] = {
        {BSP_PCA9557_ADDR, "PCA9557"},
        {BSP_ES8311_ADDR,  "ES8311"},
        {BSP_ES7210_ADDR,  "ES7210"},
        {BSP_TOUCH_ADDR,   "FT6336"},
    };
    for (size_t i = 0; i < sizeof(expect) / sizeof(expect[0]); i++) {
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, (expect[i].addr << 1) | I2C_MASTER_WRITE, true);
        i2c_master_stop(cmd);
        esp_err_t err = i2c_master_cmd_begin(port, cmd, pdMS_TO_TICKS(20));
        i2c_cmd_link_delete(cmd);
        if (err != ESP_OK) {
            ESP_LOGW(TAG, "  MISSING %s @0x%02X —— 请核对器件型号、地址与 I2C 供电",
                     expect[i].name, expect[i].addr);
        }
    }
}

/* ------------------------------------------------------------ 硬件信息打印 */
static void print_hardware_info(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);

    uint32_t flash_size = 0;
    esp_flash_get_size(NULL, &flash_size);

    size_t psram_size = esp_psram_is_initialized() ? esp_psram_get_size() : 0;

    ESP_LOGI(TAG, "================ BOARD INFO ================");
    ESP_LOGI(TAG, "chip      : %s, %d core(s), rev v%d.%d",
             CONFIG_IDF_TARGET, chip.cores, chip.revision / 100, chip.revision % 100);
    ESP_LOGI(TAG, "flash     : %" PRIu32 " MB (expect 16 MB)", flash_size / (1024 * 1024));
    ESP_LOGI(TAG, "psram     : %u MB %s", (unsigned)(psram_size / (1024 * 1024)),
             psram_size ? "(Octal OK)" : "(NOT initialized! 检查 CONFIG_SPIRAM_MODE_OCT)");
    ESP_LOGI(TAG, "heap int  : free %u KB, largest %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
             (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) / 1024));
    ESP_LOGI(TAG, "heap psram: free %u KB, largest %u KB",
             (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
             (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "============================================");

    if (psram_size == 0) {
        ESP_LOGE(TAG, "PSRAM 未启用：录音缓冲将无法分配（本工程强制要求 8MB PSRAM）");
    }
}

/* --------------------------------------------------------------- 背光自检 */
static void backlight_blink(void)
{
    ESP_LOGI(TAG, "backlight blink x2 (GPIO%d, active LOW)", BSP_LCD_BL_GPIO);
    for (int i = 0; i < 2; i++) {
        bsp_lcd_backlight(false);   /* 熄灭 */
        vTaskDelay(pdMS_TO_TICKS(250));
        bsp_lcd_backlight(true);    /* 点亮 */
        vTaskDelay(pdMS_TO_TICKS(250));
    }
}

void app_main(void)
{
    /* NVS 是 Wi-Fi 与后续配置的前提，先初始化 */
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    print_hardware_info();

    err = bsp_board_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "bsp_board_init 失败: %s —— 自检中止", esp_err_to_name(err));
        return;
    }

    i2c_scan();
    backlight_blink();

    ESP_LOGI(TAG, "self-test entered loop; press BOOT (GPIO%d) to see level change",
             BSP_BTN_REC_GPIO);

    bool last = false;
    int tick = 0;
    while (true) {
        bool now = bsp_button_pressed();
        if (now != last) {
            ESP_LOGI(TAG, "button %s", now ? "PRESSED" : "released");
            last = now;
        }
        if (++tick % 25 == 0) {   /* 每 5 s 报一次内存，便于观察泄漏 */
            ESP_LOGI(TAG, "alive: psram free %u KB, int free %u KB, sd=%d",
                     (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
                     (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
                     (int)bsp_sd_is_mounted());
        }
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

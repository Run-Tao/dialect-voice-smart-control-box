/* -*- coding: utf-8 -*-
 * board_bsp.c — 立创·实战派 ESP32-S3 板级支持包实现
 *
 * 设计要点
 * --------
 * 1. **幂等**：所有 init 函数重复调用直接返回 ESP_OK，避免状态机反复初始化时出错。
 * 2. **I2C 用 legacy 驱动**（driver/i2c.h）：官方 es8311/es7210 组件的
 *    `es8311_create(i2c_port_t, addr)` 以「端口号」为参数，与 legacy 驱动配套；
 *    混用新 i2c_master 驱动会拿到 bus handle 而无法对接组件。
 * 3. **PCA9557 必须读-改-写**：LCD_CS / PA_EN / DVP_PWDN 共用 0x01 输出寄存器，
 *    直接整字节写入会误清其它位。
 * 4. **I2S 全双工**：TX(ES8311) 与 RX(ES7210) 由**同一次** i2s_new_channel 创建，
 *    从而硬件共享 BCLK/WS/MCLK，避免两套时钟打架导致爆音。
 */
#include "board_bsp.h"

#include <string.h>

#include "board_pins.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/sdmmc_host.h"
#include "driver/spi_master.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"

/* 官方 codec 组件（由 idf_component.yml 拉取） */
#include "es7210.h"
#include "es8311.h"

static const char *TAG = "bsp";

/* ==================================================================== */
/*                              内部状态                                 */
/* ==================================================================== */
static bool s_i2c_ready = false;
static bool s_pca_ready = false;
static bool s_i2s_ready = false;
static bool s_codec_ready = false;
static bool s_sd_ready = false;
static bool s_btn_ready = false;

static i2c_port_t s_i2c_port = BSP_I2C_PORT;

static i2s_chan_handle_t s_tx = NULL;
static i2s_chan_handle_t s_rx = NULL;

static es8311_handle_t s_es8311 = NULL;
static es7210_dev_handle_t s_es7210 = NULL;
static es7210_mic_gain_t s_es7210_gain;

static es7210_mic_gain_t es7210_gain_from_db(int db)
{
    if (db <= 0) {
        return ES7210_MIC_GAIN_0DB;
    }
    if (db <= 33) {
        return (es7210_mic_gain_t)((db + 1) / 3);
    }
    if (db <= 34) {
        return ES7210_MIC_GAIN_34_5DB;
    }
    if (db <= 36) {
        return ES7210_MIC_GAIN_36DB;
    }
    return ES7210_MIC_GAIN_37_5DB;
}

static sdmmc_card_t *s_sd_card = NULL;

/* PCA9557 输出寄存器的软件镜像，减少 I2C 读次数并保证位操作一致 */
static uint8_t s_pca_out = 0x00;

/* ==================================================================== */
/*                                I2C                                   */
/* ==================================================================== */
esp_err_t bsp_i2c_init(void)
{
    if (s_i2c_ready) {
        return ESP_OK;
    }

    const i2c_config_t cfg = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = BSP_I2C_SDA_GPIO,
        .scl_io_num = BSP_I2C_SCL_GPIO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = BSP_I2C_CLK_HZ,
    };

    ESP_RETURN_ON_ERROR(i2c_param_config(s_i2c_port, &cfg), TAG, "i2c_param_config");
    ESP_RETURN_ON_ERROR(i2c_driver_install(s_i2c_port, I2C_MODE_MASTER, 0, 0, 0),
                        TAG, "i2c_driver_install");
    s_i2c_ready = true;
    ESP_LOGI(TAG, "I2C ready: SDA=%d SCL=%d %d Hz",
             BSP_I2C_SDA_GPIO, BSP_I2C_SCL_GPIO, BSP_I2C_CLK_HZ);
    return ESP_OK;
}

i2c_port_t bsp_i2c_port(void)
{
    return s_i2c_port;
}

/* ---------------------------------------------------------- I2C 原子读写 */
static esp_err_t i2c_write_reg(uint8_t dev, uint8_t reg, uint8_t val)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    esp_err_t err = ESP_OK;
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (dev << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_write_byte(cmd, val, true);
    i2c_master_stop(cmd);
    err = i2c_master_cmd_begin(s_i2c_port, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return err;
}

static esp_err_t i2c_read_reg(uint8_t dev, uint8_t reg, uint8_t *val)
{
    if (val == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (dev << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (dev << 1) | I2C_MASTER_READ, true);
    i2c_master_read_byte(cmd, val, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t err = i2c_master_cmd_begin(s_i2c_port, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return err;
}

/* ==================================================================== */
/*                              PCA9557                                 */
/* ==================================================================== */
esp_err_t bsp_pca9557_init(void)
{
    if (s_pca_ready) {
        return ESP_OK;
    }
    ESP_RETURN_ON_ERROR(bsp_i2c_init(), TAG, "i2c for pca9557");

    /* 1) 极性不反转 */
    ESP_RETURN_ON_ERROR(i2c_write_reg(BSP_PCA9557_ADDR, BSP_PCA9557_REG_POLAR, 0x00),
                        TAG, "pca9557 polar");

    /* 2) 仅 IO0..IO2 设为输出（config 位 0 = 输出，1 = 输入）。
     *    官方板级代码写 0xF8，IO3..IO7 保持输入，避免驱动未知外设。 */
    ESP_RETURN_ON_ERROR(i2c_write_reg(BSP_PCA9557_ADDR, BSP_PCA9557_REG_CONFIG, 0xF8),
                        TAG, "pca9557 config");

    /* 3) 安全初值：
     *    LCD_CS   = 1（不选中）
     *    PA_EN    = 0（功放关闭，避免上电爆音）
     *    DVP_PWDN = 1（摄像头断电；本工程不用摄像头，保持关闭以省电）
     */
    s_pca_out = (uint8_t)((1u << BSP_PCA9557_IO_LCD_CS) |
                          (0u << BSP_PCA9557_IO_PA_EN) |
                          (1u << BSP_PCA9557_IO_DVP_PWDN));
    ESP_RETURN_ON_ERROR(i2c_write_reg(BSP_PCA9557_ADDR, BSP_PCA9557_REG_OUTPUT, s_pca_out),
                        TAG, "pca9557 output init");

    s_pca_ready = true;
    ESP_LOGI(TAG, "PCA9557 ready @0x%02X, out=0x%02X (LCD_CS=1 PA_EN=0 PWDN=1)",
             BSP_PCA9557_ADDR, s_pca_out);
    return ESP_OK;
}

esp_err_t bsp_pca9557_set(uint8_t io_num, bool level)
{
    if (io_num > BSP_PCA9557_IO_DVP_PWDN) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_RETURN_ON_ERROR(bsp_pca9557_init(), TAG, "pca9557 set");

    uint8_t next = s_pca_out;
    if (level) {
        next |= (uint8_t)(1u << io_num);
    } else {
        next &= (uint8_t)~(1u << io_num);
    }
    if (next == s_pca_out) {
        return ESP_OK;   /* 值未变，省一次 I2C */
    }
    ESP_RETURN_ON_ERROR(i2c_write_reg(BSP_PCA9557_ADDR, BSP_PCA9557_REG_OUTPUT, next),
                        TAG, "pca9557 write");
    s_pca_out = next;
    return ESP_OK;
}

esp_err_t bsp_pca9557_get(uint8_t io_num, bool *level)
{
    if (io_num > 7 || level == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    ESP_RETURN_ON_ERROR(bsp_pca9557_init(), TAG, "pca9557 get");
    uint8_t output = 0;
    ESP_RETURN_ON_ERROR(i2c_read_reg(BSP_PCA9557_ADDR, BSP_PCA9557_REG_OUTPUT,
                                    &output),
                        TAG, "pca9557 read");
    s_pca_out = output;
    *level = (output >> io_num) & 0x01;
    return ESP_OK;
}

esp_err_t bsp_lcd_cs(bool enable)
{
    /* ST7789 片选为低有效：enable=true → 拉低 */
    return bsp_pca9557_set(BSP_PCA9557_IO_LCD_CS, enable ? false : true);
}

esp_err_t bsp_pa_enable(bool enable)
{
    return bsp_pca9557_set(BSP_PCA9557_IO_PA_EN, enable);
}

/* ==================================================================== */
/*                                 I2S                                  */
/* ==================================================================== */
esp_err_t bsp_i2s_init(void)
{
    if (s_i2s_ready) {
        return ESP_OK;
    }

    /* 同一次 new_channel 同时拿 tx/rx → 硬件共享 MCLK/BCLK/WS */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(BSP_I2S_PORT, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 6;
    chan_cfg.dma_frame_num = 240;   /* 240 帧 @16k ≈ 15 ms，兼顾延迟与抖动 */
    chan_cfg.auto_clear = true;
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, &s_tx, &s_rx), TAG, "i2s_new_channel");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(BSP_I2S_SAMPLE_RATE),
        /* 槽位用立体声：ES7210 双麦 / ES8311 均按双声道帧组织；
         * 单声道提取在 audio_recorder 里按通道取值，不改硬件协议。 */
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(BSP_I2S_BIT_WIDTH,
                                                       I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = BSP_I2S_MCLK_GPIO,
            .bclk = BSP_I2S_BCLK_GPIO,
            .ws = BSP_I2S_WS_GPIO,
            .dout = BSP_I2S_DOUT_GPIO,   /* ESP32-S3 → ES8311 */
            .din = BSP_I2S_SDIN_GPIO,    /* ES7210 → ESP32-S3 */
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    std_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_tx, &std_cfg), TAG, "i2s tx init");
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_rx, &std_cfg), TAG, "i2s rx init");

    s_i2s_ready = true;
    ESP_LOGI(TAG, "I2S full-duplex ready: MCLK=%d BCLK=%d WS=%d DIN=%d DOUT=%d @%d Hz",
             BSP_I2S_MCLK_GPIO, BSP_I2S_BCLK_GPIO, BSP_I2S_WS_GPIO,
             BSP_I2S_SDIN_GPIO, BSP_I2S_DOUT_GPIO, BSP_I2S_SAMPLE_RATE);
    return ESP_OK;
}

i2s_chan_handle_t bsp_i2s_tx_handle(void)
{
    return s_tx;
}

i2s_chan_handle_t bsp_i2s_rx_handle(void)
{
    return s_rx;
}

esp_err_t bsp_i2s_start(void)
{
    ESP_RETURN_ON_FALSE(s_i2s_ready, ESP_ERR_INVALID_STATE, TAG, "i2s not init");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_rx), TAG, "i2s rx enable");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_tx), TAG, "i2s tx enable");
    return ESP_OK;
}

esp_err_t bsp_i2s_stop(void)
{
    if (!s_i2s_ready) {
        return ESP_OK;
    }
    /* 先停 TX 再停 RX：避免 RX 停下时 TX 仍在推空帧产生爆音 */
    if (s_tx) {
        i2s_channel_disable(s_tx);
    }
    if (s_rx) {
        i2s_channel_disable(s_rx);
    }
    return ESP_OK;
}

/* ==================================================================== */
/*                            ES7210 / ES8311                           */
/* ==================================================================== */
static esp_err_t es8311_codec_init(void)
{
    s_es8311 = es8311_create(s_i2c_port, BSP_ES8311_ADDR);
    ESP_RETURN_ON_FALSE(s_es8311 != NULL, ESP_FAIL, TAG, "es8311_create");

    const es8311_clock_config_t clk = {
        .mclk_inverted = false,
        .sclk_inverted = false,
        .mclk_from_mclk_pin = true,
        .mclk_frequency = BSP_I2S_SAMPLE_RATE * BSP_I2S_MCLK_MULTIPLE,
        .sample_frequency = BSP_I2S_SAMPLE_RATE,
    };
    /* 输入/输出分辨率均 16 bit；ES8311 的 ADC 本工程不用，仍按 16 bit 配 */
    ESP_RETURN_ON_ERROR(es8311_init(s_es8311, &clk, ES8311_RESOLUTION_16, ES8311_RESOLUTION_16),
                        TAG, "es8311_init");
    ESP_RETURN_ON_ERROR(es8311_sample_frequency_config(
                            s_es8311,
                            BSP_I2S_SAMPLE_RATE * BSP_I2S_MCLK_MULTIPLE,
                            BSP_I2S_SAMPLE_RATE),
                        TAG, "es8311_sample_frequency_config");
    /* 关闭 ES8311 自带 ADC（麦克风走 ES7210，避免双 ADC 争用 I2S 槽位） */
    ESP_RETURN_ON_ERROR(es8311_microphone_config(s_es8311, false),
                        TAG, "es8311_microphone_config");
    ESP_RETURN_ON_ERROR(es8311_voice_volume_set(s_es8311, 70, NULL),
                        TAG, "es8311_voice_volume_set");
    ESP_LOGI(TAG, "ES8311 ready @0x%02X (playback)", BSP_ES8311_ADDR);
    return ESP_OK;
}

static esp_err_t es7210_codec_init(void)
{
    s_es7210_gain = es7210_gain_from_db(BSP_ES7210_MIC_GAIN_DB);

    const es7210_i2c_config_t i2c_cfg = {
        .i2c_port = s_i2c_port,
        .i2c_addr = BSP_ES7210_ADDR,
    };
    ESP_RETURN_ON_ERROR(es7210_new_codec(&i2c_cfg, &s_es7210), TAG, "es7210_new_codec");
    ESP_RETURN_ON_FALSE(s_es7210 != NULL, ESP_FAIL, TAG, "es7210 handle null");

    /*
     * ES7210 1.0.1~1 由 config_codec() 一次配置采样率、格式、位宽、
     * MIC 偏置与 PGA；它没有独立的 config_mics/set_gain 公共接口。
     * 标准 I2S、16 bit、非 TDM 模式下，I2S 前两个槽位对应 ADC1/ADC2。
     */
    const es7210_codec_config_t codec_cfg = {
        .sample_rate_hz = BSP_I2S_SAMPLE_RATE,
        .mclk_ratio = BSP_I2S_MCLK_MULTIPLE,
        .i2s_format = ES7210_I2S_FMT_I2S,
        .bit_width = ES7210_I2S_BITS_16B,
        .mic_bias = ES7210_MIC_BIAS_2V87,
        .mic_gain = s_es7210_gain,
        .flags.tdm_enable = 0,
    };
    ESP_RETURN_ON_ERROR(es7210_config_codec(s_es7210, &codec_cfg), TAG, "es7210_config_codec");
    ESP_RETURN_ON_ERROR(es7210_config_volume(s_es7210, 0), TAG, "es7210_config_volume");

    ESP_LOGI(TAG, "ES7210 ready @0x%02X (I2S 16-bit, non-TDM, MIC gain step=%d)",
             BSP_ES7210_ADDR, (int)s_es7210_gain);
    return ESP_OK;
}

esp_err_t bsp_codec_init(void)
{
    if (s_codec_ready) {
        return ESP_OK;
    }
    /* 依赖顺序：I2C → I2S（MCLK 必须先稳定输出，codec 才能锁定采样率） */
    ESP_RETURN_ON_ERROR(bsp_i2c_init(), TAG, "i2c for codec");
    ESP_RETURN_ON_ERROR(bsp_i2s_init(), TAG, "i2s for codec");

    /* 先开 RX（ADC 需要 MCLK）；TX 也一并使能以保证共用时钟连续 */
    ESP_RETURN_ON_ERROR(bsp_i2s_start(), TAG, "i2s start for codec");

    esp_err_t err = es8311_codec_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "ES8311 init failed (0x%02X): %s —— 播放将不可用",
                 BSP_ES8311_ADDR, esp_err_to_name(err));
        return err;
    }
    err = es7210_codec_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "ES7210 init failed (0x%02X): %s —— 请核对地址/接线",
                 BSP_ES7210_ADDR, esp_err_to_name(err));
        return err;
    }

    s_codec_ready = true;
    return ESP_OK;
}

esp_err_t bsp_codec_set_volume(int volume)
{
    ESP_RETURN_ON_FALSE(s_codec_ready, ESP_ERR_INVALID_STATE, TAG, "codec not ready");
    if (volume < 0) {
        volume = 0;
    }
    if (volume > 100) {
        volume = 100;
    }
    return es8311_voice_volume_set(s_es8311, volume, NULL);
}

esp_err_t bsp_codec_set_mic_gain(int db)
{
    ESP_RETURN_ON_FALSE(s_codec_ready, ESP_ERR_INVALID_STATE, TAG, "codec not ready");

    /* 组件只暴露 config_codec()；把 dB 映射到最近的硬件枚举后整体重配。 */
    s_es7210_gain = es7210_gain_from_db(db);

    const es7210_codec_config_t codec_cfg = {
        .sample_rate_hz = BSP_I2S_SAMPLE_RATE,
        .mclk_ratio = BSP_I2S_MCLK_MULTIPLE,
        .i2s_format = ES7210_I2S_FMT_I2S,
        .bit_width = ES7210_I2S_BITS_16B,
        .mic_bias = ES7210_MIC_BIAS_2V87,
        .mic_gain = s_es7210_gain,
        .flags.tdm_enable = 0,
    };
    return es7210_config_codec(s_es7210, &codec_cfg);
}

/* ==================================================================== */
/*                              MicroSD                                 */
/* ==================================================================== */
esp_err_t bsp_sd_init(void)
{
    if (s_sd_ready) {
        return ESP_OK;
    }

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.flags = SDMMC_HOST_FLAG_1BIT;
    host.max_freq_khz = SDMMC_FREQ_DEFAULT;

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.clk = BSP_SD_CLK_GPIO;
    slot.cmd = BSP_SD_CMD_GPIO;
    slot.d0 = BSP_SD_D0_GPIO;
    slot.width = BSP_SD_WIDTH;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    const esp_vfs_fat_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,
        .max_files = 8,
        .allocation_unit_size = 16 * 1024,
    };

    esp_err_t err = esp_vfs_fat_sdmmc_mount(BSP_SD_MOUNT_POINT, &host, &slot,
                                            &mount_cfg, &s_sd_card);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s（检查卡是否插入/是否为 FAT32）",
                 esp_err_to_name(err));
        return err;
    }

    s_sd_ready = true;
    ESP_LOGI(TAG, "SD mounted at %s, %llu MB", BSP_SD_MOUNT_POINT,
             ((uint64_t)s_sd_card->csd.capacity) * s_sd_card->csd.sector_size / (1024 * 1024));
    return ESP_OK;
}

bool bsp_sd_is_mounted(void)
{
    return s_sd_ready;
}

esp_err_t bsp_sd_deinit(void)
{
    if (!s_sd_ready) {
        return ESP_OK;
    }
    esp_err_t err = esp_vfs_fat_sdcard_unmount(BSP_SD_MOUNT_POINT, s_sd_card);
    if (err == ESP_OK) {
        s_sd_card = NULL;
        s_sd_ready = false;
    }
    return err;
}

/* ==================================================================== */
/*                            背光 / 按键                                */
/* ==================================================================== */
esp_err_t bsp_lcd_backlight(bool on)
{
    static bool inited = false;
    if (!inited) {
        gpio_config_t io = {
            .pin_bit_mask = (1ULL << BSP_LCD_BL_GPIO),
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        ESP_RETURN_ON_ERROR(gpio_config(&io), TAG, "backlight gpio_config");
        inited = true;
    }
    /* ⚠ GPIO42 低电平点亮 */
    ESP_RETURN_ON_ERROR(gpio_set_level(BSP_LCD_BL_GPIO,
                                       on ? BSP_LCD_BL_LEVEL_ON : BSP_LCD_BL_LEVEL_OFF),
                        TAG, "backlight set_level");
    return ESP_OK;
}

esp_err_t bsp_button_init(void)
{
    if (s_btn_ready) {
        return ESP_OK;
    }
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << BSP_BTN_REC_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,   /* 采用轮询 + 软件消抖，避免 ISR 里做业务 */
    };
    ESP_RETURN_ON_ERROR(gpio_config(&io), TAG, "button gpio_config");
    s_btn_ready = true;
    ESP_LOGI(TAG, "user button ready: GPIO%d (active low)", BSP_BTN_REC_GPIO);
    return ESP_OK;
}

bool bsp_button_pressed(void)
{
    if (!s_btn_ready) {
        return false;
    }
    return gpio_get_level(BSP_BTN_REC_GPIO) == BSP_BTN_ACTIVE_LEVEL;
}

/* ==================================================================== */
/*                              总初始化                                 */
/* ==================================================================== */
esp_err_t bsp_board_init(void)
{
    esp_err_t err;

    /* 1. I2C 总线（后续所有 I2C 外设的前提） */
    err = bsp_i2c_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "[1/6] I2C 失败: %s", esp_err_to_name(err));
        return err;
    }

    /* 2. PCA9557：先把 LCD_CS 置高、功放关闭，避免上电异常 */
    err = bsp_pca9557_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "[2/6] PCA9557 失败: %s（I2C 上应能在 0x19 扫到）",
                 esp_err_to_name(err));
        return err;
    }

    /* 3. I2S 全双工 */
    err = bsp_i2s_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "[3/6] I2S 失败: %s", esp_err_to_name(err));
        return err;
    }

    /* 4. 编解码器（ES8311 播放 + ES7210 录音） */
    err = bsp_codec_init();
    if (err != ESP_OK) {
        /* codec 失败不阻断整机：至少屏幕/网络可用，便于现场诊断 */
        ESP_LOGW(TAG, "[4/6] codec 初始化异常（继续启动）: %s", esp_err_to_name(err));
    }

    /* 5. SD 卡（失败不阻断：允许无卡进入「仅上传不归档」模式） */
    err = bsp_sd_init();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "[5/6] SD 未挂载（继续启动，录音将不落盘）");
    }

    /* 6. 背光 + 按键 */
    err = bsp_lcd_backlight(true);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "[6/6] 背光失败: %s", esp_err_to_name(err));
    }
    err = bsp_button_init();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "[6/6] 按键失败: %s", esp_err_to_name(err));
    }

    ESP_LOGI(TAG, "board init done: i2c=%d pca=%d i2s=%d codec=%d sd=%d",
             s_i2c_ready, s_pca_ready, s_i2s_ready, s_codec_ready, s_sd_ready);
    return ESP_OK;
}

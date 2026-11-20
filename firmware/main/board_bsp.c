/**
 * board_bsp.c — 立创·实战派 ESP32-S3 V1.0 板级支持包实现
 * 全双工音频拓扑与 esp_codec_dev 用法对齐 xiaozhi-esp32 在本板上的量产实现
 * (main/audio/codecs/box_audio_codec.cc)：TX=std 立体声、RX=TDM 4 槽、同端口共时钟。
 */
#include "board_bsp.h"
#include "board_pins.h"

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_check.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "driver/i2s_tdm.h"
#include "driver/ledc.h"
#include "driver/sdmmc_host.h"

#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"

static const char *TAG = "board_bsp";

static i2c_master_bus_handle_t s_i2c_bus;
static i2c_master_dev_handle_t s_tca_dev;
static SemaphoreHandle_t s_tca_mutex;

static i2s_chan_handle_t s_i2s_tx;
static i2s_chan_handle_t s_i2s_rx;

static const audio_codec_data_if_t *s_data_if;
static const audio_codec_ctrl_if_t *s_in_ctrl_if;
static const audio_codec_ctrl_if_t *s_out_ctrl_if;
static const audio_codec_if_t *s_in_codec_if;
static const audio_codec_if_t *s_out_codec_if;
static const audio_codec_gpio_if_t *s_gpio_if;
static esp_codec_dev_handle_t s_in_dev;
static esp_codec_dev_handle_t s_out_dev;
static bool s_in_open;
static bool s_out_open;

static sdmmc_card_t *s_sd_card;
static bool s_sd_ready;
static bool s_exp_configured;

/* ---------------- TCA9557 ---------------- */

static esp_err_t tca_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = { reg, val };
    return i2c_master_transmit(s_tca_dev, buf, sizeof(buf), pdMS_TO_TICKS(100));
}

static esp_err_t tca_read_reg(uint8_t reg, uint8_t *val)
{
    return i2c_master_transmit_receive(s_tca_dev, &reg, 1, val, 1, pdMS_TO_TICKS(100));
}

static esp_err_t tca_set_bit(uint8_t bit, uint8_t level)
{
    esp_err_t ret;
    uint8_t out;

    xSemaphoreTake(s_tca_mutex, portMAX_DELAY);
    ret = tca_read_reg(BSP_TCA9557_REG_OUT, &out);
    if (ret == ESP_OK) {
        out = (out & ~(1u << bit)) | ((level & 1u) << bit);
        ret = tca_write_reg(BSP_TCA9557_REG_OUT, out);
    }
    xSemaphoreGive(s_tca_mutex);
    return ret;
}

esp_err_t board_bsp_pa_enable(bool enable)
{
    ESP_RETURN_ON_FALSE(s_tca_dev, ESP_ERR_INVALID_STATE, TAG, "bsp not init");
    return tca_set_bit(BSP_TCA9557_BIT_PA_CTRL, enable ? 1 : 0);
}

esp_err_t board_bsp_lcd_reset(void)
{
    ESP_RETURN_ON_FALSE(s_tca_dev, ESP_ERR_INVALID_STATE, TAG, "bsp not init");
    ESP_RETURN_ON_ERROR(tca_set_bit(BSP_TCA9557_BIT_LCD_RST, 0), TAG, "lcd reset assert");
    vTaskDelay(pdMS_TO_TICKS(20));
    ESP_RETURN_ON_ERROR(tca_set_bit(BSP_TCA9557_BIT_LCD_RST, 1), TAG, "lcd reset release");
    vTaskDelay(pdMS_TO_TICKS(120));
    return ESP_OK;
}

/* ---------------- 背光 ---------------- */

esp_err_t board_bsp_backlight_set(uint8_t percent)
{
    if (percent > 100) {
        percent = 100;
    }
    uint32_t duty = (1023u * percent) / 100u;
    if (BSP_LCD_BL_INVERT) {
        duty = 1023u - duty;
    }
    ESP_RETURN_ON_ERROR(ledc_set_duty(LEDC_LOW_SPEED_MODE, BSP_LCD_BL_LEDC_CHAN, duty), TAG, "bl duty");
    return ledc_update_duty(LEDC_LOW_SPEED_MODE, BSP_LCD_BL_LEDC_CHAN);
}

static esp_err_t backlight_init(void)
{
    const ledc_timer_config_t timer_cfg = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_10_BIT,
        .timer_num = BSP_LCD_BL_LEDC_TIMER,
        .freq_hz = BSP_LCD_BL_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_RETURN_ON_ERROR(ledc_timer_config(&timer_cfg), TAG, "bl timer");

    const ledc_channel_config_t chan_cfg = {
        .gpio_num = BSP_LCD_BL_GPIO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = BSP_LCD_BL_LEDC_CHAN,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = BSP_LCD_BL_LEDC_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_RETURN_ON_ERROR(ledc_channel_config(&chan_cfg), TAG, "bl channel");
    return board_bsp_backlight_set(BSP_LCD_BL_DEFAULT_PCT);
}

/* ---------------- I2S 全双工 + Codec ---------------- */

static esp_err_t audio_i2s_init(void)
{
    const i2s_chan_config_t chan_cfg = {
        .id = BSP_I2S_PORT,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = BSP_I2S_DMA_DESC_NUM,
        .dma_frame_num = BSP_I2S_DMA_FRAME_NUM,
        .auto_clear_after_cb = true,
        .auto_clear_before_cb = false,
        .intr_priority = 0,
    };
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, &s_i2s_tx, &s_i2s_rx), TAG, "i2s channel");

    /* TX：std 立体声 16bit，ES8311 消费左槽；与 RX 共享 MCLK/BCLK/WS */
    const i2s_std_config_t std_cfg = {
        .clk_cfg = {
            .sample_rate_hz = BSP_AUDIO_SAMPLE_RATE,
            .clk_src = I2S_CLK_SRC_DEFAULT,
            .ext_clk_freq_hz = 0,
            .mclk_multiple = BSP_AUDIO_MCLK_MULT,
        },
        .slot_cfg = {
            .data_bit_width = I2S_DATA_BIT_WIDTH_16BIT,
            .slot_bit_width = I2S_SLOT_BIT_WIDTH_AUTO,
            .slot_mode = I2S_SLOT_MODE_STEREO,
            .slot_mask = I2S_STD_SLOT_BOTH,
            .ws_width = I2S_DATA_BIT_WIDTH_16BIT,
            .ws_pol = false,
            .bit_shift = true,
            .left_align = true,
            .big_endian = false,
            .bit_order_lsb = false,
        },
        .gpio_cfg = {
            .mclk = BSP_I2S_MCLK_GPIO,
            .bclk = BSP_I2S_BCLK_GPIO,
            .ws = BSP_I2S_WS_GPIO,
            .dout = BSP_I2S_DOUT_GPIO,
            .din = I2S_GPIO_UNUSED,
            .invert_flags = { false, false, false },
        },
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(s_i2s_tx, &std_cfg), TAG, "i2s std");

    /* RX：TDM 4 槽 16bit（槽序 MIC1,MIC3,MIC2,MIC4），主麦在 slot0 */
    const i2s_tdm_config_t tdm_cfg = {
        .clk_cfg = {
            .sample_rate_hz = BSP_AUDIO_SAMPLE_RATE,
            .clk_src = I2S_CLK_SRC_DEFAULT,
            .ext_clk_freq_hz = 0,
            .mclk_multiple = BSP_AUDIO_MCLK_MULT,
            .bclk_div = 8,
        },
        .slot_cfg = {
            .data_bit_width = I2S_DATA_BIT_WIDTH_16BIT,
            .slot_bit_width = I2S_SLOT_BIT_WIDTH_AUTO,
            .slot_mode = I2S_SLOT_MODE_STEREO,
            .slot_mask = (i2s_tdm_slot_mask_t)(I2S_TDM_SLOT0 | I2S_TDM_SLOT1 |
                                               I2S_TDM_SLOT2 | I2S_TDM_SLOT3),
            .ws_width = I2S_TDM_AUTO_WS_WIDTH,
            .ws_pol = false,
            .bit_shift = true,
            .left_align = false,
            .big_endian = false,
            .bit_order_lsb = false,
            .skip_mask = false,
            .total_slot = I2S_TDM_AUTO_SLOT_NUM,
        },
        .gpio_cfg = {
            .mclk = BSP_I2S_MCLK_GPIO,
            .bclk = BSP_I2S_BCLK_GPIO,
            .ws = BSP_I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din = BSP_I2S_DIN_GPIO,
            .invert_flags = { false, false, false },
        },
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_tdm_mode(s_i2s_rx, &tdm_cfg), TAG, "i2s tdm");

    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_i2s_tx), TAG, "i2s tx enable");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(s_i2s_rx), TAG, "i2s rx enable");
    return ESP_OK;
}

static esp_err_t audio_codec_init(void)
{
    const audio_codec_i2s_cfg_t i2s_cfg = {
        .port = BSP_I2S_PORT,
        .rx_handle = s_i2s_rx,
        .tx_handle = s_i2s_tx,
    };
    s_data_if = audio_codec_new_i2s_data(&i2s_cfg);
    ESP_RETURN_ON_FALSE(s_data_if, ESP_FAIL, TAG, "codec data if");

    audio_codec_i2c_cfg_t i2c_cfg = {
        .port = BSP_I2C_PORT,
        .addr = BSP_ES8311_ADDR,
        .bus_handle = s_i2c_bus,
    };
    s_out_ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    ESP_RETURN_ON_FALSE(s_out_ctrl_if, ESP_FAIL, TAG, "es8311 ctrl if");

    s_gpio_if = audio_codec_new_gpio();
    ESP_RETURN_ON_FALSE(s_gpio_if, ESP_FAIL, TAG, "codec gpio if");

    es8311_codec_cfg_t es8311_cfg = { 0 };
    es8311_cfg.ctrl_if = s_out_ctrl_if;
    es8311_cfg.gpio_if = s_gpio_if;
    es8311_cfg.codec_mode = ESP_CODEC_DEV_WORK_MODE_DAC;
    es8311_cfg.pa_pin = GPIO_NUM_NC;   /* 功放在 TCA9557 P1，不走 codec GPIO */
    es8311_cfg.use_mclk = true;
    es8311_cfg.hw_gain.pa_voltage = 5.0f;
    es8311_cfg.hw_gain.codec_dac_voltage = 3.3f;
    s_out_codec_if = es8311_codec_new(&es8311_cfg);
    ESP_RETURN_ON_FALSE(s_out_codec_if, ESP_FAIL, TAG, "es8311 codec");

    esp_codec_dev_cfg_t dev_cfg = {
        .dev_type = ESP_CODEC_DEV_TYPE_OUT,
        .codec_if = s_out_codec_if,
        .data_if = s_data_if,
    };
    s_out_dev = esp_codec_dev_new(&dev_cfg);
    ESP_RETURN_ON_FALSE(s_out_dev, ESP_FAIL, TAG, "out dev");

    i2c_cfg.addr = BSP_ES7210_ADDR;
    s_in_ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    ESP_RETURN_ON_FALSE(s_in_ctrl_if, ESP_FAIL, TAG, "es7210 ctrl if");

    es7210_codec_cfg_t es7210_cfg = { 0 };
    es7210_cfg.ctrl_if = s_in_ctrl_if;
    es7210_cfg.mic_selected = ES7210_SEL_MIC1 | ES7210_SEL_MIC2 |
                              ES7210_SEL_MIC3 | ES7210_SEL_MIC4;
    s_in_codec_if = es7210_codec_new(&es7210_cfg);
    ESP_RETURN_ON_FALSE(s_in_codec_if, ESP_FAIL, TAG, "es7210 codec");

    dev_cfg.dev_type = ESP_CODEC_DEV_TYPE_IN;
    dev_cfg.codec_if = s_in_codec_if;
    s_in_dev = esp_codec_dev_new(&dev_cfg);
    ESP_RETURN_ON_FALSE(s_in_dev, ESP_FAIL, TAG, "in dev");
    return ESP_OK;
}

esp_err_t board_bsp_audio_open_input(void)
{
    ESP_RETURN_ON_FALSE(s_in_dev, ESP_ERR_INVALID_STATE, TAG, "bsp not init");
    if (s_in_open) {
        return ESP_OK;
    }
    /* channel=4 对应 TDM 4 槽；channel_mask=bit0 只取主麦 MIC1 => 单声道输出 */
    const esp_codec_dev_sample_info_t fs = {
        .bits_per_sample = 16,
        .channel = 4,
        .channel_mask = ESP_CODEC_DEV_MAKE_CHANNEL_MASK(0),
        .sample_rate = BSP_AUDIO_SAMPLE_RATE,
        .mclk_multiple = 0,
    };
    ESP_RETURN_ON_ERROR(esp_codec_dev_open(s_in_dev, &fs), TAG, "in open");
    ESP_RETURN_ON_ERROR(esp_codec_dev_set_in_channel_gain(
                            s_in_dev, ESP_CODEC_DEV_MAKE_CHANNEL_MASK(0), BSP_MIC_GAIN_DB),
                        TAG, "mic gain");
    s_in_open = true;
    return ESP_OK;
}

esp_err_t board_bsp_audio_close_input(void)
{
    if (!s_in_open) {
        return ESP_OK;
    }
    s_in_open = false;
    return esp_codec_dev_close(s_in_dev);
}

esp_err_t board_bsp_audio_read_mono(int16_t *pcm, size_t samples)
{
    ESP_RETURN_ON_FALSE(s_in_open, ESP_ERR_INVALID_STATE, TAG, "input not open");
    return esp_codec_dev_read(s_in_dev, pcm, (int)(samples * sizeof(int16_t)));
}

esp_err_t board_bsp_audio_open_output(void)
{
    ESP_RETURN_ON_FALSE(s_out_dev, ESP_ERR_INVALID_STATE, TAG, "bsp not init");
    if (s_out_open) {
        return ESP_OK;
    }
    const esp_codec_dev_sample_info_t fs = {
        .bits_per_sample = 16,
        .channel = 1,
        .channel_mask = 0,
        .sample_rate = BSP_AUDIO_SAMPLE_RATE,
        .mclk_multiple = 0,
    };
    ESP_RETURN_ON_ERROR(esp_codec_dev_open(s_out_dev, &fs), TAG, "out open");
    ESP_RETURN_ON_ERROR(esp_codec_dev_set_out_vol(s_out_dev, (float)BSP_SPK_VOL_DEFAULT), TAG, "out vol");
    ESP_RETURN_ON_ERROR(board_bsp_pa_enable(true), TAG, "pa on");
    s_out_open = true;
    return ESP_OK;
}

esp_err_t board_bsp_audio_close_output(void)
{
    if (!s_out_open) {
        return ESP_OK;
    }
    s_out_open = false;
    esp_err_t ret = esp_codec_dev_close(s_out_dev);
    board_bsp_pa_enable(false);
    return ret;
}

esp_err_t board_bsp_audio_write_mono(const int16_t *pcm, size_t samples)
{
    ESP_RETURN_ON_FALSE(s_out_open, ESP_ERR_INVALID_STATE, TAG, "output not open");
    return esp_codec_dev_write(s_out_dev, (void *)pcm, (int)(samples * sizeof(int16_t)));
}

esp_err_t board_bsp_audio_set_volume(uint8_t percent)
{
    ESP_RETURN_ON_FALSE(s_out_dev, ESP_ERR_INVALID_STATE, TAG, "bsp not init");
    if (percent > 100) {
        percent = 100;
    }
    return esp_codec_dev_set_out_vol(s_out_dev, (float)percent);
}

/* ---------------- MicroSD ---------------- */

esp_err_t board_bsp_sd_mount(void)
{
    if (s_sd_ready) {
        return ESP_OK;
    }
    const esp_vfs_fat_sdmmc_mount_config_t mount_cfg = {
        .format_if_mount_failed = false,   /* 绝不自动格式化用户卡 */
        .max_files = BSP_SD_MAX_FILES,
        .allocation_unit_size = BSP_SD_ALLOC_UNIT,
    };
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = BSP_SD_BUS_WIDTH;
    slot.clk = BSP_SD_CLK_GPIO;
    slot.cmd = BSP_SD_CMD_GPIO;
    slot.d0 = BSP_SD_D0_GPIO;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    esp_err_t ret = esp_vfs_fat_sdmmc_mount(BSP_SD_MOUNT_POINT, &host, &slot, &mount_cfg, &s_sd_card);
    if (ret != ESP_OK) {
        s_sd_card = NULL;
        s_sd_ready = false;
        ESP_LOGE(TAG, "SD mount failed: %s", esp_err_to_name(ret));
        return ret;
    }
    s_sd_ready = true;
    sdmmc_card_print_info(stdout, s_sd_card);
    return ESP_OK;
}

esp_err_t board_bsp_sd_unmount(void)
{
    if (!s_sd_ready) {
        return ESP_OK;
    }
    esp_err_t ret = esp_vfs_fat_sdcard_unmount(BSP_SD_MOUNT_POINT, s_sd_card);
    s_sd_card = NULL;
    s_sd_ready = false;
    return ret;
}

bool board_bsp_sd_ready(void)
{
    return s_sd_ready;
}

const sdmmc_card_t *board_bsp_sd_card(void)
{
    return s_sd_card;
}

/* ---------------- 外扩口 ---------------- */

esp_err_t board_bsp_exp_set(uint8_t index, bool level)
{
    gpio_num_t gpio = (index == 0) ? BSP_EXP1_GPIO : BSP_EXP2_GPIO;
    ESP_RETURN_ON_FALSE(index < 2, ESP_ERR_INVALID_ARG, TAG, "exp index");
    if (!s_exp_configured) {
        const gpio_config_t cfg = {
            .pin_bit_mask = (1ULL << BSP_EXP1_GPIO) | (1ULL << BSP_EXP2_GPIO),
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLUP_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        ESP_RETURN_ON_ERROR(gpio_config(&cfg), TAG, "exp gpio");
        s_exp_configured = true;
    }
    return gpio_set_level(gpio, level ? 1 : 0);
}

/* ---------------- 总初始化 ---------------- */

i2c_master_bus_handle_t board_bsp_i2c_bus(void)
{
    return s_i2c_bus;
}

esp_err_t board_bsp_init(void)
{
    /* 1) I2C 总线 */
    const i2c_master_bus_config_t bus_cfg = {
        .i2c_port = BSP_I2C_PORT,
        .sda_io_num = BSP_I2C_SDA_GPIO,
        .scl_io_num = BSP_I2C_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .intr_priority = 0,
        .trans_queue_depth = 0,
        .flags = { .enable_internal_pullup = 1 },
    };
    ESP_RETURN_ON_ERROR(i2c_new_master_bus(&bus_cfg, &s_i2c_bus), TAG, "i2c bus");

    /* 2) TCA9557：先于一切外设 */
    s_tca_mutex = xSemaphoreCreateMutex();
    ESP_RETURN_ON_FALSE(s_tca_mutex, ESP_ERR_NO_MEM, TAG, "tca mutex");
    const i2c_device_config_t tca_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = BSP_TCA9557_ADDR,
        .scl_speed_hz = BSP_I2C_FREQ_HZ,
    };
    ESP_RETURN_ON_ERROR(i2c_master_bus_add_device(s_i2c_bus, &tca_cfg, &s_tca_dev), TAG, "tca9557");
    ESP_RETURN_ON_ERROR(tca_write_reg(BSP_TCA9557_REG_OUT, BSP_TCA9557_OUT_INIT), TAG, "tca out");
    ESP_RETURN_ON_ERROR(tca_write_reg(BSP_TCA9557_REG_CFG, BSP_TCA9557_CFG_VAL), TAG, "tca cfg");

    /* 3) LCD/触摸复位释放 + 背光 */
    ESP_RETURN_ON_ERROR(board_bsp_lcd_reset(), TAG, "lcd reset");
    ESP_RETURN_ON_ERROR(backlight_init(), TAG, "backlight");

    /* 4) 全双工 I2S + ES8311/ES7210 */
    ESP_RETURN_ON_ERROR(audio_i2s_init(), TAG, "i2s");
    ESP_RETURN_ON_ERROR(audio_codec_init(), TAG, "codec");

    /* 5) MicroSD：失败不阻断音频链路，录音模块自行检查 sd_ready */
    if (board_bsp_sd_mount() != ESP_OK) {
        ESP_LOGW(TAG, "continue without SD card");
    }

    ESP_LOGI(TAG, "board ready (16k/16bit mono, PSRAM policy active)");
    return ESP_OK;
}
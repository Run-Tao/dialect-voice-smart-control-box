/* -*- coding: utf-8 -*-
 * board_bsp.h — 立创·实战派 ESP32-S3 板级支持包（BSP）接口
 *
 * 职责边界：只做「硬件可用」——I2C 总线、PCA9557 IO 扩展、全双工 I2S、
 * ES7210/ES8311 编解码器、MicroSD 挂载、背光与按键。
 * **不含**录音缓冲、网络、UI 等业务逻辑（分别在 audio_recorder / network_client / ui_manager）。
 *
 * 初始化顺序有硬依赖，务必按 bsp_board_init() 内部顺序执行：
 *   I2C 总线 → PCA9557（先拉低 LCD_CS / 关闭功放）→ I2S → ES7210/ES8311 → SD → 背光
 * 原因：ES7210/ES8311 挂在 I2C 上，PCA9557 未配置时功放可能处于未知状态。
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "driver/i2c.h"
#include "driver/i2s_std.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- 总入口 */
/**
 * @brief 一次性初始化板级外设（幂等：重复调用直接返回 ESP_OK）。
 * @note  需在 app_main 中最先调用；内部打印每个子步骤的成败，便于定位。
 */
esp_err_t bsp_board_init(void);

/* ------------------------------------------------------------------ I2C */
/** @brief 初始化 I2C 主机总线（100 kHz）。重复调用安全。 */
esp_err_t bsp_i2c_init(void);
/** @brief 取 I2C 端口号（供 codec / 触摸驱动复用同一总线）。 */
i2c_port_t bsp_i2c_port(void);

/* -------------------------------------------------------------- PCA9557 */
/**
 * @brief 初始化 PCA9557：IO0..IO2 设为输出，并置安全初值
 *        （LCD_CS=高/不选中、PA_EN=低/功放关闭、DVP_PWDN=高/摄像头断电）。
 */
esp_err_t bsp_pca9557_init(void);
/**
 * @brief 单独设置某个 IO（内部读-改-写，不会影响同一寄存器的其它 IO）。
 * @param io_num 0..2，见 BSP_PCA9557_IO_xxx；其余 IO 保持输入
 */
esp_err_t bsp_pca9557_set(uint8_t io_num, bool level);
/** @brief 读回某个 IO 的输出锁存值。 */
esp_err_t bsp_pca9557_get(uint8_t io_num, bool *level);

/** @brief LCD 片选：enable=true 时拉低选中。ST7789 的 CS 只在 PCA9557 上。 */
esp_err_t bsp_lcd_cs(bool enable);
/** @brief NS4150B 功放使能：enable=true 时拉高开启。播放前置 true，播完置 false 省电。 */
esp_err_t bsp_pa_enable(bool enable);

/* ------------------------------------------------------------------ I2S */
/**
 * @brief 初始化 I2S0 全双工（TX=ES8311 播放，RX=ES7210 录音），共享 MCLK/BCLK/WS。
 *        格式固定 16 kHz / 16 bit / 立体声槽位；单声道提取在应用层完成。
 * @note  必须先于 bsp_codec_init()，因为 codec 需要 MCLK 已稳定输出。
 */
esp_err_t bsp_i2s_init(void);
i2s_chan_handle_t bsp_i2s_tx_handle(void);
i2s_chan_handle_t bsp_i2s_rx_handle(void);
/** @brief 使能 TX/RX 通道（开始时钟）。 */
esp_err_t bsp_i2s_start(void);
/** @brief 关闭 TX/RX 通道（停止时钟，省电）。 */
esp_err_t bsp_i2s_stop(void);

/* --------------------------------------------------------------- codecs */
/**
 * @brief 初始化 ES7210（ADC，双麦）与 ES8311（DAC，喇叭）。
 * @note  依赖组件：espressif/es8311、espressif/es7210（见 idf_component.yml）。
 */
esp_err_t bsp_codec_init(void);
/** @brief 设置播放音量 0..100。 */
esp_err_t bsp_codec_set_volume(int volume);
/** @brief 设置麦克风增益（dB，典型 0..30，ES7210 单级 3 dB 步进）。 */
esp_err_t bsp_codec_set_mic_gain(int db);

/* ------------------------------------------------------------- MicroSD */
/** @brief 以 SDMMC 1-bit 模式挂载到 /sdcard。已挂载时直接返回 ESP_OK。 */
esp_err_t bsp_sd_init(void);
bool bsp_sd_is_mounted(void);
esp_err_t bsp_sd_deinit(void);

/* --------------------------------------------------------------- 其它 */
/** @brief 背光开关（注意 GPIO42 为低电平点亮）。 */
esp_err_t bsp_lcd_backlight(bool on);
/** @brief 配置用户按键（GPIO0，低电平按下，内部上拉）。 */
esp_err_t bsp_button_init(void);
/** @brief 读取按键是否按下（已做消抖之外的原始电平判断）。 */
bool bsp_button_pressed(void);

#ifdef __cplusplus
}
#endif

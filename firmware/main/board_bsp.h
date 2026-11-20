/**
 * board_bsp.h — 板级支持包：I2C / TCA9557 / 全双工 I2S+Codec / MicroSD / 背光
 *
 * 初始化顺序（board_bsp_init 内部保证）：
 *   I2C 总线 -> TCA9557(方向+初始输出) -> LCD 复位释放 -> 背光
 *   -> I2S0 全双工通道 + ES8311/ES7210 codec 设备 -> MicroSD 挂载
 * 任何外设都不得在 TCA9557 配置之前被访问（功放/复位/摄像头电源都在其上）。
 *
 * 音频数据面统一为 16kHz / 16bit / 单声道：
 *   采集侧 ES7210 以 TDM 4 槽运行，esp_codec_dev 按 channel_mask=bit0
 *   提取主麦(MIC1)槽，read_mono 拿到的即单声道 PCM；
 *   播放侧 write_mono 送单声道，codec 层自动适配 ES8311。
 *
 * 线程约定：本层只做阻塞式硬件访问；SD 写文件与 HTTP 上传必须由
 * audio_recorder / network_client 的独立 FreeRTOS 任务调用，
 * 采集任务只调用 audio_read_mono，避免 DMA 溢出。
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "driver/i2c_master.h"
#include "sdmmc_cmd.h"

#ifdef __cplusplus
extern "C" {
#endif

/** 按上述顺序初始化全部板级外设。SD 挂载失败仅告警（sd_ready=false），
 *  不阻断音频链路；其余步骤失败返回错误码。 */
esp_err_t board_bsp_init(void);

/** 共享 I2C 总线句柄（供 ui_manager 挂 FT6336 触摸）。 */
i2c_master_bus_handle_t board_bsp_i2c_bus(void);

/** NS4150B 功放使能（TCA9557 P1）。播放前开、播放后关，避免底噪/爆音。 */
esp_err_t board_bsp_pa_enable(bool enable);

/** ST7789/FT6336 共用复位脉冲（TCA9557 P0 低 20ms + 释放后 120ms）。 */
esp_err_t board_bsp_lcd_reset(void);

/** 背光亮度 0-100（GPIO42 LEDC，硬件反相已在内部处理）。 */
esp_err_t board_bsp_backlight_set(uint8_t percent);

/** 打开采集通路：ES7210 TDM 4 槽 -> 提取 MIC1 单声道 16k/16bit。 */
esp_err_t board_bsp_audio_open_input(void);
esp_err_t board_bsp_audio_close_input(void);

/** 阻塞读取 samples 个单声道采样（内部即 samples*2 字节）。 */
esp_err_t board_bsp_audio_read_mono(int16_t *pcm, size_t samples);

/** 打开播放通路：ES8311 + 功放使能 + 默认音量。 */
esp_err_t board_bsp_audio_open_output(void);
esp_err_t board_bsp_audio_close_output(void);

/** 阻塞写入 samples 个单声道采样。 */
esp_err_t board_bsp_audio_write_mono(const int16_t *pcm, size_t samples);

/** 播放音量 0-100。 */
esp_err_t board_bsp_audio_set_volume(uint8_t percent);

/** MicroSD（1-bit SDMMC）挂载/卸载；重复挂载为幂等。 */
esp_err_t board_bsp_sd_mount(void);
esp_err_t board_bsp_sd_unmount(void);
bool board_bsp_sd_ready(void);
const sdmmc_card_t *board_bsp_sd_card(void);

/** GH1.25 外扩口演示 GPIO（index 0/1），首次调用自动配置为推挽输出。 */
esp_err_t board_bsp_exp_set(uint8_t index, bool level);

#ifdef __cplusplus
}
#endif
/* -*- coding: utf-8 -*-
 * board_pins.h — 立创·实战派 ESP32-S3（LCKFB-SZPI-ESP32-S3 V1.0）引脚总表
 *
 * 模组：ESP32-S3-WROOM-1-N16R8（16MB Flash / 8MB Octal PSRAM）
 *
 * ─────────────────────────────────────────────────────────────────────
 * 引脚来源与核实状态（2026-11-20 交叉核对三处独立来源）
 *   1) 立创官方例程仓库 gitee.com/zhanghaogit/lichuang-esp32s3-demo
 *      commit 828edecc8d5526369b9ee55262bf86a9f81695e1
 *      —— LCD / SD / I2S / 摄像头 / PCA9557 地址
 *   2) 实物读取记录（board 参数表）
 *      —— 背光低电平有效、I2C=GPIO1/2、用户键=GPIO0
 *   3) PCA9557 / ES8311 实际用法文章
 *      —— IO0=LCD_CS、IO1=PA_EN、IO2=DVP_PWDN、ES8311=0x18
 *
 * ⚠ 标记 [待核] 的项目未能在**一手原理图**上确认，首次上电请优先核对；
 *   核对方法见 README「引脚核对清单」。
 *
 * ⚠ 器件命名更正：用户描述中的 "TCA9557" 在实板上为 **PCA9557**
 *   （TI 原厂型号）。两者寄存器映射与 I2C 地址完全兼容（0x19），
 *   本工程按 PCA9557 编写，代码对 TCA9557 同样可用。
 * ─────────────────────────────────────────────────────────────────────
 */
#pragma once

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "driver/spi_common.h"   /* SPI3_HOST 等总线编号 */
#include "hal/i2s_types.h"       /* I2S_NUM_0 / I2S_MCLK_MULTIPLE_256 */

/* =====================================================================
 * 一、I2C 总线（PCA9557 / ES7210 / ES8311 / FT6336 四者共用）
 *   官方例程网络：IIC_SDA / IIC_SCL，总线使用内部/板上上拉
 * ===================================================================== */
#define BSP_I2C_PORT            I2C_NUM_0
#define BSP_I2C_SDA_GPIO        GPIO_NUM_1
#define BSP_I2C_SCL_GPIO        GPIO_NUM_2
#define BSP_I2C_CLK_HZ          100000      /* 100 kHz：codec 配置对时序敏感，勿随意升到 400k */

/* =====================================================================
 * 二、PCA9557 IO 扩展（I2C 地址 0x19）
 *   注意：这 3 个 IO 共用同一个 8 位输出寄存器，
 *   驱动必须「读-改-写」，不能直接整字节覆盖（否则会误改其它 IO）。
 * ===================================================================== */
#define BSP_PCA9557_ADDR        0x19
#define BSP_PCA9557_REG_INPUT   0x00        /* 输入端口（只读） */
#define BSP_PCA9557_REG_OUTPUT  0x01        /* 输出端口 */
#define BSP_PCA9557_REG_POLAR   0x02        /* 极性反转（默认 0） */
#define BSP_PCA9557_REG_CONFIG  0x03        /* 方向：1=输入，0=输出 */

#define BSP_PCA9557_IO_LCD_CS   0           /* IO0：LCD 片选，低=选中（ST7789 无 GPIO CS） */
#define BSP_PCA9557_IO_PA_EN    1           /* IO1：NS4150B 功放使能，高=开启 */
#define BSP_PCA9557_IO_DVP_PWDN 2           /* IO2：摄像头掉电，低=使能（本工程不使用摄像头） */
/* IO3..IO7 在官方板级初始化中保持输入，禁止当作普通输出使用。 */

/* =====================================================================
 * 三、I2S 音频（全双工：ES7210 采集 + ES8311 播放，共享 MCLK/BCLK/WS）
 *   原理图网络：I2S_MCLK / I2S_BCLK / I2S_LRCK / I2S_SDIN / I2S_DOUT
 * ===================================================================== */
#define BSP_I2S_PORT            I2S_NUM_0
#define BSP_I2S_MCLK_GPIO       GPIO_NUM_38
#define BSP_I2S_BCLK_GPIO       GPIO_NUM_14
#define BSP_I2S_WS_GPIO         GPIO_NUM_13   /* 即 LRCK */
#define BSP_I2S_SDIN_GPIO       GPIO_NUM_12   /* ES7210(ADC) → ESP32-S3，录音 */
#define BSP_I2S_DOUT_GPIO       GPIO_NUM_45   /* ESP32-S3 → ES8311(DAC)，播放 */

#define BSP_I2S_SAMPLE_RATE     16000
#define BSP_I2S_BIT_WIDTH       16
#define BSP_I2S_MCLK_MULTIPLE   256           /* MCLK = 16000 × 256 = 4.096 MHz */

/* 音频编解码器 I2C 地址 */
#define BSP_ES8311_ADDR         0x18          /* 已核实 */
#define BSP_ES7210_ADDR         0x41          /* 官方 04-audio_es7210 例程确认（AD1=0, AD0=1） */
#define BSP_ES7210_MIC_GAIN_DB  30            /* 双麦模拟 PGA 默认增益，常见 18..30 dB */
/* 触摸：FT6336（用 FT5x06 驱动） */
#define BSP_TOUCH_ADDR          0x38          /* esp_lcd_touch_ft5x06 组件默认地址 */
#define BSP_TOUCH_INT_GPIO      GPIO_NUM_NC   /* 官方配置为 NC，使用轮询 */
#define BSP_TOUCH_RST_GPIO      GPIO_NUM_NC   /* 官方配置为 NC，由板级复位 */

/* =====================================================================
 * 四、ST7789 LCD（2.0" IPS 320×240，SPI3）
 *   ⚠ CS 不接 GPIO！由 PCA9557 IO0 控制，调用 bsp_lcd_cs(true/false)
 *   ⚠ 背光 GPIO42 为**低电平点亮**，上电默认呈熄灭状态
 * ===================================================================== */
#define BSP_LCD_SPI_HOST        SPI3_HOST
#define BSP_LCD_SCLK_GPIO       GPIO_NUM_41
#define BSP_LCD_MOSI_GPIO       GPIO_NUM_40
#define BSP_LCD_DC_GPIO         GPIO_NUM_39
#define BSP_LCD_BL_GPIO         GPIO_NUM_42   /* 低电平点亮 */
#define BSP_LCD_BL_LEVEL_ON     0
#define BSP_LCD_BL_LEVEL_OFF    1

#define BSP_LCD_H_RES           320
#define BSP_LCD_V_RES           240
#define BSP_LCD_SPI_HZ          (40 * 1000 * 1000)

/* =====================================================================
 * 五、MicroSD（SDMMC 1-bit 模式）
 *   原理图网络：SD_CLK / SD_CMD / SD_D0
 * ===================================================================== */
#define BSP_SD_CLK_GPIO         GPIO_NUM_47
#define BSP_SD_CMD_GPIO         GPIO_NUM_48
#define BSP_SD_D0_GPIO          GPIO_NUM_21
#define BSP_SD_WIDTH            1             /* 1-bit：板上仅引出 D0 */
#define BSP_SD_MOUNT_POINT      "/sdcard"
#define BSP_SD_RECORD_DIR       "/sdcard/dialect"

/* =====================================================================
 * 六、用户交互
 * ===================================================================== */
#define BSP_BTN_REC_GPIO        GPIO_NUM_0    /* BOOT 键，低电平按下，用于「按住录音」 */
#define BSP_BTN_ACTIVE_LEVEL    0

/* =====================================================================
 * 七、GH1.25-5P 扩展接口（演示指令执行）
 *   [待核] 扩接口具体 GPIO 随跳线选择而变，未在官方例程中固定。
 *   默认留空，由用户在 menuconfig 或本文件按实际接线填写。
 * ===================================================================== */
#define BSP_EXT1_GPIO_A         GPIO_NUM_NC
#define BSP_EXT1_GPIO_B         GPIO_NUM_NC
#define BSP_EXT2_GPIO_A         GPIO_NUM_NC
#define BSP_EXT2_GPIO_B         GPIO_NUM_NC

/* =====================================================================
 * 八、录音参数（与后端 / 模型约定一致，勿单独修改）
 * ===================================================================== */
#define BSP_REC_MAX_SECONDS     10            /* 单次录音上限 10 s */
#define BSP_REC_BYTES_PER_SEC   (BSP_I2S_SAMPLE_RATE * (BSP_I2S_BIT_WIDTH / 8))  /* 32000 B/s */
#define BSP_REC_MAX_PCM_BYTES   (BSP_REC_BYTES_PER_SEC * BSP_REC_MAX_SECONDS)   /* 320000 B */
/* PSRAM 缓冲留 1.5 倍余量，容纳对齐与重采样抖动 */
#define BSP_REC_PSRAM_BYTES     (BSP_REC_MAX_PCM_BYTES * 3 / 2)

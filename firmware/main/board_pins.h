/**
 * board_pins.h — 立创·实战派 ESP32-S3 (LCKFB-SZPI-ESP32-S3) V1.0 引脚集中定义
 *
 * 依据（核对顺序）：
 *   [1] 立创官方 wiki《【立创·实战派ESP32-S3】文档教程》第 6/7/8/9 章例程引脚宏
 *       https://wiki.lckfb.com/zh-hans/szpi-esp32s3/beginner/sd-card.html
 *       https://wiki.lckfb.com/zh-hans/szpi-esp32s3/beginner/audio-input-es7210.html
 *   [2] xiaozhi-esp32 官方板级支持 main/boards/lckfb/szpi-esp32s3/{config.h, lichuang_dev_board.cc}
 *       https://github.com/78/xiaozhi-esp32/tree/main/main/boards/lckfb/szpi-esp32s3
 *   [3] 立创开源硬件平台原理图（oshwhub 实战派 ESP32-S3 工程，V1.0/V1.1）
 *
 * 注意：本文件所有编号均已与 [1][2] 交叉核对；若手中原理图版本不同（V0.3/V0.4），
 *       以原理图为准并仅修改本文件，其余模块不得散落 GPIO 字面量。
 */
#pragma once

#include "driver/gpio.h"

/* ------------------------------------------------------------------ */
/* 1. I2C 控制总线（ES8311 / ES7210 / TCA9557 / FT6336 / QMI8658 共用） */
/*    wiki 第 7 章: EXAMPLE_I2C_SDA_IO=1, EXAMPLE_I2C_SCL_IO=2          */
/* ------------------------------------------------------------------ */
#define BSP_I2C_PORT            I2C_NUM_0
#define BSP_I2C_SDA_GPIO        GPIO_NUM_1
#define BSP_I2C_SCL_GPIO        GPIO_NUM_2
#define BSP_I2C_FREQ_HZ         400000      /* FT6336 支持 400kHz */

/* ------------------------------------------------------------------ */
/* 2. TCA9557 IO 扩展器（原理图网络名 PCA9557，寄存器兼容 TCA9557）     */
/*    xiaozhi: 地址 0x19；上电写 OUT=0x03, CFG=0xF8                     */
/*    P0 = LCD_RST   (低电平复位 ST7789/FT6336，共用复位)               */
/*    P1 = PA_CTRL   (高电平使能 NS4150B 功放)                          */
/*    P2 = CAM_PWR   (低电平给摄像头供电；本项目不用摄像头，保持高)     */
/*    P3..P7 = 输入方向保留，禁止当作输出使用                           */
/* ------------------------------------------------------------------ */
#define BSP_TCA9557_ADDR        0x19
#define BSP_TCA9557_REG_IN      0x00
#define BSP_TCA9557_REG_OUT     0x01
#define BSP_TCA9557_REG_POL     0x02
#define BSP_TCA9557_REG_CFG     0x03
#define BSP_TCA9557_BIT_LCD_RST 0
#define BSP_TCA9557_BIT_PA_CTRL 1
#define BSP_TCA9557_BIT_CAM_PWR 2
/* 方向：P0-P2 输出，P3-P7 输入 */
#define BSP_TCA9557_CFG_VAL     0xF8
/* 初始输出：LCD 复位释放(1)、功放关(0)、摄像头断电(1) */
#define BSP_TCA9557_OUT_INIT    0x05

/* ------------------------------------------------------------------ */
/* 3. I2S0 全双工（ES7210 ADC 与 ES8311 DAC 共享 MCLK/BCLK/WS）         */
/*    wiki 第 7 章: MCK=38, BCK=14, WS=13, DI=12                        */
/*    xiaozhi config.h: DOUT=45 (ESP32 -> ES8311 DSDIN)                 */
/*    DIN=12 为 ES7210 SDOUT -> ESP32                                  */
/* ------------------------------------------------------------------ */
#define BSP_I2S_PORT            I2S_NUM_0
#define BSP_I2S_MCLK_GPIO       GPIO_NUM_38
#define BSP_I2S_BCLK_GPIO       GPIO_NUM_14
#define BSP_I2S_WS_GPIO         GPIO_NUM_13
#define BSP_I2S_DIN_GPIO        GPIO_NUM_12
#define BSP_I2S_DOUT_GPIO       GPIO_NUM_45
#define BSP_AUDIO_SAMPLE_RATE   16000       /* 全链路统一 16kHz/16bit/单声道 */
#define BSP_AUDIO_MCLK_MULT     I2S_MCLK_MULTIPLE_256
#define BSP_I2S_DMA_DESC_NUM    6           /* 6 x 240 帧 = 90ms DMA 余量 */
#define BSP_I2S_DMA_FRAME_NUM   240

/* ES7210 通道地图（wiki 第 7 章）：
 *   MIC1/MIC2 = 双 ZTS6216 模拟麦克风；MIC3 = ES8311 播放回采(回声参考)；MIC4 悬空。
 *   TDM 槽序为 MIC1, MIC3, MIC2, MIC4（xiaozhi box_audio_codec 注释），
 *   故主麦 = slot0，用 channel_mask bit0 提取，勿按物理 MIC 编号猜槽位。 */
#define BSP_ES7210_ADDR         0x41        /* AD0=1, AD1=0 (wiki 第 7 章) */
#define BSP_ES8311_ADDR         0x18        /* ES8311_CODEC_DEFAULT_ADDR */
#define BSP_MIC_GAIN_DB         30.0f       /* 主麦 PGA 增益 */
#define BSP_SPK_VOL_DEFAULT     70          /* 0-100 */

/* ------------------------------------------------------------------ */
/* 4. MicroSD —— 1-bit SDMMC（wiki 第 6 章：BSP_SD_CLK=47/CMD=48/D0=21）*/
/*    注意：GPIO48 被 SD CMD 占用，故本板无 GPIO48 板载 LED 可用。       */
/* ------------------------------------------------------------------ */
#define BSP_SD_CLK_GPIO         GPIO_NUM_47
#define BSP_SD_CMD_GPIO         GPIO_NUM_48
#define BSP_SD_D0_GPIO          GPIO_NUM_21
#define BSP_SD_BUS_WIDTH        1
#define BSP_SD_MOUNT_POINT      "/sdcard"
#define BSP_SD_MAX_FILES        5
#define BSP_SD_ALLOC_UNIT       (16 * 1024)

/* ------------------------------------------------------------------ */
/* 5. ST7789 2.0" IPS 320x240（SPI3_HOST，无 CS，复位走 TCA9557 P0）    */
/*    xiaozhi: MOSI=40, SCLK=41, DC=39, spi_mode=2, pclk=80MHz          */
/* ------------------------------------------------------------------ */
#define BSP_LCD_SPI_HOST        SPI3_HOST
#define BSP_LCD_MOSI_GPIO       GPIO_NUM_40
#define BSP_LCD_MISO_GPIO       GPIO_NUM_NC
#define BSP_LCD_SCLK_GPIO       GPIO_NUM_41
#define BSP_LCD_CS_GPIO         GPIO_NUM_NC
#define BSP_LCD_DC_GPIO         GPIO_NUM_39
#define BSP_LCD_SPI_MODE        2           /* 同 xiaozhi；花屏时对照原理图试 0/3 */
#define BSP_LCD_PCLK_HZ         (80 * 1000 * 1000)
#define BSP_LCD_H_RES           320
#define BSP_LCD_V_RES           240
#define BSP_LCD_SWAP_XY         true
#define BSP_LCD_MIRROR_X        true
#define BSP_LCD_MIRROR_Y        false

/* 背光：GPIO42 LEDC PWM，反相（duty=0 最亮）。不在 TCA9557 上。 */
#define BSP_LCD_BL_GPIO         GPIO_NUM_42
#define BSP_LCD_BL_INVERT       true
#define BSP_LCD_BL_LEDC_TIMER   LEDC_TIMER_0
#define BSP_LCD_BL_LEDC_CHAN    LEDC_CHANNEL_0
#define BSP_LCD_BL_FREQ_HZ      5000
#define BSP_LCD_BL_DEFAULT_PCT  80

/* FT6336 电容触摸：与 LCD 共用 I2C 总线与复位线，无独立 INT */
#define BSP_TOUCH_ADDR          0x38
#define BSP_TOUCH_INT_GPIO      GPIO_NUM_NC
#define BSP_TOUCH_RST_GPIO      GPIO_NUM_NC   /* 共用 TCA9557 P0 */

/* ------------------------------------------------------------------ */
/* 6. 用户按键（BOOT键，低电平按下）——按住说话触发源                    */
/* ------------------------------------------------------------------ */
#define BSP_KEY_USER_GPIO       GPIO_NUM_0
#define BSP_KEY_ACTIVE_LEVEL    0

/* ------------------------------------------------------------------ */
/* 7. GH1.25-5P 外扩口演示 GPIO                                        */
/*    推断值：全板分配后仅剩 GPIO10/11 空闲（摄像头占用 3-9/15-18/46，  */
/*    USB=19/20，UART0=43/44）。烧录前务必对照原理图外扩接口页核实！    */
/* ------------------------------------------------------------------ */
#define BSP_EXP1_GPIO           GPIO_NUM_10
#define BSP_EXP2_GPIO           GPIO_NUM_11

/* 板载 LED：V1.0 上 GPIO48 已作 SD CMD，无可用板载 LED，状态指示用屏幕。 */
#define BSP_LED_GPIO            GPIO_NUM_NC
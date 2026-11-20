# DialectVoice SZPI · 项目进度

**一句话状态**：第 1 步目录结构与第 2 步 `board_pins.h`、`sdkconfig.defaults`、`board_bsp.c/h` 已落到 `D:\Run_Tao\DialectVoice_SZPI`；ESP-IDF 5.4.1 目标 `esp32s3` 全量构建通过，待真实开发板烧录与 I2C/I2S/SD 实机自检。

**最后更新**：2026-11-20 14:26（按立创官方示例 `lichuang-esp32s3-demo@828edec` 核实引脚；ES7210 地址修正为 0x41；PCA9557 仅将 IO0..IO2 配置为输出；锁定 ES7210/ES8311/LVGL 8.3/触摸组件；构建产出 `firmware/build/dialect_voice_szpi.bin`）

## 目录地图

| 路径 | 作用 |
|---|---|
| `firmware/` | ESP-IDF v5.x 固件工程 |
| `firmware/main/` | BSP、引脚表、阶段一硬件自检入口 |
| `backend/` | 后续 FastAPI/ASR/NLU 代码占位 |

## 关键文件

- 引脚与硬件参数：`firmware/main/board_pins.h`
- 板级驱动：`firmware/main/board_bsp.c`、`firmware/main/board_bsp.h`
- 默认配置：`firmware/sdkconfig.defaults`
- 分区表：`firmware/partitions.csv`
- 组件锁定：`firmware/dependencies.lock`
- 阶段一自检：`firmware/main/main.c`
- 工程说明：`README.md`

## 已完成

- 完整规划 `firmware/` 与 `backend/` 目录。
- 按官方示例核实 I2C=GPIO1/2、I2S=38/14/13/12/45、LCD=41/40/39/42、SDMMC=47/48/21、按键=GPIO0。
- 核实 PCA9557=0x19、ES8311=0x18、ES7210=0x41、FT6336=0x38；PCA9557 IO0=LCD_CS、IO1=PA_EN、IO2=DVP_PWDN。
- 实现 I2C、PCA9557 安全初始化与读改写、I2S0 全双工、ES8311、ES7210、SD/MMC、背光与按键。
- ES7210 已按组件 `1.0.1~1` 的 `new_codec/config_codec/config_volume` API 实现。
- 默认值采用 8 MB Octal PSRAM、仅 `heap_caps_malloc(...MALLOC_CAP_SPIRAM)` 外部内存路径、FATFS LFN、LVGL 8.3 RGB565。
- `idf.py build` 全量通过；最终固件 0x8b170 字节（569712 B），4 MB app 分区剩余 86%。
- 最终固件 SHA-256：`4E2267E1AE9F2368C2CECC939E7ACA81D4A44FC68317E5093A2D159675C9248F`。

## 下一步

1. 实板烧录阶段一自检，确认 PSRAM=8 MB、四个 I2C 地址、SD 容量、背光与按键。
2. 实机确认 ES7210 两路 I2S 槽位与主麦通道索引，再写 `audio_recorder`。
3. 生成实现 `network_client`、`ui_manager` 与状态机。
4. 实现 FastAPI ASR/NLU 后端。

## 已知陷阱

- 板上实际器件是 **PCA9557**（与用户描述 TCA9557 寄存器兼容）；PCA9557 的 IO3..IO7 必须保持输入。
- ST7789 的 CS 不在 GPIO 上，必须通过 PCA9557 IO0 控制；拉低才选中。
- GPIO42 背光为低电平点亮。
- 8 MB PSRAM 是 Octal 模式，不能用 quad 配置；普通 `malloc` 不用于大块音频/HTTP 缓冲。
- GH1.25-5P 扩展口的具体 GPIO 未在官方例程固定，当前保留为 `GPIO_NUM_NC`。
- 当前只有编译验证，尚无真实硬件烧录/波形/录音验证。

## 验证命令

```powershell
cd D:\Run_Tao\DialectVoice_SZPI
powershell -ExecutionPolicy Bypass -File .\flash.ps1 -BuildOnly
```

实板：

```powershell
powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Reconfig -Port COM7 -Monitor
```

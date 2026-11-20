# 固件：立创·实战派 ESP32-S3 方言语音终端

端云协同终端固件（ESP-IDF v5.x，C）。按住 BOOT 键或屏幕「按住说话」→ 录音（最长 10s）
→ SD 存档 → multipart 上传 PC 云端 → 屏幕展示识别结果并执行 GPIO → ✓/✗ 人工标注回传。

## 目录

```
firmware/
  CMakeLists.txt / partitions.csv / sdkconfig.defaults
  main/
    board_pins.h        全部 GPIO 集中定义（附 wiki/xiaozhi 来源注释）
    board_bsp.c/h       I2C→TCA9557→LCD 复位→背光→I2S+ES8311/ES7210→SD
    audio_recorder.c/h  PSRAM 录音缓冲 + 采集/写卡双任务 + WAV 头
    network_client.c/h  Wi-Fi/SNTP + PSRAM multipart 上传 + cJSON 契约解析 + feedback
    ui_manager.c/h      ST7789+FT6336+LVGL8 五态界面（手写端口层）
    main.c              状态机 IDLE→RECORDING→UPLOADING→DISPLAY→FEEDBACK
    fonts/              gen_symbols.py + lv_font_conv 生成的 ui_font_16.c（243 汉字子集）
```

## 构建与烧录

需要 ESP-IDF **v5.1+**（新 I2S 驱动 + esp_codec_dev 托管组件）。本交付 PC 未安装
ESP-IDF 工具链，**固件未在本机编译验证**；以下命令在装有 IDF 的机器执行：

```bash
idf.py set-target esp32s3
# 修改 main.c 顶部宏：WIFI_SSID / WIFI_PASSWORD / CLOUD_BASE_URL / DEVICE_ID
idf.py build
idf.py -p COMx flash monitor
```

组件由 `main/idf_component.yml` 自动拉取：esp_codec_dev、esp_lcd_st7789、
esp_lcd_touch_ft5x06、lvgl^8.4。

字库重新生成（新增说法/汉字后）：

```bash
cd main/fonts
python gen_symbols.py          # 从 commands.yaml/语料/UI 文案收集汉字
node node_modules/lv_font_conv/lv_font_conv.js --font C:/Windows/Fonts/simhei.ttf \
  --size 16 --bpp 4 --range 0x20-0x7F --symbols "$(cat symbols.txt)" \
  --format lvgl --lv-include lvgl.h --output ui_font_16.c
```

## 任务与优先级（采集/写卡/上传解耦，防 DMA 丢帧）

| 任务 | 优先级 | 核心 | 职责 |
|---|---|---|---|
| rec_cap | 10 | core1 | 只读 I2S 主麦通道 → PSRAM 缓冲 |
| rec_sd | 5 | core0 | WAV 头 + fwrite /sdcard/dialect/ |
| net_up | 5 | core0 | PSRAM 拼 multipart + esp_http_client |
| ui_lvgl | 4 | core0 | lv_timer_handler + 触摸读取 |
| app_fsm | 6 | core0 | 事件队列状态机、GPIO 执行、提示音 |

## 硬约束落地清单

- 音频缓冲 320000B、HTTP 请求体（≤320KB+边界）、LVGL 双缓冲 2×25.6KB 全部
  `heap_caps_malloc(..., MALLOC_CAP_SPIRAM)`；`SPIRAM_MALLOC_ALWAYSINTERNAL=16384`
  兜底保证 >16KB 连续块不进内部 SRAM。
- SD 写、HTTP 上传各自独立 FreeRTOS 任务，与 I2S 采集任务不同核/不同优先级。
- `CONFIG_I2S_ISR_IRAM_SAFE=y`，Flash 操作期间 DMA 中断不延迟。
- GPIO 全部集中在 `board_pins.h`，注释标明 wiki/原理图对应关系。
- 无伪代码：所有模块为完整可编译意图的实现。

## 执行绑定与门槛

`confidence >= 0.7 && intent == device_control` 才执行：
扩展口 1（GPIO10）= 灯 on/off，扩展口 2（GPIO11）= 风扇 on/off；
其余设备云端状态已更新、终端仅日志展示。
**GPIO10/11 为按原理图剩余空闲脚推断，烧录前务必对照立创实战派 V1.0 原理图核实。**

## 已知待硬件验证项

1. 扩展口引脚 GPIO10/11（推断值）。
2. LCD `spi_mode=2`（沿用 xiaozhi 板级配置；花屏时试 0/3）。
3. ES7210 TDM slot 顺序 MIC1,MIC3,MIC2,MIC4 → 主麦取 slot0（channel_mask bit0）。
4. 背光 GPIO42 PWM 反相；SD 为 1-bit SDMMC（GPIO47/48/21），GPIO48 被占用故无板载 LED。
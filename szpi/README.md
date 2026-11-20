# 方言语音指令 · 端云协同系统（立创·实战派 ESP32-S3）

基于 **立创·实战派 ESP32-S3（LCKFB-SZPI-ESP32-S3 V1.0，ESP32-S3-WROOM-1-N16R8）**
的方言语音采集 / 转写 / 语义理解闭环。

- 端侧：按键或触屏触发录音 → PSRAM 缓冲 → SD 归档 → Wi-Fi 上传 → 屏显结果 → 触屏标注纠错 → 语音播报
- 云端：FastAPI 接收 WAV → 方言 ASR → LLM 结构化意图 → JSONL 数据集与标注管理

---

## 一、工程目录结构

```
DialectVoice_SZPI/
├── README.md                          # 本文件
├── firmware/                          # ── ESP-IDF v5.x 工程 ──
│   ├── CMakeLists.txt                 # 顶层工程定义
│   ├── sdkconfig.defaults             # N16R8 / Octal PSRAM / FATFS LFN / LVGL 关键配置
│   ├── partitions.csv                 # 16MB Flash 分区（factory 4MB + storage 11MB）
│   └── main/
│       ├── CMakeLists.txt             # 组件源文件与依赖声明
│       ├── idf_component.yml          # 组件管理器依赖（es8311 / es7210）
│       ├── board_pins.h               # ★ 引脚总表（含来源标注与 [待核] 项）
│       ├── board_bsp.h                # ★ 板级支持包接口
│       ├── board_bsp.c                # ★ I2C / PCA9557 / 全双工 I2S / 双 codec / SD
│       ├── main.c                     # 阶段一：硬件自检入口
│       │
│       │   ── 以下为阶段二/三交付，尚未创建 ──
│       ├── audio_recorder.h / .c      # PSRAM 录音缓冲、WAV 头、SD 落盘
│       ├── network_client.h / .c      # Wi-Fi、multipart 上传、cJSON、反馈接口
│       ├── ui_manager.h / .c          # LVGL 320×240 界面与触控回调
│       └── assets/                    # LVGL 中文字库、提示音 PCM
│
└── backend/                           # ── Python 3.10+ FastAPI ──
    ├── main.py                        # /api/v1/speech/understand、/feedback
    ├── asr_nlu_pipeline.py            # ASR 适配层 + LLM 结构化意图
    ├── requirements.txt
    ├── .env.example                   # API Key / 模型名 / 方言配置
    ├── README.md                      # 部署与 curl 验证
    └── dataset/
        ├── audio/                     # WAV 归档（按日期分目录）
        └── metadata.jsonl             # 元数据 + 标注状态（append-only）
```

**设计理由**

| 决策 | 原因 |
|---|---|
| BSP 与业务分层 | `board_bsp` 只做「硬件可用」，业务模块可独立单测；换板只改 BSP |
| 阶段一先出自检 `main.c` | 硬件问题（I2C 不通 / PSRAM 未开 / CS 漏控）必须在写业务前暴露，否则后面所有 bug 都会怀疑到错误的地方 |
| 引脚集中在 `board_pins.h` | 用户明确要求可配置 + 原理图对照注释；单文件改完即生效 |
| 分区表 factory 4MB | 当前固件约 1MB，给 LVGL + 中文字库留足空间；录音不占 Flash |

---

## 二、引脚表（已交叉核实，⚠ 含待核项）

来源：① 立创官方例程 `gitee.com/zhanghaogit/lichuang-esp32s3-demo`（commit `828edec`）；② 实物读取记录；③ 板卡宣传资料。三处一致项标 ✅，仅单一来源标 ⚠。

### I2C（四器件共用）

| 信号 | GPIO | 状态 |
|---|---:|---|
| SDA | 1 | ✅ |
| SCL | 2 | ✅ |

### PCA9557（IO 扩展，⚠ 用户描述为 TCA9557，实板为 PCA9557，寄存器/地址兼容）

| 位 | 用途 | 有效电平 | 状态 |
|---|---|---|---|
| IO0 | LCD_CS | 低 = 选中 | ✅ |
| IO1 | PA_EN（NS4150B 功放） | 高 = 开启 | ✅ |
| IO2 | DVP_PWDN（摄像头） | 低 = 使能 | ✅（本工程不用） |
| I2C 地址 | 0x19 | — | ✅ |

> **最容易踩的坑**：ST7789 的 CS **不在任何 GPIO 上**，只能通过 PCA9557 IO0 控制。
> 若按常规 `spi_bus_config` 配 CS，屏永远不亮。

### I2S 音频（全双工，共享 MCLK/BCLK/WS）

| 信号 | GPIO | 状态 |
|---|---:|---|
| MCLK | 38 | ✅ |
| BCLK | 14 | ✅ |
| WS / LRCK | 13 | ✅ |
| SDIN（ES7210 → SoC，录音） | 12 | ✅ |
| DOUT（SoC → ES8311，播放） | 45 | ✅ |
| ES8311 I2C 地址 | 0x18 | ✅ |
| ES7210 I2C 地址 | 0x41 | ✅ 官方 `04-audio_es7210` 例程确认 |

### ST7789 LCD（SPI3，320×240）

| 信号 | GPIO | 状态 |
|---|---:|---|
| SCLK | 41 | ✅ |
| MOSI | 40 | ✅ |
| DC | 39 | ✅ |
| BL 背光 | 42 | ✅ **低电平点亮** |
| CS | PCA9557 IO0 | ✅ |

### MicroSD（SDMMC 1-bit）

| 信号 | GPIO | 状态 |
|---|---:|---|
| CLK | 47 | ✅ |
| CMD | 48 | ✅ |
| D0 | 21 | ✅ |
| 挂载点 | `/sdcard` | — |

### 其它

| 项 | 值 | 状态 |
|---|---|---|
| 用户按键 | GPIO0（BOOT，低有效） | ✅ |
| 触摸 FT6336 | I2C 0x38，INT/RST 均为 NC | ✅ 官方 `esp_lcd_touch_ft5x06` 配置 |
| GH1.25-5P 扩展 | 未固定 | ⚠ 待核，由使用者按实际接线填 |

### 上电必做的 4 项核对

1. **PSRAM 是否真的 8MB**：自检日志出现 `psram : 8 MB (Octal OK)`；若为 0 或 2MB，检查 `CONFIG_SPIRAM_MODE_OCT=y` 与是否删过 `sdkconfig`。
2. **I2C 是否扫到 4 个器件**：`0x19 / 0x18 / 0x41 / 0x38`。缺哪个按上表优先级核对。
3. **背光是否闪 2 次**：不闪说明 GPIO42 极性或 PCA9557 初值有问题。
4. **屏是否能亮**：本阶段只验证 BSP，ST7789 面板初始化在阶段二（`ui_manager`）。

---

## 三、烧录（Windows，本机环境已就绪）

### 3.1 本机现状（已核实并修复）

| 项 | 值 |
|---|---|
| ESP-IDF | **v5.4.1**，路径 `D:\Run_Tao\esp-idf` |
| 工具链 | `C:\Users\Taoli\.espressif`（xtensa-esp-elf / riscv32 齐全） |
| Python 环境 | `C:\Users\Taoli\.espressif\python_env\idf5.4_py3.13_env` |
| `idf.py` | 不在全局 PATH，**必须**先 import `export.ps1`（脚本已代劳） |

**本工程首次编译前修复的 4 个环境问题**（记录下来避免重复踩坑）：

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | `Unable to import the rich module` | IDF venv 缺 `rich` | `pip install rich` |
| 2 | `No module named 'pip._internal.operations.install.wheel'` | venv 里 pip **26.3** 过新，内部 API 与 IDF 5.4.1 不兼容 | 降级 `pip==24.3.1` |
| 3 | `"powershell.EXE" shell is not among the supported options` | IDF 5.4.1 的 `activate_venv.py` 查表**大小写敏感**，Windows 返回 `powershell.EXE` | 改 `detected_shell.lower()`（一行，已改在 `esp-idf/tools/export_utils/activate_venv.py`） |
| 4 | `CMAKE_RC_COMPILER not set` / `needs language RC` | CMake 3.30 在 Windows 上自动启用 RC 语言，xtensa 工具链无 windres | 顶层 `project(... C CXX ASM)` 显式声明语言 |

> 第 3 项改的是 IDF 源码（一行）。若日后 `git pull` 更新 IDF 被覆盖，重跑 `export.ps1` 报同样错误时按上表再改一次即可；`flash.ps1` 也会在 `rich` 缺失时自动补装。

**已验证的编译结果**（2026-11-20）：

```
build/dialect_voice_szpi.bin        556.4 KB
build/bootloader/bootloader.bin      20.5 KB
build/partition_table/partition-table.bin  3 KB
```

配置确认生效：`SPIRAM_MODE_OCT=y`、`ESPTOOLPY_FLASHSIZE_16MB=y`、`FATFS_LFN_HEAP=y`、
`FATFS_MAX_LFN=255`、`CPU_FREQ_240MHz`、`FREERTOS_HZ=1000`、自定义分区表。

### 3.2 一块命令完成编译 + 烧录 + 串口监视

```powershell
cd D:\Run_Tao\DialectVoice_SZPI

# 自动探测串口、编译、烧录、打开监视器（Ctrl+] 退出）
powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Monitor

# 指定串口（推荐，避免多串口选错）
powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Port COM7 -Monitor

# 板子上跑过别的固件（如小智），先整片擦除再烧
powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Erase -Port COM7 -Monitor

# 只编译不烧（检查代码是否能过编译）
powershell -ExecutionPolicy Bypass -File .\flash.ps1 -BuildOnly

# 改过 sdkconfig.defaults 后，删配置重建
powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Reconfig -Port COM7
```

### 3.3 串口怎么找

1. 板子上通常有**两个 USB-C**：一个是 USB-HUB/UART（CH343 类），一个是 ESP32-S3 原生 USB（VID 303A）。**优先用标着 UART / COM 的那个**，它烧录最稳。
2. 插上后打开「设备管理器 → 端口(COM 和 LPT)」，应出现：
   - `USB-SERIAL CH343 (COMx)`（UART 口），或
   - `USB JTAG/serial debug unit (COMx)`（原生 USB 口）
3. 若显示为**未知设备/带感叹号**：装 CH343 驱动（沁恒官网）或让 Windows Update 自动装；ESP32-S3 原生 USB 在 Win10+ 免驱。
4. 命令行快速查：
   ```powershell
   [System.IO.Ports.SerialPort]::GetPortNames()
   ```

### 3.4 进不了下载模式怎么办

ESP32-S3 一般能自动进入下载模式。若 `flash` 报 `Failed to connect`：

1. **按住 BOOT（GPIO0）不放 → 点一下 RST → 松开 BOOT**，再执行烧录；
2. 确认没有其它程序占用串口（串口助手、Arduino IDE、上一次没退出的 `monitor`）；
3. 换一根**能传数据**的 USB 线（很多线只供电）；
4. 换另一个 USB-C 口再试。

### 3.5 手动命令（不用脚本时）

```powershell
# 每个新开的 PowerShell 窗口都要先做这一步
. D:\Run_Tao\esp-idf\export.ps1

cd D:\Run_Tao\DialectVoice_SZPI\firmware
idf.py set-target esp32s3      # 仅首次
idf.py build
idf.py -p COM7 flash monitor
```

### 3.6 烧完应该看到什么

```
I (xxx) selftest: ================ BOARD INFO ================
I (xxx) selftest: chip      : esp32s3, 2 core(s), rev v0.2
I (xxx) selftest: flash     : 16 MB (expect 16 MB)
I (xxx) selftest: psram     : 8 MB (Octal OK)          <-- 关键
I (xxx) selftest: heap psram: free xxxx KB, largest xxxx KB
I (xxx) bsp: I2C ready: SDA=1 SCL=2 100000 Hz
I (xxx) bsp: PCA9557 ready @0x19, out=0x05 (LCD_CS=1 PA_EN=0 PWDN=1)
I (xxx) bsp: I2S full-duplex ready: MCLK=38 BCLK=14 WS=13 DIN=12 DOUT=45 @16000 Hz
I (xxx) bsp: ES8311 ready @0x18 (playback)
I (xxx) bsp: ES7210 ready @0x41 (I2S 16-bit, non-TDM, MIC gain step=10)
I (xxx) bsp: SD mounted at /sdcard, xxxxx MB
I (xxx) selftest:   found 0x18  ES8311 (DAC/speaker)
I (xxx) selftest:   found 0x19  PCA9557 (IO expander)
I (xxx) selftest:   found 0x38  FT6336 (touch)
I (xxx) selftest:   found 0x41  ES7210 (ADC/mic)
```

**逐项判读**

| 现象 | 含义 | 处理 |
|---|---|---|
| `psram : 0 MB` 或 `2 MB` | Octal PSRAM 未启用 | 删 `sdkconfig` 重建；确认 `CONFIG_SPIRAM_MODE_OCT=y` |
| `PCA9557 失败` | I2C 不通或地址不对 | 查 SDA=1/SCL=2、上拉；扫不到 0x19 说明 IO 扩展未上电 |
| 缺 `ES7210` | 录音不可用 | 官方例程地址为 0x41；若仍扫不到，检查 AD0/AD1 与焊接 |
| 缺 `FT6336` | 触摸不可用 | 地址可能非 0x38，同上 |
| `SD mount failed` | 卡未插 / 非 FAT32 | 格式化为 FAT32，或先不插卡（自检仍能跑） |
| 背光不闪 | GPIO42 极性或 PCA9557 初值问题 | 对照 §二 表格核对 |

> 注：本阶段 ST7789 面板尚未初始化（`ui_manager` 在阶段二），**屏幕保持黑屏是正常的**，只有背光亮灭可观察。

### 3.7 首次编译说明

- 会自动拉取组件 `espressif/es8311`、`espressif/es7210`，需要联网；
- 首次编译约 3–8 分钟，之后增量编译 10–30 秒；
- 若组件下载失败（公司网络/代理），可先执行 `idf.py --no-ccache reconfigure` 重试，或手动把组件放到 `firmware/managed_components/`。

### 3.8 若编译报 codec API 不匹配

组件版本升级可能改签名。三处最可能变化，按报错改对应一行即可（都在 `board_bsp.c`）：

| 位置 | 可能的变化 |
|---|---|
| `es8311_create(port, addr)` | 新版本可能要求 `i2c_master_bus_handle_t` |
| `es7210_init(&i2c_cfg, &handle)` | 结构体字段名可能为 `i2c_port` / `bus_handle` |
| `es7210_set_gain(handle, db)` | 可能叫 `es7210_config_gain` |

> 也可锁定版本：`idf_component.yml` 里把 `^1.0.0` 改为确定版本如 `1.0.1`。

---

## 四、本阶段交付范围

| 项 | 状态 |
|---|---|
| 目录结构 | ✅ |
| `board_pins.h`（含来源与待核标注） | ✅ |
| `sdkconfig.defaults`（Octal PSRAM / FATFS LFN / LVGL） | ✅ |
| `board_bsp.c/h`（I2C / PCA9557 / 全双工 I2S / ES7210+ES8311 / SD / 背光 / 按键） | ✅ |
| 工程骨架（CMake / 分区表 / 组件依赖） | ✅（为可编译而补） |
| 硬件自检 `main.c` | ✅（为可上电验证而补） |

**尚未交付（等确认后继续）**：
`audio_recorder`、`network_client`、`ui_manager`、状态机 `main.c`、FastAPI 后端全套。

---

## 五、下一步（阶段二）预告

1. `audio_recorder.c`：`heap_caps_malloc(BSP_REC_PSRAM_BYTES, MALLOC_CAP_SPIRAM)` + 高优先级采集任务（RX 读 16bit 立体声 → 取主麦单声道 → 写 PSRAM）+ 44 字节 WAV 头 + 落盘 `/sdcard/dialect/`。
2. `network_client.c`：`esp_http_client` multipart 上传 + cJSON 解析 + `POST /feedback`。
3. `ui_manager.c`：LVGL 状态提示 / 录音指示 / 结果卡片 / ✓✗ 按钮。
4. 状态机整合：`IDLE → RECORDING → UPLOADING → DISPLAY_AND_EXECUTE → FEEDBACK`。

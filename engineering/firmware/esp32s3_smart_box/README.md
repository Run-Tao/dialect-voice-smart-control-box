# ESP32-S3 方言智控执行固件

这是电脑端识别程序与 ESP32-S3 执行端之间的最小可烧录固件。

边界：

- PC 负责麦克风、VAD、识别、拒识和对话决策；
- ESP32-S3 只接收命令、驱动灯和窗帘舵机，并返回 ACK/STATE；
- 不包含麦克风识别模型，不修改智能音箱固件；
- 默认引脚是可替换的临时配置，最终以硬件负责人接线表为准。

## 默认引脚

| 功能 | GPIO | 说明 |
|---|---:|---|
| 灯/继电器 | 4 | `DV_LIGHT_ACTIVE_HIGH=1` |
| 窗帘舵机 | 5 | 50 Hz PWM，开 90°、关 0° |
| 状态 LED | 6 | 每收到有效命令闪 20 ms |

所有引脚都可在 `platformio.ini` 的 `build_flags` 中覆盖。

硬件负责人需要填写的引脚、端口和验收项见
[`HARDWARE_HANDOFF.md`](HARDWARE_HANDOFF.md)。

> SG90 等舵机需使用独立 5 V 电源，并与 ESP32-S3 共地。不要直接用开发板
> 3.3 V 引脚给舵机供电，也不要把市电直接接到开发板或继电器低压侧。

当前镜像按 `esp32-s3-devkitc-1` 的 8 MB Flash、无 PSRAM 配置编译。固件本身
不使用 PSRAM，因此 N16R8 板也可运行，只会使用前 8 MB 分区；若硬件负责人
确认了具体开发板型号，可在 `platformio.ini` 中换成对应 board。

## 编译与烧录

安装 PlatformIO Core 后：

```powershell
cd D:\Run_Tao\DialectVoice_Engineering\firmware\esp32s3_smart_box
pio device list
pio run
pio run -t upload --upload-port COM5
pio device monitor --baud 115200 --port COM5
```

原生 USB 口使用默认环境：

```powershell
pio run -t upload --upload-port COM5
```

如果开发板只能通过 USB-UART 桥接口通信：

```powershell
pio run -e esp32-s3-uart -t upload --upload-port COM5
```

## 不安装 PlatformIO 时的直接烧录

`dist/` 内提供已经合并好 bootloader、分区表和应用的整包：

| 文件 | 适用串口 |
|---|---|
| `merged-usb.bin` | 开发板原生 USB |
| `merged-uart.bin` | USB-UART 桥接串口 |

先让开发板进入下载模式（通常是按住 BOOT、点一下 RESET、松开 BOOT），然后：

```powershell
python -m pip install esptool
python -m esptool --chip esp32s3 --port COM5 --baud 921600 `
  write_flash 0x0 dist\merged-usb.bin
```

如使用 UART 桥，换成 `dist\merged-uart.bin`。烧录完成后按一次 RESET，
以 115200 波特率打开串口。

也可以直接使用脚本：

```powershell
.\flash.ps1 -Port COM5 -Transport usb
# 或
.\flash.ps1 -Port COM5 -Transport uart
```

`dist/BUILD_INFO.json` 和 `dist/SHA256SUMS.txt` 记录构建版本与哈希。
这些镜像尚未在真实硬件上烧录验证，最终烧录端口、Flash 容量和引脚必须由
硬件负责人确认。

启动后应看到：

```text
BOOT,ESP32S3_DIALECTVOICE,READY,version=0.1.0
STATE,0,LIGHT=off,CURTAIN=closed
```

## 串口协议

PC -> ESP32-S3：

```text
CMD,<seq>,<intent>\n
```

允许的 intent：

```text
LIGHT_ON
LIGHT_OFF
CURTAIN_OPEN
CURTAIN_CLOSE
ALL_OFF
STATUS_ALL
```

正常回复：

```text
ACK,<seq>,OK
```

重复序号：

```text
ACK,<seq>,DUP
```

状态查询：

```text
ACK,<seq>,OK
STATE,<seq>,LIGHT=off,CURTAIN=closed
```

错误：

```text
ERR,<seq>,BAD_FORMAT
ERR,<seq>,BAD_SEQ
ERR,<seq>,UNKNOWN_CMD
```

## 手动验收

串口监视器依次发送：

```text
CMD,1,STATUS_ALL
CMD,2,LIGHT_ON
CMD,3,STATUS_ALL
CMD,4,CURTAIN_OPEN
CMD,5,STATUS_ALL
CMD,6,ALL_OFF
CMD,7,STATUS_ALL
```

预期最终状态：

```text
STATE,7,LIGHT=off,CURTAIN=closed
```

PC 端也可以运行：

```powershell
python D:\Run_Tao\DialectVoice_Engineering\scripts\esp32_serial_smoke.py --port COM5
```

默认只做状态查询；加 `--full` 才会实际驱动灯和窗帘。

## 与 PC 识别端联用

确认串口协议通过后，回到工程根目录启动识别：

```powershell
cd D:\Run_Tao\DialectVoice_Engineering
python scripts\run_monitor.py --executor serial --port COM5
```

此时 PC 负责听写和决策，ESP32-S3 负责执行并回 ACK/STATE。当前固件不包含
麦克风识别模型，不能脱离 PC 独立完成方言识别。

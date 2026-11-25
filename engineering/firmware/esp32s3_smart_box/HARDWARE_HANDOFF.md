# 硬件负责人交接单

软件侧已经提供可编译、可烧录的执行固件。硬件侧不需要修改 PC 协议，
只需要确认开发板、端口和引脚。

## 必须确认

| 项目 | 当前临时值 | 硬件负责人填写 |
|---|---|---|
| 开发板型号 | 通用 ESP32-S3-DevKitC-1 8 MB | |
| 电脑连接端口 | 未指定 | COM? |
| 原生 USB 或 UART 桥 | 两版固件都已提供 | |
| 灯/继电器控制 GPIO | 4 | |
| 继电器高有效/低有效 | 高有效 | |
| 窗帘舵机 GPIO | 5 | |
| 舵机开角度 | 90° | |
| 舵机关角度 | 0° | |
| 状态 LED GPIO | 6 | |
| 舵机独立 5 V 与共地 | 必须 | [ ] 已确认 |

## 硬件侧只需修改

优先修改 `platformio.ini` 中对应环境的 `build_flags`：

```ini
-D DV_LIGHT_PIN=4
-D DV_CURTAIN_SERVO_PIN=5
-D DV_STATUS_LED_PIN=6
-D DV_LIGHT_ACTIVE_HIGH=1
-D DV_SERVO_OPEN_ANGLE=90
-D DV_SERVO_CLOSE_ANGLE=0
```

不要修改 `CMD,<seq>,<intent>`、`ACK`、`STATE` 的文本格式，否则 PC 侧
`src/control.py` 与 `runtime/serial_protocol.py` 将无法对接。

## 烧录后验收

1. 按住 BOOT、点 RESET、松开 BOOT，进入下载模式。
2. 烧录 `dist/merged-usb.bin` 或 `dist/merged-uart.bin`。
3. 按 RESET，以 115200 打开串口。
4. 确认启动行包含 `version=0.1.0`。
5. 运行：

```powershell
python D:\Run_Tao\DialectVoice_Engineering\scripts\esp32_serial_smoke.py --port COM5
python D:\Run_Tao\DialectVoice_Engineering\scripts\esp32_serial_smoke.py --port COM5 --full
```

第一条只查询状态；第二条会实际驱动灯、窗帘和全部关闭。

## 返回给软件侧的信息

- 实际开发板型号与 Flash/PSRAM 配置；
- 实际 COM 口与使用的是原生 USB 还是 UART；
- 最终 GPIO 表和有效电平；
- 舵机实际方向是否正确；
- 六条命令的 ACK/STATE 验收结果；
- 若烧录失败，附 esptool 完整错误信息。

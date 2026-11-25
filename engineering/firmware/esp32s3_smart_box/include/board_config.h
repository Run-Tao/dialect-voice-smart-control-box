#pragma once

// 这些默认值只用于“先烧录、先跑通协议”。
// 实际开发板和接线由硬件负责人确认后，优先通过 platformio.ini 的
// build_flags 覆盖，不要在 main.cpp 里散落改引脚。

#ifndef DV_LIGHT_PIN
#define DV_LIGHT_PIN 4
#endif

#ifndef DV_CURTAIN_SERVO_PIN
#define DV_CURTAIN_SERVO_PIN 5
#endif

#ifndef DV_STATUS_LED_PIN
#define DV_STATUS_LED_PIN 6
#endif

#ifndef DV_LIGHT_ACTIVE_HIGH
#define DV_LIGHT_ACTIVE_HIGH 1
#endif

#ifndef DV_SERVO_OPEN_ANGLE
#define DV_SERVO_OPEN_ANGLE 90
#endif

#ifndef DV_SERVO_CLOSE_ANGLE
#define DV_SERVO_CLOSE_ANGLE 0
#endif

#define DV_SERIAL_BAUD 115200
#define DV_MAX_LINE_LEN 96
#define DV_SEEN_SEQ_SLOTS 64
#define DV_FIRMWARE_VERSION "0.1.0"

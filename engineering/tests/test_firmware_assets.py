# -*- coding: utf-8 -*-
"""ESP32-S3 可烧录固件的静态契约测试。

不依赖 PlatformIO 或真实串口：这些测试只保证工程入口、命令表和默认环境
没有被误删，真正的编译由 `pio run` 单独验证。
"""
from __future__ import annotations

import re
from pathlib import Path

from runtime.serial_protocol import ALLOWED_INTENTS

ROOT = Path(__file__).resolve().parents[1]
FW = ROOT / "firmware" / "esp32s3_smart_box"


def test_firmware_project_files_exist():
    for rel in (
        "platformio.ini",
        "include/board_config.h",
        "src/main.cpp",
        "README.md",
        "HARDWARE_HANDOFF.md",
        "flash.ps1",
    ):
        assert (FW / rel).is_file(), rel


def test_platformio_targets_esp32s3_usb_and_uart():
    text = (FW / "platformio.ini").read_text(encoding="utf-8")
    assert "board = esp32-s3-devkitc-1" in text
    assert "[env:esp32-s3-usb]" in text
    assert "[env:esp32-s3-uart]" in text
    assert "ARDUINO_USB_CDC_ON_BOOT=1" in text
    assert "ARDUINO_USB_CDC_ON_BOOT=0" in text


def test_firmware_intents_match_pc_protocol():
    text = (FW / "src" / "main.cpp").read_text(encoding="utf-8")
    got = set(re.findall(r'intent == "([A-Z_]+)"', text))
    assert got == ALLOWED_INTENTS


def test_default_pins_are_configurable_and_not_usb_pins():
    text = (FW / "include" / "board_config.h").read_text(encoding="utf-8")
    pins = {
        int(m.group(2))
        for m in re.finditer(r"#define (DV_[A-Z_]+_PIN) (\d+)", text)
    }
    assert len(pins) == 3
    assert not ({19, 20} & pins), "GPIO19/20 是 ESP32-S3 原生 USB"
    assert "DV_LIGHT_PIN" in text
    assert "DV_CURTAIN_SERVO_PIN" in text


def test_pc_smoke_script_exists():
    assert (ROOT / "scripts" / "esp32_serial_smoke.py").is_file()

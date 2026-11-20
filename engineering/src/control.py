# -*- coding: utf-8 -*-
"""执行层：把识别出的命令落到真实设备或模拟设备。

通道：
- ``LoggingExecutor``         只记录，用于无硬件演示；
- ``MockEsp32Executor``       走串口协议的状态机，模拟 ESP32-S3；
- ``SerialExecutor``          真串口，接真实 ESP32-S3；
- ``RelayExecutor``           树莓派 GPIO 继电器直控灯；
- ``HomeAssistantExecutor``   Home Assistant REST 服务调用（S5 状态通道）；
- ``MqttExecutor``            MQTT 发布（S5 状态通道）。

控制类命令（查状态）只读状态、不触发动作。

ACK 闭环（S4b）
--------------
执行分两步：**下发**（``dispatch``）与**等回执**（``wait_ack``）。
旧调用点用 ``execute``（= 下发 + 等回执），语义不变；
新管线用 ``dispatch`` 立即返回，让动作先于播报发生，回执到了再播「已完成」。
``async_ack`` 为 False 的执行器在 ``dispatch`` 里就同步拿到回执，
``wait_ack`` 只是把结果再取一次。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from .commands import COMMANDS, INITIAL_STATE


class CommandExecutor:
    """执行器基类。

    子类至少实现 ``dispatch``；需要异步回执的实现 ``wait_ack`` 并把
    ``async_ack`` 置 True。
    """

    async_ack = False

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def dispatch(self, command: str) -> Dict[str, object]:
        """下发命令并立即返回（不等回执）。"""
        raise NotImplementedError

    def wait_ack(self, timeout: Optional[float] = None) -> Dict[str, object]:
        """等执行器回执；返回 ``{'ok': bool, 'detail': str}``。"""
        return {'ok': True, 'detail': 'sync'}

    def execute(self, command: str) -> Dict[str, object]:
        """同步执行：下发 + 等回执（旧调用点语义不变）。"""
        info = dict(self.dispatch(command))
        if self.async_ack:
            ack = self.wait_ack()
            info.update(ack)
        return info

    # ------------------------------------------------------------------
    def _write_log(self, command: str, status: str, detail: str = '') -> None:
        if not self.log_path:
            return
        row = {'command': command, 'status': status, 'detail': detail}
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


class LoggingExecutor(CommandExecutor):
    """不接硬件，只打印并记录，用于软件闭环演示。"""

    def __init__(self, log_path: Optional[str] = None):
        super().__init__(log_path)
        self.state = dict(INITIAL_STATE)

    def dispatch(self, command: str) -> Dict[str, object]:
        entry = COMMANDS.get(command)
        if entry is None:
            self._write_log(command, 'UNKNOWN')
            raise ValueError(f'未知命令: {command}')
        if entry['action'] != 'query':
            if entry['actuator'] == 'all':
                self.state = dict(INITIAL_STATE)
            else:
                self.state[entry['actuator']] = entry['action']
        self._write_log(command, 'OK', json.dumps(self.state, ensure_ascii=False))
        print(f'[执行] {command} -> {self.state}')
        return {'command': command, 'state': dict(self.state), 'ok': True,
                'detail': 'logged'}


class MockEsp32Executor(CommandExecutor):
    """用串口协议的状态机模拟 ESP32-S3，验证协议闭环。"""

    def __init__(self, log_path: Optional[str] = None):
        super().__init__(log_path)
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from runtime.serial_protocol import MockEsp32Controller
        self.controller = MockEsp32Controller()
        self.seq = 0

    def dispatch(self, command: str) -> Dict[str, object]:
        self.seq += 1
        response = self.controller.handle_line(f'CMD,{self.seq},{command}')
        self._write_log(command, response.strip())
        print(f'[模拟ESP32] {command} -> {response.strip()}')
        return {'command': command, 'response': response.strip(),
                'state': dict(self.controller.state), 'ok': True,
                'detail': response.strip()}


class SerialExecutor(CommandExecutor):
    """真实串口，向 ESP32-S3 下发命令并等待 ACK。

    ``dispatch`` 只写串口（动作立即生效），``wait_ack`` 读回执行；
    两者分开才能做到「动作优先、播报随后」。
    """

    async_ack = True

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0,
                 log_path: Optional[str] = None):
        super().__init__(log_path)
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from runtime.serial_protocol import build_command
        try:
            import serial
        except ImportError as exc:
            raise ImportError('需要安装 pyserial: pip install pyserial') from exc
        self.build_command = build_command
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.seq = 0
        self._default_timeout = float(timeout)

    def dispatch(self, command: str) -> Dict[str, object]:
        self.seq += 1
        self.ser.reset_input_buffer()
        self.ser.write(self.build_command(self.seq, command).encode('ascii'))
        return {'command': command, 'seq': self.seq, 'ok': True, 'detail': 'sent'}

    def wait_ack(self, timeout: Optional[float] = None) -> Dict[str, object]:
        old = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = float(timeout)
        try:
            line = self.ser.readline().decode('ascii', errors='replace').strip()
        finally:
            self.ser.timeout = old
        if not line:
            self._write_log(self.command if hasattr(self, 'command') else '', 'TIMEOUT')
            raise TimeoutError('ESP32 未返回 ACK')
        self._write_log(line.split(',')[2] if line.count(',') >= 2 else '', line)
        print(f'[ESP32] ACK -> {line}')
        return {'ok': True, 'detail': line}

    def close(self) -> None:
        if getattr(self, 'ser', None) is not None:
            self.ser.close()


class RelayExecutor(CommandExecutor):
    """树莓派 GPIO 继电器直控灯，作为智能音箱联动失败时的备用通道。"""

    def __init__(self, pin: int = 17, mock: bool = True,
                 log_path: Optional[str] = None):
        super().__init__(log_path)
        self.pin = pin
        self.mock = mock
        self.state = 'off'
        self.gpio = None
        if not mock:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
                self.gpio = GPIO
            except Exception as exc:
                print('GPIO 不可用，回退到 mock:', exc)
                self.mock = True

    def dispatch(self, command: str) -> Dict[str, object]:
        entry = COMMANDS.get(command)
        if entry is None:
            raise ValueError(f'未知命令: {command}')
        if entry['action'] == 'query':
            self._write_log(command, 'QUERY', self.state)
            return {'command': command, 'state': self.state, 'ok': True,
                    'detail': 'query'}
        target = entry['action'] if entry['actuator'] == 'light' else self.state
        if entry['actuator'] == 'light':
            self.state = entry['action']
            target = self.state
        if self.mock:
            print('[mock relay]', target)
        else:
            self.gpio.output(self.pin, 1 if target == 'on' else 0)
        self._write_log(command, 'OK', target)
        return {'command': command, 'state': target, 'ok': True, 'detail': 'ok'}


class HomeAssistantExecutor(CommandExecutor):
    """Home Assistant REST 服务调用（S5 通道）。

    ``services`` 形如::

        {'LIGHT_ON': {'service': 'light.turn_on', 'entity_id': 'light.kitchen'},
         'CURTAIN_OPEN': {'service': 'cover.open_cover', 'entity_id': 'cover.living'}}

    未配置的命令直接回 ``ok=False``，不猜服务名。
    """

    def __init__(self, url: str, token: str,
                 services: Optional[Dict[str, Dict[str, str]]] = None,
                 log_path: Optional[str] = None, timeout: float = 3.0):
        super().__init__(log_path)
        if not url or not token:
            raise ValueError('Home Assistant 执行器需要 url 与 token')
        self.url = str(url).rstrip('/')
        self.token = str(token)
        self.services = dict(services or {})
        self.timeout = float(timeout)

    def dispatch(self, command: str) -> Dict[str, object]:
        spec = self.services.get(command)
        if not spec:
            self._write_log(command, 'NO_SERVICE')
            return {'command': command, 'ok': False, 'detail': 'no_service_mapping'}
        service = str(spec.get('service', ''))
        payload: Dict[str, object] = {}
        if spec.get('entity_id'):
            payload['entity_id'] = spec['entity_id']
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f'{self.url}/api/services/{service}', data=data, method='POST',
            headers={'Authorization': f'Bearer {self.token}',
                     'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode('utf-8', errors='replace')
                code = int(resp.status)
        except urllib.error.HTTPError as exc:
            self._write_log(command, 'HTTP_ERROR', str(exc))
            return {'command': command, 'ok': False, 'detail': f'http_{exc.code}'}
        except Exception as exc:
            self._write_log(command, 'ERROR', str(exc))
            return {'command': command, 'ok': False, 'detail': str(exc)}
        self._write_log(command, 'OK', f'{code} {body[:120]}')
        return {'command': command, 'ok': 200 <= code < 300,
                'detail': f'http_{code}', 'response': body[:200]}


class MqttExecutor(CommandExecutor):
    """MQTT 发布执行通道（S5）。

    需要 ``paho-mqtt``；未安装时构造即报错。
    """

    def __init__(self, host: str, port: int = 1883,
                 topic: str = 'dialectvoice/command',
                 username: Optional[str] = None, password: Optional[str] = None,
                 qos: int = 0, keepalive: int = 60,
                 log_path: Optional[str] = None):
        super().__init__(log_path)
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError('MQTT 执行器需要 paho-mqtt: pip install paho-mqtt') from exc
        if not host:
            raise ValueError('MQTT 执行器需要 host')
        self.host = str(host)
        self.port = int(port)
        self.topic = str(topic)
        self.qos = int(qos)
        self._client = mqtt.Client()
        if username:
            self._client.username_pw_set(username, password)
        self._client.connect(self.host, self.port, keepalive=keepalive)
        self._client.loop_start()

    def dispatch(self, command: str) -> Dict[str, object]:
        payload = json.dumps({'command': command, 'ts': time.time()},
                             ensure_ascii=False)
        info = self._client.publish(self.topic, payload, qos=self.qos)
        self._write_log(command, 'PUBLISHED', payload)
        return {'command': command, 'ok': True, 'detail': 'published',
                'mid': getattr(info, 'mid', None)}

    def close(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass


def build_executor(kind: str, log_path: Optional[str] = None, **kwargs):
    """按名字造执行器；``kind`` 取 log / mock_esp32 / serial / relay / ha / mqtt。"""
    kind = (kind or 'log').strip().lower()
    if kind == 'log':
        return LoggingExecutor(log_path=log_path)
    if kind in ('mock_esp32', 'mock'):
        return MockEsp32Executor(log_path=log_path)
    if kind == 'serial':
        return SerialExecutor(port=kwargs.get('port', ''),
                              baudrate=kwargs.get('baudrate', 115200),
                              timeout=kwargs.get('timeout', 1.0),
                              log_path=log_path)
    if kind == 'relay':
        return RelayExecutor(pin=kwargs.get('pin', 17),
                              mock=kwargs.get('mock', True), log_path=log_path)
    if kind in ('ha', 'home_assistant'):
        return HomeAssistantExecutor(url=kwargs.get('url', ''),
                                     token=kwargs.get('token', ''),
                                     services=kwargs.get('services'),
                                     log_path=log_path,
                                     timeout=kwargs.get('timeout', 3.0))
    if kind == 'mqtt':
        return MqttExecutor(host=kwargs.get('host', ''), port=kwargs.get('port', 1883),
                             topic=kwargs.get('topic', 'dialectvoice/command'),
                             username=kwargs.get('username'),
                             password=kwargs.get('password'),
                             qos=kwargs.get('qos', 0), log_path=log_path)
    raise ValueError(f'未知执行器类型: {kind}')

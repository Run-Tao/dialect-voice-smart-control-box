# -*- coding: utf-8 -*-
"""设备状态源：mock 状态机 / Home Assistant REST / MQTT。

对应 v5 方案设计 §6.3 的「状态源」一节与 §11 的 R-9（HA 无法在课程环境部署
时用 mock 状态机，结论需标注为「仿真状态源」）。

三种源都实现同一个接口 ``read() -> DeviceState``，因此决策层
（``decision.py``）不关心状态从哪来。**离线默认可跑**：``mock`` 不需要任何
外部依赖，这是课程环境里唯一能跑通的选项，所以它是默认值。

状态取值与 ``src/commands.py`` 的 ``INITIAL_STATE`` 保持一致：
灯 ``on`` / ``off``，窗帘 ``open`` / ``closed``。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ['DeviceState', 'StateSource', 'MockStateSource',
           'HomeAssistantStateSource', 'MqttStateSource', 'build_state_source']

# 命令 -> (actuator, action)。与 src/commands.py 的 COMMANDS 表必须一致；
# tests/test_state.py 会断言两者相同，防止两处各改一处。
COMMAND_TABLE: Dict[str, tuple] = {
    'LIGHT_ON': ('light', 'on'),
    'LIGHT_OFF': ('light', 'off'),
    'CURTAIN_OPEN': ('curtain', 'open'),
    'CURTAIN_CLOSE': ('curtain', 'close'),
    'ALL_OFF': ('all', 'off'),
    'STATUS_ALL': ('none', 'query'),
}


@dataclass
class DeviceState:
    """设备状态快照。"""

    light: str = 'off'          # on / off
    curtain: str = 'closed'     # open / closed

    def as_dict(self) -> dict:
        return {'light': self.light, 'curtain': self.curtain}

    def get(self, actuator: str) -> str:
        return getattr(self, actuator, '')


class StateSource:
    """状态源接口。"""

    def read(self) -> DeviceState:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockStateSource(StateSource):
    """内存状态机：执行命令后状态随之变化，供演示与测试离线跑通。"""

    def __init__(self, state: Optional[DeviceState] = None,
                 command_table: Optional[Dict[str, tuple]] = None):
        self.state = state or DeviceState()
        self.table = dict(command_table or COMMAND_TABLE)

    def read(self) -> DeviceState:
        return DeviceState(light=self.state.light, curtain=self.state.curtain)

    def apply(self, command: str) -> DeviceState:
        """执行一条命令后更新状态（查询类命令不改状态）。"""
        entry = self.table.get(command)
        if entry is None:
            raise ValueError(f'未知命令: {command}')
        actuator, action = entry
        if actuator == 'all':
            self.state = DeviceState(light='off', curtain='closed')
        elif actuator in ('light', 'curtain'):
            setattr(self.state, actuator, action)
        return self.read()


def _map_ha_state(actuator: str, raw: str) -> str:
    """把 HA 的实体状态翻译成本项目的取值。

    HA 里灯是 ``on``/``off``，cover（窗帘）除了 ``open``/``closed`` 还可能出现
    ``opening``/``closing``，后者按「正在动」处理：取目标态的方向不明确时
    归到 ``closed``（保守：先验更倾向「关」，避免把「开」当成冗余动作）。
    """
    raw = (raw or '').strip().lower()
    if actuator == 'light':
        return 'on' if raw == 'on' else 'off'
    if raw in ('open', 'opening'):
        return 'open'
    if raw in ('closed', 'closing'):
        return 'closed'
    return 'closed'


class HomeAssistantStateSource(StateSource):
    """从 Home Assistant REST API 读实体状态。

    只用标准库 ``urllib``，不引入 requests；HA 不可达时抛异常由调用方降级。
    """

    def __init__(self, url: str, token: str,
                 entities: Optional[Dict[str, str]] = None,
                 timeout: float = 3.0):
        if not url or not token:
            raise ValueError('Home Assistant 需要 url 与 token')
        self.url = str(url).rstrip('/')
        self.token = str(token)
        # entities: {'light': 'light.kitchen', 'curtain': 'cover.living_room'}
        self.entities = dict(entities or {})
        self.timeout = float(timeout)

    def _get_entity(self, entity_id: str) -> dict:
        req = urllib.request.Request(
            f'{self.url}/api/states/{entity_id}',
            headers={'Authorization': f'Bearer {self.token}',
                     'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def read(self) -> DeviceState:
        if not self.entities:
            raise ValueError('未配置 HA 实体映射')
        out = {}
        for actuator, entity_id in self.entities.items():
            data = self._get_entity(entity_id)
            out[actuator] = _map_ha_state(actuator, str(data.get('state', '')))
        return DeviceState(light=out.get('light', 'off'),
                           curtain=out.get('curtain', 'closed'))


class MqttStateSource(StateSource):
    """从 MQTT broker 读状态（保留最近一次 payload）。

    需要 ``paho-mqtt``；未安装时构造即报错，不静默降级。
    """

    def __init__(self, host: str, port: int = 1883,
                 topics: Optional[Dict[str, str]] = None,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 keepalive: int = 60, wait_timeout: float = 2.0):
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError('MQTT 状态源需要 paho-mqtt: pip install paho-mqtt') from exc
        if not host:
            raise ValueError('MQTT 需要 host')
        self.host = str(host)
        self.port = int(port)
        # topics: {'light': 'home/light/state', 'curtain': 'home/curtain/state'}
        self.topics = dict(topics or {})
        self.wait_timeout = float(wait_timeout)
        self._latest: Dict[str, str] = {}
        self._client = mqtt.Client()
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_message = self._on_message
        self._client.connect(self.host, self.port, keepalive=keepalive)
        self._client.loop_start()
        for topic in self.topics.values():
            self._client.subscribe(topic)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = msg.payload.decode('utf-8').strip()
        except Exception:
            payload = ''
        value = payload
        if payload.startswith('{'):
            try:
                data = json.loads(payload)
                value = str(data.get('state', data.get('value', payload)))
            except Exception:
                value = payload
        for actuator, topic in self.topics.items():
            if topic == msg.topic:
                self._latest[actuator] = value

    def read(self) -> DeviceState:
        if not self._latest:
            import time
            time.sleep(self.wait_timeout)
        out = {a: _map_ha_state(a, v) for a, v in self._latest.items()}
        return DeviceState(light=out.get('light', 'off'),
                           curtain=out.get('curtain', 'closed'))

    def close(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass


def build_state_source(kind: str = 'mock', **kwargs) -> StateSource:
    """按名字构造状态源；``kind`` 取 ``mock`` / ``home_assistant`` / ``mqtt``。"""
    kind = (kind or 'mock').strip().lower()
    if kind == 'mock':
        return MockStateSource(command_table=kwargs.get('command_table'))
    if kind in ('home_assistant', 'ha'):
        return HomeAssistantStateSource(
            url=kwargs.get('url', ''), token=kwargs.get('token', ''),
            entities=kwargs.get('entities'), timeout=kwargs.get('timeout', 3.0))
    if kind == 'mqtt':
        return MqttStateSource(
            host=kwargs.get('host', ''), port=kwargs.get('port', 1883),
            topics=kwargs.get('topics'), username=kwargs.get('username'),
            password=kwargs.get('password'),
            wait_timeout=kwargs.get('wait_timeout', 2.0))
    raise ValueError(f'未知状态源: {kind}')

# -*- coding: utf-8 -*-
"""执行层测试：Serial / Relay / HomeAssistant / MQTT / build_executor。

这四类执行器与工厂此前零测试（PROGRESS.md「测试盲区」第 5 条）。
本文件只断言对外契约——返回结构、设备状态迁移、写出的协议字节、日志行——
全部用 fake serial / fake paho / monkeypatch urlopen，不碰真实串口、网络与 GPIO。
"""
from __future__ import annotations

import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import control
from src.commands import INITIAL_STATE, resulting_state
from runtime.state import DeviceState
from runtime.decision import state_weight


# ----------------------------------------------------------------------
# fake 依赖
# ----------------------------------------------------------------------
class FakeSerial:
    """最小串口替身：记录写入、按队列返回 readline。"""

    def __init__(self, port, baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.written = []
        self.responses = []
        self.reset_count = 0
        self.closed = False

    def reset_input_buffer(self):
        self.reset_count += 1

    def write(self, payload):
        self.written.append(payload)
        return len(payload)

    def readline(self):
        return self.responses.pop(0) if self.responses else b''

    def close(self):
        self.closed = True


@pytest.fixture
def fake_serial(monkeypatch):
    module = types.ModuleType('serial')
    module.Serial = FakeSerial
    monkeypatch.setitem(sys.modules, 'serial', module)
    return module


def install_fake_paho(monkeypatch):
    """伪造 paho.mqtt.client，返回 FakeClient 类供断言。"""
    paho = types.ModuleType('paho')
    paho.__path__ = []                     # 让 import paho.mqtt.client 认得它是包
    mqtt_pkg = types.ModuleType('paho.mqtt')
    mqtt_pkg.__path__ = []
    client_mod = types.ModuleType('paho.mqtt.client')

    class FakeClient:
        def __init__(self):
            self.connected = None
            self.published = []
            self.creds = None
            self.loop_started = False
            self.stopped = False
            self.disconnected = False

        def username_pw_set(self, username, password=None):
            self.creds = (username, password)

        def connect(self, host, port, keepalive=60):
            self.connected = (host, port, keepalive)
            return 0

        def loop_start(self):
            self.loop_started = True

        def publish(self, topic, payload, qos=0):
            self.published.append((topic, payload, qos))
            return types.SimpleNamespace(mid=len(self.published))

        def loop_stop(self):
            self.stopped = True

        def disconnect(self):
            self.disconnected = True

    client_mod.Client = FakeClient
    paho.mqtt = mqtt_pkg
    mqtt_pkg.client = client_mod
    monkeypatch.setitem(sys.modules, 'paho', paho)
    monkeypatch.setitem(sys.modules, 'paho.mqtt', mqtt_pkg)
    monkeypatch.setitem(sys.modules, 'paho.mqtt.client', client_mod)
    return FakeClient


class FakeResponse:
    def __init__(self, status=200, body=b'[]'):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ----------------------------------------------------------------------
# LoggingExecutor
# ----------------------------------------------------------------------
def test_logging_executor_updates_state_and_writes_jsonl(tmp_path):
    log = tmp_path / 'exec.jsonl'
    ex = control.LoggingExecutor(log_path=str(log))
    info = ex.dispatch('LIGHT_ON')
    assert info['ok'] is True
    assert ex.state['light'] == 'on'
    rows = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    assert rows[-1]['command'] == 'LIGHT_ON'
    assert rows[-1]['status'] == 'OK'


def test_logging_executor_all_off_resets_state():
    ex = control.LoggingExecutor()
    ex.dispatch('LIGHT_ON')
    ex.dispatch('CURTAIN_OPEN')
    ex.dispatch('ALL_OFF')
    assert ex.state == INITIAL_STATE


def test_curtain_state_value_matches_decision_expectation():
    """CURTAIN_CLOSE 执行后 state 必须是 'closed' 而非 action 字面 'close'。

    2026-11-19 审查发现的真实缺陷：control.py 把 action='close' 直接写进 state，
    而 INITIAL_STATE / runtime.state.DeviceState / runtime.decision.state_weight
    都用 'closed'，导致「窗帘已关」的状态先验冗余判断失效。
    """
    ex = control.LoggingExecutor()
    ex.dispatch('CURTAIN_CLOSE')
    assert ex.state['curtain'] == 'closed'
    assert ex.state['curtain'] != 'close'
    assert resulting_state('CURTAIN_CLOSE') == 'closed'
    assert resulting_state('CURTAIN_OPEN') == 'open'
    assert resulting_state('LIGHT_ON') == 'on'
    # 跨模块：用执行后的 state 构造 DeviceState，state_weight 应判为冗余（返回 r<1）
    ds = DeviceState(light=ex.state['light'], curtain=ex.state['curtain'])
    w_close = state_weight('CURTAIN_CLOSE', ds, redundancy_weight=0.25)
    w_open = state_weight('CURTAIN_OPEN', ds, redundancy_weight=0.25)
    assert w_close < 1.0, '窗帘已关时重复 close 应判冗余'
    assert w_open == 1.0, '窗帘已关时 open 不是冗余'


def test_logging_executor_status_does_not_mutate():
    ex = control.LoggingExecutor()
    ex.dispatch('LIGHT_ON')
    before = dict(ex.state)
    info = ex.dispatch('STATUS_ALL')
    assert info['state'] == before
    assert ex.state == before


def test_logging_executor_unknown_logs_then_raises(tmp_path):
    log = tmp_path / 'x.jsonl'
    ex = control.LoggingExecutor(log_path=str(log))
    with pytest.raises(ValueError):
        ex.dispatch('NOT_A_COMMAND')
    assert 'UNKNOWN' in log.read_text(encoding='utf-8')


def test_execute_skips_wait_ack_for_sync_executor(monkeypatch):
    ex = control.LoggingExecutor()

    def boom(*args, **kwargs):
        raise AssertionError('同步执行器不应调用 wait_ack')

    monkeypatch.setattr(ex, 'wait_ack', boom)
    assert ex.execute('LIGHT_ON')['ok'] is True


# ----------------------------------------------------------------------
# MockEsp32Executor
# ----------------------------------------------------------------------
def test_mock_esp32_applies_command_and_acks(tmp_path):
    ex = control.MockEsp32Executor(log_path=str(tmp_path / 'm.jsonl'))
    info = ex.dispatch('LIGHT_ON')
    assert info['ok'] is True
    assert info['response'].startswith('ACK,1')
    assert ex.controller.state['light'] == 'on'


def test_mock_esp32_seq_increases_and_status_returns_state():
    ex = control.MockEsp32Executor()
    ex.dispatch('LIGHT_ON')
    info = ex.dispatch('STATUS_ALL')
    assert ex.seq == 2
    assert 'STATE,2,' in info['response']


# ----------------------------------------------------------------------
# SerialExecutor
# ----------------------------------------------------------------------
def test_serial_dispatch_writes_protocol_line(fake_serial):
    ex = control.SerialExecutor(port='COM_TEST', seq_start=0)
    info = ex.dispatch('LIGHT_ON')
    assert info['seq'] == 1
    assert ex.ser.written == [b'CMD,1,LIGHT_ON\n']
    assert ex.ser.reset_count == 1


def test_serial_wait_ack_returns_line_and_logs_real_command(fake_serial, tmp_path):
    log = tmp_path / 's.jsonl'
    ex = control.SerialExecutor(port='COM_TEST', log_path=str(log))
    ex.dispatch('ALL_OFF')
    ex.ser.responses.append(b'ACK,1,OK\n')
    assert ex.wait_ack() == {'ok': True, 'detail': 'ACK,1,OK', 'status': 'OK'}
    row = json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
    assert row['command'] == 'ALL_OFF'          # 回归：此前恒为空串
    assert 'ACK,1,OK' in row['detail']


def test_serial_wait_ack_timeout_logs_real_command(fake_serial, tmp_path):
    log = tmp_path / 's.jsonl'
    ex = control.SerialExecutor(port='COM_TEST', log_path=str(log))
    ex.dispatch('CURTAIN_CLOSE')
    with pytest.raises(TimeoutError):
        ex.wait_ack()
    row = json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
    assert row['status'] == 'TIMEOUT'
    assert row['command'] == 'CURTAIN_CLOSE'


def test_serial_wait_ack_restores_timeout(fake_serial):
    ex = control.SerialExecutor(port='COM_TEST', timeout=1.0)
    ex.dispatch('LIGHT_ON')
    ex.ser.responses.append(b'ACK,1,OK\n')
    ex.wait_ack(timeout=0.25)
    assert ex.ser.timeout == pytest.approx(1.0)


def test_serial_execute_is_dispatch_plus_ack(fake_serial):
    ex = control.SerialExecutor(port='COM_TEST')
    ex.ser.responses.append(b'ACK,1,OK\n')
    info = ex.execute('LIGHT_ON')
    assert info['ok'] is True
    assert info['detail'] == 'ACK,1,OK'


def test_serial_unknown_command_raises_before_write(fake_serial):
    ex = control.SerialExecutor(port='COM_TEST')
    with pytest.raises(ValueError):
        ex.dispatch('NOT_A_COMMAND')
    assert ex.ser.written == []


# ----------------------------------------------------------------------
# RelayExecutor
# ----------------------------------------------------------------------
def test_relay_mock_tracks_light_state():
    ex = control.RelayExecutor(pin=17, mock=True)
    assert ex.state == 'off'
    ex.dispatch('LIGHT_ON')
    assert ex.state == 'on'


def test_relay_all_off_actually_turns_light_off():
    """回归：ALL_OFF 原先 target 取旧状态，继电器等于没执行。"""
    ex = control.RelayExecutor(pin=17, mock=True)
    ex.dispatch('LIGHT_ON')
    info = ex.dispatch('ALL_OFF')
    assert ex.state == 'off'
    assert info['state'] == 'off'


def test_relay_ignores_curtain_commands():
    ex = control.RelayExecutor(pin=17, mock=True)
    ex.dispatch('LIGHT_ON')
    ex.dispatch('CURTAIN_OPEN')
    assert ex.state == 'on'


def test_relay_status_is_query_without_mutation():
    ex = control.RelayExecutor(pin=17, mock=True)
    ex.dispatch('LIGHT_ON')
    info = ex.dispatch('STATUS_ALL')
    assert info['detail'] == 'query'
    assert ex.state == 'on'


def test_relay_unknown_command_raises():
    ex = control.RelayExecutor(pin=17, mock=True)
    with pytest.raises(ValueError):
        ex.dispatch('NOT_A_COMMAND')


def test_relay_falls_back_to_mock_without_gpio():
    ex = control.RelayExecutor(pin=17, mock=False)
    assert ex.mock is True          # Windows / 未装 RPi.GPIO


# ----------------------------------------------------------------------
# HomeAssistantExecutor
# ----------------------------------------------------------------------
def test_ha_requires_url_and_token():
    with pytest.raises(ValueError):
        control.HomeAssistantExecutor(url='', token='x')
    with pytest.raises(ValueError):
        control.HomeAssistantExecutor(url='http://ha', token='')


def test_ha_unmapped_command_returns_false_and_logs(tmp_path):
    log = tmp_path / 'ha.jsonl'
    ex = control.HomeAssistantExecutor('http://ha:8123', 'tok', services={},
                                       log_path=str(log))
    info = ex.dispatch('LIGHT_ON')
    assert info['ok'] is False
    assert info['detail'] == 'no_service_mapping'
    assert 'NO_SERVICE' in log.read_text(encoding='utf-8')


def test_ha_posts_service_with_bearer_and_entity(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured['url'] = req.full_url
        captured['method'] = req.get_method()
        captured['auth'] = req.get_header('Authorization')
        captured['body'] = json.loads(req.data.decode('utf-8'))
        return FakeResponse(200, b'[{"entity_id":"light.kitchen"}]')

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    ex = control.HomeAssistantExecutor(
        'http://ha:8123/', 'tok',
        services={'LIGHT_ON': {'service': 'light.turn_on',
                               'entity_id': 'light.kitchen'}})
    info = ex.dispatch('LIGHT_ON')
    assert info['ok'] is True
    assert info['detail'] == 'http_200'
    assert captured['url'] == 'http://ha:8123/api/services/light.turn_on'
    assert captured['method'] == 'POST'
    assert captured['auth'] == 'Bearer tok'
    assert captured['body'] == {'entity_id': 'light.kitchen'}


def test_ha_http_error_is_returned_not_raised(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, 'nope', {}, None)

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    ex = control.HomeAssistantExecutor(
        'http://ha', 'tok', services={'LIGHT_ON': {'service': 'light.turn_on'}})
    info = ex.dispatch('LIGHT_ON')
    assert info['ok'] is False
    assert info['detail'] == 'http_404'


def test_ha_network_error_is_caught(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise OSError('boom')

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    ex = control.HomeAssistantExecutor(
        'http://ha', 'tok', services={'LIGHT_ON': {'service': 'light.turn_on'}})
    info = ex.dispatch('LIGHT_ON')
    assert info['ok'] is False
    assert 'boom' in info['detail']


# ----------------------------------------------------------------------
# MqttExecutor
# ----------------------------------------------------------------------
def test_mqtt_publishes_json_payload(monkeypatch):
    install_fake_paho(monkeypatch)
    ex = control.MqttExecutor(host='broker', port=1883, topic='dv/cmd')
    assert ex._client.connected == ('broker', 1883, 60)
    assert ex._client.loop_started is True
    info = ex.dispatch('LIGHT_ON')
    assert info['ok'] is True
    assert info['mid'] == 1
    topic, payload, qos = ex._client.published[-1]
    assert topic == 'dv/cmd'
    assert json.loads(payload)['command'] == 'LIGHT_ON'
    assert qos == 0


def test_mqtt_credentials_are_applied(monkeypatch):
    install_fake_paho(monkeypatch)
    ex = control.MqttExecutor(host='broker', username='u', password='p')
    assert ex._client.creds == ('u', 'p')


def test_mqtt_close_stops_loop(monkeypatch):
    install_fake_paho(monkeypatch)
    ex = control.MqttExecutor(host='broker')
    ex.close()
    assert ex._client.stopped is True
    assert ex._client.disconnected is True


def test_mqtt_requires_host(monkeypatch):
    install_fake_paho(monkeypatch)
    with pytest.raises(ValueError):
        control.MqttExecutor(host='')


# ----------------------------------------------------------------------
# build_executor
# ----------------------------------------------------------------------
def test_build_executor_dispatches_known_kinds(fake_serial):
    assert isinstance(control.build_executor('log'), control.LoggingExecutor)
    assert isinstance(control.build_executor('mock'), control.MockEsp32Executor)
    assert isinstance(control.build_executor('mock_esp32'), control.MockEsp32Executor)
    relay = control.build_executor('relay', pin=5)
    assert isinstance(relay, control.RelayExecutor)
    assert relay.pin == 5
    serial = control.build_executor('serial', port='COM9', baudrate=9600)
    assert serial.ser.port == 'COM9'
    assert serial.ser.baudrate == 9600


def test_build_executor_ha_passes_services():
    ex = control.build_executor('ha', url='http://ha', token='tok',
                                services={'LIGHT_ON': {'service': 'light.turn_on'}})
    assert ex.services['LIGHT_ON']['service'] == 'light.turn_on'


def test_build_executor_unknown_kind_raises():
    with pytest.raises(ValueError):
        control.build_executor('not_a_channel')


def test_build_executor_defaults_to_log():
    assert isinstance(control.build_executor(''), control.LoggingExecutor)
    assert isinstance(control.build_executor(None), control.LoggingExecutor)


# ----------------------------------------------------------------------
# seq 跨进程不撞 DUP（固件保留 64 个 seq 判重）
# ----------------------------------------------------------------------
def test_serial_seq_uses_injected_start(fake_serial):
    ex = control.SerialExecutor('COM9', seq_start=5000)
    assert ex.seq == 5000
    ex.dispatch('LIGHT_ON')
    assert ex.seq == 5001
    assert ex.ser.written[-1] == b'CMD,5001,LIGHT_ON\n'


def test_serial_seq_seed_is_time_based_not_zero(fake_serial):
    """默认起点来自毫秒时间戳，不是固定在 0/1——避免固件未重启时二次运行撞 DUP。"""
    ex = control.SerialExecutor('COM9')
    assert ex.seq > 100000, f'seq 起点应是大数，实际 {ex.seq}'


def test_serial_wait_ack_dup_is_not_ok(fake_serial, tmp_path):
    """ACK,DUP 表示固件静默丢弃了命令，不能当成执行成功去播报已完成。"""
    log = tmp_path / 's.jsonl'
    ex = control.SerialExecutor(port='COM_TEST', log_path=str(log))
    ex.dispatch('LIGHT_ON')
    ex.ser.responses.append(b'ACK,1,DUP\n')
    got = ex.wait_ack()
    assert got['ok'] is False
    assert got['status'] == 'DUP'
    assert got['detail'] == 'ACK,1,DUP'
    row = json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
    assert row['status'] == 'DUP'


def test_serial_wait_ack_err_is_not_ok(fake_serial):
    ex = control.SerialExecutor(port='COM_TEST')
    ex.dispatch('LIGHT_ON')
    ex.ser.responses.append(b'ERR,1,UNKNOWN_CMD\n')
    got = ex.wait_ack()
    assert got['ok'] is False
    assert got['status'] == 'UNKNOWN_CMD'


def test_serial_seq_stays_within_uint32(fake_serial):
    ex = control.SerialExecutor('COM9', seq_start=0xFFFFFFFF)
    ex.dispatch('LIGHT_ON')
    assert ex.seq == 0, 'uint32 边界应回绕，不能发出 >2^32 的 seq（固件会 BAD_SEQ）'


def test_serial_two_instances_get_distinct_seqs(fake_serial, monkeypatch):
    """两个不同时刻启动的进程，默认起点不同，不会都从 1 开始。"""
    ticks = iter([1_700_000_000.000, 1_700_000_001.000])
    monkeypatch.setattr(control.time, 'time', lambda: next(ticks))
    a = control.SerialExecutor('COM9')
    b = control.SerialExecutor('COM8')
    a.dispatch('LIGHT_ON')
    b.dispatch('LIGHT_ON')
    assert a.seq != b.seq

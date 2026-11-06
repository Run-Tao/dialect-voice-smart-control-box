# -*- coding: utf-8 -*-
"""状态源测试：mock 状态机 + 与命令表的一致性。

``runtime/state.py`` 的 ``COMMAND_TABLE`` 和 ``src/commands.py`` 的
``COMMANDS`` 是两份命令表，两处各改一处就会让「状态更新」和「执行动作」
对不上（例如表里把 ALL_OFF 的 actuator 写成 curtain）。这里钉死它们一致。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.state import (COMMAND_TABLE, DeviceState, HomeAssistantStateSource,
                           MockStateSource, _map_ha_state, build_state_source)
from src.commands import COMMANDS, INITIAL_STATE


def test_command_table_matches_commands():
    assert set(COMMAND_TABLE) == set(COMMANDS)
    for name, (actuator, action) in COMMAND_TABLE.items():
        entry = COMMANDS[name]
        assert entry['actuator'] == actuator, name
        assert entry['action'] == action, name


def test_initial_state_matches():
    state = DeviceState()
    assert state.as_dict() == dict(INITIAL_STATE)


def test_mock_state_source_applies_actions():
    src = MockStateSource()
    assert src.read().as_dict() == {'light': 'off', 'curtain': 'closed'}
    src.apply('LIGHT_ON')
    assert src.read().light == 'on'
    src.apply('CURTAIN_OPEN')
    assert src.read().curtain == 'open'
    src.apply('ALL_OFF')
    assert src.read().as_dict() == {'light': 'off', 'curtain': 'closed'}


def test_query_command_does_not_change_state():
    src = MockStateSource()
    src.apply('LIGHT_ON')
    before = src.read().as_dict()
    src.apply('STATUS_ALL')
    assert src.read().as_dict() == before


def test_mock_state_source_rejects_unknown_command():
    src = MockStateSource()
    with pytest.raises(ValueError):
        src.apply('NOT_A_COMMAND')


def test_build_state_source_defaults_to_mock():
    assert isinstance(build_state_source(), MockStateSource)
    assert isinstance(build_state_source('mock'), MockStateSource)


def test_ha_state_mapping_handles_transitional_states():
    """HA 的 cover 可能报 opening/closing，取目标态方向不明确时归到 closed。"""
    assert _map_ha_state('light', 'on') == 'on'
    assert _map_ha_state('light', 'off') == 'off'
    assert _map_ha_state('curtain', 'open') == 'open'
    assert _map_ha_state('curtain', 'closed') == 'closed'
    assert _map_ha_state('curtain', 'opening') == 'open'
    assert _map_ha_state('curtain', 'closing') == 'closed'
    assert _map_ha_state('curtain', 'unknown') == 'closed'


def test_ha_state_source_requires_url_and_token():
    with pytest.raises(ValueError):
        HomeAssistantStateSource(url='', token='x')
    with pytest.raises(ValueError):
        HomeAssistantStateSource(url='http://x', token='')


def test_unknown_state_source_kind_raises():
    with pytest.raises(ValueError):
        build_state_source('nonexistent')

from pathlib import Path

import pytest

from dialect_home.executor.simulator import SimulatorExecutor
from dialect_home.understand import Command, CommandLibrary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sim():
    return SimulatorExecutor()


@pytest.fixture(scope="module")
def lib():
    return CommandLibrary(ROOT / "config" / "commands.yaml")


def _cmd(device, action, room="living_room", **slots):
    return Command(device=device, action=action, room=room,
                   room_name="客厅", slots=slots, confidence=1.0)


def test_light_on_off(sim, lib):
    spec = lib.devices["light"]["actions"]["turn_on"]
    res = sim.execute(_cmd("light", "turn_on"), spec)
    assert res.ok and sim.state["living_room"]["light"]["on"] is True
    assert "客厅" in res.message
    sim.execute(_cmd("light", "turn_off"), lib.devices["light"]["actions"]["turn_off"])
    assert sim.state["living_room"]["light"]["on"] is False


def test_ac_temp_delta_and_clamp(sim, lib):
    spec = lib.devices["ac"]["actions"]["temp_down"]
    sim.execute(_cmd("ac", "temp_down", delta=2), spec)
    assert sim.state["living_room"]["ac"]["temp"] == 24
    for _ in range(20):
        sim.execute(_cmd("ac", "temp_down", delta=1), spec)
    assert sim.state["living_room"]["ac"]["temp"] == 16  # 下限钳位


def test_set_temp_requires_slot(sim, lib):
    spec = lib.devices["ac"]["actions"]["set_temp"]
    res = sim.execute(_cmd("ac", "set_temp"), spec)
    assert not res.ok
    res = sim.execute(_cmd("ac", "set_temp", temperature=28), spec)
    assert res.ok and sim.state["living_room"]["ac"]["temp"] == 28


def test_curtain_and_fan(sim, lib):
    sim.execute(_cmd("curtain", "close"), lib.devices["curtain"]["actions"]["close"])
    assert sim.state["living_room"]["curtain"]["open"] is False
    sim.execute(_cmd("fan", "speed_up"), lib.devices["fan"]["actions"]["speed_up"])
    assert sim.state["living_room"]["fan"]["speed"] == 2


def test_rooms_isolated(sim, lib):
    sim.execute(_cmd("light", "turn_on", room="bedroom"),
                lib.devices["light"]["actions"]["turn_on"])
    assert sim.state["bedroom"]["light"]["on"] is True
    assert sim.state["living_room"]["light"]["on"] is False
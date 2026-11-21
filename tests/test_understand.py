from pathlib import Path

import pytest

from dialect_home.understand import CommandLibrary, Understander

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def u():
    lib = CommandLibrary(ROOT / "config" / "commands.yaml")
    return Understander(lib)


CASES = [
    # 粤语口语
    ("唔该帮我熄咗客厅盏灯", "light", "turn_off", "living_room", {}),
    ("帮我开客厅盏灯", "light", "turn_on", "living_room", {}),
    ("客厅盏灯较光啲", "light", "brighter", "living_room", {}),
    ("卧室灯较暗啲", "light", "dimmer", "bedroom", {}),
    ("开冷气", "ac", "turn_on", "living_room", {}),
    ("熄咗冷气", "ac", "turn_off", "living_room", {}),
    ("冷气较低两度", "ac", "temp_down", "living_room", {"delta": 2}),
    ("冷气调到二十六度", "ac", "set_temp", "living_room", {"temperature": 26}),
    ("卧室冷气较到廿八度", "ac", "set_temp", "bedroom", {"temperature": 28}),
    ("冷气冻啲", "ac", "temp_down", "living_room", {"delta": 1}),
    ("熄电视", "tv", "turn_off", "living_room", {}),
    ("电视大声啲", "tv", "volume_up", "living_room", {}),
    ("电视细声啲", "tv", "volume_down", "living_room", {}),
    ("帮我转台", "tv", "channel_up", "living_room", {}),
    ("拉开客厅窗帘", "curtain", "open", "living_room", {}),
    ("拉埋卧室窗帘", "curtain", "close", "bedroom", {}),
    ("风扇快啲", "fan", "speed_up", "living_room", {}),
    ("开厨房插座", "socket", "turn_on", "kitchen", {}),
    ("唔该熄咗睡房盏灯", "light", "turn_off", "bedroom", {}),
    # 普通话
    ("打开客厅的灯", "light", "turn_on", "living_room", {}),
    ("关掉卧室的灯", "light", "turn_off", "bedroom", {}),
    ("把空调打开", "ac", "turn_on", "living_room", {}),
    ("空调温度调低两度", "ac", "temp_down", "living_room", {"delta": 2}),
    ("空调调到二十四度", "ac", "set_temp", "living_room", {"temperature": 24}),
    ("关闭窗帘", "curtain", "close", "living_room", {}),
    ("打开书房的风扇", "fan", "turn_on", "study", {}),
    ("厨房的灯调暗一点", "light", "dimmer", "kitchen", {}),
    ("把客厅电视关掉", "tv", "turn_off", "living_room", {}),
    # 口音错字（ASR 同音误识）：窗连=窗帘，开等=开灯
    ("打开窗连", "curtain", "open", "living_room", {}),
    ("帮我开等", "light", "turn_on", "living_room", {}),
    # 实测 ASR 粤语近音变体：较到→降到、插座→插咗
    ("卧室冷气降到廿八度", "ac", "set_temp", "bedroom", {"temperature": 28}),
    ("熄咗浴室插咗", "socket", "turn_off", "bathroom", {}),
    # ASR 同音/近音扰动与语序变化（robust_dev 回归）
    ("客厅盏灯较光滴", "light", "brighter", "living_room", {}),
    ("开冷汽", "ac", "turn_on", "living_room", {}),
    ("熄咗冷汽", "ac", "turn_off", "living_room", {}),
    ("帮我开客厅展灯", "light", "turn_on", "living_room", {}),
    ("客厅电视大声滴", "tv", "volume_up", "living_room", {}),
    ("着灯", "light", "turn_on", "living_room", {}),
    ("返上一个台", "tv", "channel_down", "living_room", {}),
    ("较暗啲卧室盏灯", "light", "dimmer", "bedroom", {}),
    ("下一个台", "tv", "channel_up", "living_room", {}),
    ("冷气冻滴", "ac", "temp_down", "living_room", {"delta": 1}),
    ("空调冻一点", "ac", "temp_down", "living_room", {"delta": 1}),
    # 第 6 轮：守卫词边界不误伤 / 温度同义词 / ASR 字母变体 D
    ("帮妈咪开灯", "light", "turn_on", "living_room", {}),
    ("客厅特别暗开灯", "light", "turn_on", "living_room", {}),
    ("冷气暖滴", "ac", "temp_up", "living_room", {"delta": 1}),
    ("冷气凉滴", "ac", "temp_down", "living_room", {"delta": 1}),
    ("冷气冻D", "ac", "temp_down", "living_room", {"delta": 1}),
    ("客厅盏灯较暗D", "light", "dimmer", "living_room", {}),
]


@pytest.mark.parametrize("text,device,action,room,slots", CASES)
def test_parse(u, text, device, action, room, slots):
    cmd = u.parse(text)
    assert cmd is not None, f"未解析: {text}"
    assert cmd.device == device, f"{text}: device {cmd.device} != {device}"
    assert cmd.action == action, f"{text}: action {cmd.action} != {action}"
    assert cmd.room == room, f"{text}: room {cmd.room} != {room}"
    for k, v in slots.items():
        assert cmd.slots.get(k) == v, f"{text}: slot {k}={cmd.slots.get(k)} != {v}"


def test_gibberish_rejected(u):
    assert u.parse("今日天气真系好好") is None
    assert u.parse("") is None


def test_non_control_speech_acts_rejected(u):
    # 含设备词但非控制意图：观看/购买/损坏/疑问，不得产出可执行指令
    for text in ["我想看电视", "唔该帮我买部冷气", "盏灯系边度买嘅",
                 "空调坏咗点算", "电视真好看", "客厅灯好像坏了",
                 "灯太光", "灯唔够光", "空调唔冷", "风扇唔转",
                 "窗帘卡住拉唔开", "睇下电视", "查下客厅灯", "盏灯几钱",
                 "唔好开灯", "咪开冷气", "不要关窗帘", "别开电视",
                 "灯开咗未", "我开咗灯", "边个开咗灯", "唔想开灯", "唔使开冷气",
                 "如果落雨就关窗帘", "电视太大声", "灯好光", "风扇好快",
                 "冷气好冻", "米开冷气", "落雨就关窗帘", "灯开未"]:
        assert u.parse(text) is None, f"误执行: {text}"


def test_ambiguous_or_slotless_temp_rejected(u):
    assert u.parse("冷气调D") is None      # 升降歧义
    assert u.parse("空调设定") is None     # set_temp 缺温度槽


def test_desire_with_explicit_alias_still_executes(u):
    cmd = u.parse("我想开灯")
    assert cmd is not None and cmd.action == "turn_on"


def test_set_temp_without_number_rejected(u):
    assert u.parse("空调调到") is None
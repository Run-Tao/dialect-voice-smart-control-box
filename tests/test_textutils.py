from dialect_home.textutils import cn2num, extract_numbers_with_unit, fuzzy_pinyin


def test_cn2num_basic():
    assert cn2num("二十六") == 26
    assert cn2num("十") == 10
    assert cn2num("十八") == 18
    assert cn2num("两") == 2
    assert cn2num("一百") == 100


def test_cn2num_cantonese_short():
    assert cn2num("廿五") == 25
    assert cn2num("廿八") == 28
    assert cn2num("卅") == 30


def test_cn2num_arabic():
    assert cn2num("26") == 26
    assert cn2num("") is None
    assert cn2num("abc") is None


def test_extract_numbers():
    assert extract_numbers_with_unit("空调调到26度") == [26]
    assert extract_numbers_with_unit("较低两度") == [2]


def test_fuzzy_pinyin_confusions():
    # 窗帘 vs 窗连（口音错字）应在模糊域相同
    assert fuzzy_pinyin("窗帘") == fuzzy_pinyin("窗连")
    # n/l 混淆：牛 vs 刘
    assert fuzzy_pinyin("牛") == fuzzy_pinyin("刘")
    # zh/z 混淆：灯 vs 登 同音；知 vs 资
    assert fuzzy_pinyin("知") == fuzzy_pinyin("资")
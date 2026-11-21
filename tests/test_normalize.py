from dialect_home.normalize import normalize


def test_traditional_to_simplified():
    assert normalize("關燈") == "关灯"
    assert normalize("打開客廳的燈") == "打开客厅的灯"


def test_cantonese_lexicon():
    assert normalize("熄咗客厅盏灯") == "关了客厅灯"
    assert normalize("冷气较低两度") == "空调调低2度"
    assert normalize("拉埋窗帘") == "拉上窗帘"
    assert normalize("电视细声啲") == "电视小声一点"


def test_fillers_removed():
    assert normalize("唔该帮我熄咗客厅盏灯") == "关了客厅灯"
    assert normalize("麻烦开一下灯") == "开灯"


def test_cantonese_numbers():
    assert normalize("冷气较到廿八度") == "空调调到28度"
    assert normalize("调到二十六度") == "调到26度"


def test_asr_homophones():
    assert normalize("开冷汽") == "开空调"
    assert normalize("帮我开客厅展灯") == "开客厅灯"
    assert normalize("客厅盏灯较光滴") == "客厅灯调亮一点"
    assert normalize("客厅电视大声滴") == "客厅电视大声一点"
"""收集 UI/命令库/语料中出现的全部汉字与中文标点，供 lv_font_conv 生成子集字库。"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[3] / "dialect-smart-home"

UI_STRINGS = """
方言语音终端 未连接 已连接 按住说话 按住 BOOT 键或屏幕按钮说话 松开结束 最长 秒
正在聆听方言 上传并识别中 识别结果 原文 归一化 动作 置信度 已执行 仅展示
识别准确 识别有误 返回重试 错误 卡未就绪 网络未连接 云端无响应 录音失败 保存失败
请再说一次 未识别到指令 客厅 卧室 厨房 浴室 书房 灯 空调 电视 窗帘 风扇 插座
开 关 调亮 调暗 升温 降温 设定温度 打开 关闭 加速 减速 换台 回台 音量加 音量减
好嘅 已经帮你 咗 较光啲 较暗啲 冷气 设定到 转咗台 返上一个台 大声啲 细声啲 较快啲 较慢啲
唔好意思 我听唔清要控制边个设备 唔该再讲一次
"""

texts = [UI_STRINGS]
for rel in ["config/commands.yaml", "data/corpus/manifest.json", "data/results/transcripts.json"]:
    p = ENGINE / rel
    if p.exists():
        texts.append(p.read_text(encoding="utf-8"))
chars = sorted(set(re.findall(r"[一-鿿　-〿＀-￯]", "".join(texts))))
(HERE / "symbols.txt").write_text("".join(chars), encoding="utf-8")
print(f"{len(chars)} symbols -> symbols.txt")
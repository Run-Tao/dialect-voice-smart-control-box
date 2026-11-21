"""用 edge-tts 合成粤语(zh-HK)/普通话(zh-CN)测试语料。

口径声明：本语料为 TTS 合成，用于可复现的自动化评测；
真实方言录音验证规程见 docs/05-测试报告.md。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"

YUE_VOICES = ["zh-HK-HiuGaaiNeural", "zh-HK-WanLungNeural"]
ZH_VOICES = ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"]

# (id, 文本, 语言, 期望设备, 期望动作, 期望房间, 期望槽位)
CORPUS = [
    ("yue01", "唔该帮我熄咗客厅盏灯", "yue", "light", "turn_off", "living_room", {}),
    ("yue02", "帮我开客厅盏灯", "yue", "light", "turn_on", "living_room", {}),
    ("yue03", "客厅盏灯较光啲", "yue", "light", "brighter", "living_room", {}),
    ("yue04", "卧室灯较暗啲", "yue", "light", "dimmer", "bedroom", {}),
    ("yue05", "开冷气", "yue", "ac", "turn_on", "living_room", {}),
    ("yue06", "熄咗冷气", "yue", "ac", "turn_off", "living_room", {}),
    ("yue07", "冷气较低两度", "yue", "ac", "temp_down", "living_room", {"delta": 2}),
    ("yue08", "冷气调高一度", "yue", "ac", "temp_up", "living_room", {"delta": 1}),
    ("yue09", "冷气调到二十六度", "yue", "ac", "set_temp", "living_room", {"temperature": 26}),
    ("yue10", "卧室冷气较到廿八度", "yue", "ac", "set_temp", "bedroom", {"temperature": 28}),
    ("yue11", "开电视", "yue", "tv", "turn_on", "living_room", {}),
    ("yue12", "熄电视", "yue", "tv", "turn_off", "living_room", {}),
    ("yue13", "电视大声啲", "yue", "tv", "volume_up", "living_room", {}),
    ("yue14", "电视细声啲", "yue", "tv", "volume_down", "living_room", {}),
    ("yue15", "帮我转台", "yue", "tv", "channel_up", "living_room", {}),
    ("yue16", "拉开客厅窗帘", "yue", "curtain", "open", "living_room", {}),
    ("yue17", "拉埋卧室窗帘", "yue", "curtain", "close", "bedroom", {}),
    ("yue18", "开风扇", "yue", "fan", "turn_on", "living_room", {}),
    ("yue19", "熄风扇", "yue", "fan", "turn_off", "living_room", {}),
    ("yue20", "风扇快啲", "yue", "fan", "speed_up", "living_room", {}),
    ("yue21", "风扇慢啲", "yue", "fan", "speed_down", "living_room", {}),
    ("yue22", "开厨房插座", "yue", "socket", "turn_on", "kitchen", {}),
    ("yue23", "熄咗浴室插座", "yue", "socket", "turn_off", "bathroom", {}),
    ("yue24", "书房灯较光啲", "yue", "light", "brighter", "study", {}),
    ("yue25", "唔该熄咗睡房盏灯", "yue", "light", "turn_off", "bedroom", {}),
    ("yue26", "冷气冻啲", "yue", "ac", "temp_down", "living_room", {"delta": 1}),
    ("zh01", "打开客厅的灯", "zh", "light", "turn_on", "living_room", {}),
    ("zh02", "关掉卧室的灯", "zh", "light", "turn_off", "bedroom", {}),
    ("zh03", "把空调打开", "zh", "ac", "turn_on", "living_room", {}),
    ("zh04", "空调温度调低两度", "zh", "ac", "temp_down", "living_room", {"delta": 2}),
    ("zh05", "空调调到二十四度", "zh", "ac", "set_temp", "living_room", {"temperature": 24}),
    ("zh06", "关闭窗帘", "zh", "curtain", "close", "living_room", {}),
    ("zh07", "打开窗帘", "zh", "curtain", "open", "living_room", {}),
    ("zh08", "电视大声一点", "zh", "tv", "volume_up", "living_room", {}),
    ("zh09", "风扇调快一点", "zh", "fan", "speed_up", "living_room", {}),
    ("zh10", "打开书房的风扇", "zh", "fan", "turn_on", "study", {}),
    ("zh11", "厨房的灯调暗一点", "zh", "light", "dimmer", "kitchen", {}),
    ("zh12", "关上浴室的插座", "zh", "socket", "turn_off", "bathroom", {}),
    ("zh13", "空调温度调高一度", "zh", "ac", "temp_up", "living_room", {"delta": 1}),
    ("zh14", "把客厅电视关掉", "zh", "tv", "turn_off", "living_room", {}),
]


async def synth_one(item, voice: str, out_dir: Path) -> dict:
    import edge_tts

    uid, text, lang, device, action, room, slots = item
    mp3 = out_dir / f"{uid}.mp3"
    wav = out_dir / f"{uid}.wav"
    await edge_tts.Communicate(text, voice).save(str(mp3))
    import imageio_ffmpeg
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(mp3),
         "-ar", "16000", "-ac", "1", str(wav)],
        check=True, capture_output=True,
    )
    mp3.unlink(missing_ok=True)
    return {"id": uid, "text": text, "lang": lang, "voice": voice,
            "wav": f"data/corpus/{uid}.wav",
            "expected": {"device": device, "action": action, "room": room, "slots": slots}}


async def run(proxy: str | None) -> None:
    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["HTTP_PROXY"] = proxy
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, item in enumerate(CORPUS):
        voices = YUE_VOICES if item[2] == "yue" else ZH_VOICES
        voice = voices[i % len(voices)]
        rec = await synth_one(item, voice, CORPUS_DIR)
        manifest.append(rec)
        print(f"[{i + 1}/{len(CORPUS)}] {rec['id']} {voice} {rec['text']}")
    (CORPUS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"语料合成完成: {len(manifest)} 条 → {CORPUS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default="http://127.0.0.1:7890",
                        help="edge-tts 需要的代理（默认 FlClash 7890，传 none 关闭）")
    args = parser.parse_args()
    proxy = None if args.proxy.lower() == "none" else args.proxy
    asyncio.run(run(proxy))


if __name__ == "__main__":
    sys.exit(main())
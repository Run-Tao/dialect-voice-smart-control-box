"""命令行入口：wav 文件 / 麦克风 / 文本直输。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dialect_home.audio import record  # noqa: E402
from dialect_home.pipeline import Pipeline, load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="方言智居 · 方言语音指令转换器")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="识别 wav 音频文件")
    src.add_argument("--mic", type=float, metavar="SEC", help="录音 SEC 秒后识别")
    src.add_argument("--text", help="跳过 ASR，直接解析文本（调试理解层）")
    parser.add_argument("--backend", choices=["sensevoice", "whisper", "mock"])
    parser.add_argument("--lang", choices=["auto", "yue", "zh"], default=None)
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args()

    settings = load_settings()
    if args.backend:
        settings["asr"]["backend"] = args.backend
    if args.no_tts:
        settings["tts"]["enabled"] = False

    pipe = Pipeline(settings)
    if args.text:
        result = pipe.process_text(args.text)
    elif args.wav:
        result = pipe.process_wav(args.wav, language=args.lang)
    else:
        print(f"录音 {args.mic:.0f} 秒，请说指令…")
        clip = record(args.mic)
        result = pipe.process_audio(clip, language=args.lang)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"识别文本 [{result.backend}/{result.language}]: {result.raw_text}")
    print(f"归一化: {result.normalized_text}")
    if result.command:
        print(f"命令: {json.dumps(result.command, ensure_ascii=False)}")
        print(f"执行: {'成功' if result.ok else '失败'} · {result.feedback}")
    else:
        print(f"未执行: {result.error}")
    print(f"耗时(ms): {result.timings_ms}")


if __name__ == "__main__":
    main()
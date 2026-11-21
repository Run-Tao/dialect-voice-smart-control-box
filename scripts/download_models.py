"""下载 ASR 模型到项目 models/ 目录（ModelScope 国内源，无需代理）。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", action="store_true",
                        help="同时下载 faster-whisper small 备用模型（走 hf-mirror）")
    args = parser.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(MODELS)
    os.environ.setdefault("MODELSCOPE_DOMAIN", "www.modelscope.cn")

    from modelscope import snapshot_download

    for model_id in ["iic/SenseVoiceSmall",
                     "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"]:
        print(f"下载 {model_id} …")
        path = snapshot_download(model_id, cache_dir=str(MODELS))
        print(f"  → {path}")

    if args.whisper:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download as hf_download
        print("下载 Systran/faster-whisper-small …")
        path = hf_download("Systran/faster-whisper-small",
                           local_dir=str(MODELS / "whisper" / "small"))
        print(f"  → {path}")
    print("模型下载完成。")


if __name__ == "__main__":
    sys.exit(main())
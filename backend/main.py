"""端云协同云端服务：/api/v1/speech/understand 与 /api/v1/speech/feedback。

识别/理解全部委托 PC 版引擎（asr_nlu_pipeline.CloudEngine）；本文件只做协议、
WAV 存档（dataset/audio/）与标注状态回写（dataset/metadata.jsonl）。
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from asr_nlu_pipeline import CloudEngine

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "cloud_config.yaml").read_text(encoding="utf-8"))
ENGINE_ROOT = (ROOT / CFG.get("engine_root", "../../dialect-smart-home")).resolve()
DATASET = ROOT / CFG.get("dataset_dir", "dataset")
AUDIO_DIR = DATASET / "audio"
METADATA = DATASET / "metadata.jsonl"

engine = CloudEngine(ENGINE_ROOT)
_meta_lock = threading.Lock()
_seq_lock = threading.Lock()
_seq = 0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _next_sample_id() -> str:
    global _seq
    with _seq_lock:
        _seq += 1
        return time.strftime("%Y%m%d_%H%M%S") + f"_{_seq:03d}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    METADATA.touch(exist_ok=True)
    yield


app = FastAPI(title="Dialect Voice Cloud (PC engine adapter)", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    with _meta_lock:
        n = sum(1 for _ in METADATA.open(encoding="utf-8")) if METADATA.exists() else 0
    return {"status": "ok", "engine_ready": engine.ready,
            "engine_root": str(ENGINE_ROOT), "samples": n}


@app.post("/api/v1/speech/understand")
def understand(file: UploadFile = File(...), device_id: str = Form(...),
               timestamp: str = Form(...), sample_id: str = Form("")):
    data = file.file.read()
    if len(data) < 44 or not data.startswith(b"RIFF"):
        raise HTTPException(status_code=400, detail="file 必须是 RIFF/WAV 音频")
    sid = sample_id.strip() or _next_sample_id()
    wav_path = AUDIO_DIR / f"{sid}.wav"
    wav_path.write_bytes(data)

    result = engine.understand(wav_path)   # 同步端点 → FastAPI 线程池，不阻塞事件循环
    result["sample_id"] = sid

    record = {
        "sample_id": sid,
        "device_id": device_id,
        "timestamp": timestamp,
        "received_at": _now_iso(),
        "wav_path": f"audio/{sid}.wav",
        **{k: result[k] for k in ("dialect", "raw_text", "normalized_text",
                                  "intent", "slots", "confidence", "reply_text")},
        "annotation": {"state": "pending", "is_correct": None,
                       "corrected_text": None, "feedback_at": None},
    }
    with _meta_lock:
        with METADATA.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    return JSONResponse(result)


class FeedbackIn(BaseModel):
    sample_id: str
    is_correct: bool
    corrected_text: str | None = None


@app.post("/api/v1/speech/feedback")
def feedback(fb: FeedbackIn):
    with _meta_lock:
        lines = METADATA.read_text(encoding="utf-8").splitlines() if METADATA.exists() else []
        found = False
        out = []
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("sample_id") == fb.sample_id:
                rec["annotation"] = {
                    "state": "annotated",
                    "is_correct": fb.is_correct,
                    "corrected_text": fb.corrected_text,
                    "feedback_at": _now_iso(),
                }
                found = True
            out.append(json.dumps(rec, ensure_ascii=False))
        if not found:
            raise HTTPException(status_code=404, detail=f"未知 sample_id: {fb.sample_id}")
        tmp = METADATA.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        tmp.replace(METADATA)
    return {"sample_id": fb.sample_id, "annotation_state": "annotated"}
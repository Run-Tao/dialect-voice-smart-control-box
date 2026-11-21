"""端到端评测：合成语料 → ASR → 理解 → 执行，统计意图/房间/槽位准确率与延迟。"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dialect_home.pipeline import Pipeline, load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="sensevoice")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "corpus" / "manifest.json"))
    parser.add_argument("--out", default=str(ROOT / "data" / "results" / "eval_results.json"))
    parser.add_argument("--no-tts", action="store_true", default=True)
    args = parser.parse_args()

    settings = load_settings()
    settings["asr"]["backend"] = args.backend
    settings["tts"]["enabled"] = False
    pipe = Pipeline(settings)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows, transcripts = [], []
    for rec in manifest:
        exp = rec["expected"]
        t0 = time.perf_counter()
        res = pipe.process_wav(ROOT / rec["wav"])
        wall_ms = (time.perf_counter() - t0) * 1000
        cmd = res.command or {}
        got_slots = cmd.get("slots", {})
        intent_ok = cmd.get("device") == exp["device"] and cmd.get("action") == exp["action"]
        room_ok = cmd.get("room") == exp["room"]
        slots_ok = all(got_slots.get(k) == v for k, v in exp["slots"].items())
        rows.append({
            "id": rec["id"], "lang": rec["lang"], "text": rec["text"],
            "asr_text": res.raw_text, "asr_lang": res.language,
            "normalized": res.normalized_text,
            "expected": exp, "got": cmd,
            "intent_ok": intent_ok, "room_ok": room_ok, "slots_ok": slots_ok,
            "exact_ok": intent_ok and room_ok and slots_ok,
            "exec_ok": res.ok, "asr_ms": res.timings_ms.get("asr", 0),
            "wall_ms": round(wall_ms, 1),
        })
        transcripts.append({"id": rec["id"], "text": rec["text"], "asr": res.raw_text})
        mark = "OK " if intent_ok and room_ok and slots_ok else "FAIL"
        print(f"[{mark}] {rec['id']} 「{rec['text']}」→「{res.raw_text}」 "
              f"{cmd.get('device')}.{cmd.get('action')} @{cmd.get('room')} {got_slots}")

    def acc(key, subset=None):
        data = [r for r in rows if subset is None or r["lang"] == subset]
        return round(100 * sum(r[key] for r in data) / len(data), 1) if data else 0.0

    asr_times = [r["asr_ms"] for r in rows]
    wall_times = [r["wall_ms"] for r in rows]
    summary = {
        "backend": args.backend,
        "corpus": "TTS 合成（zh-HK 粤语 + zh-CN 普通话），非真实录音",
        "n": len(rows),
        "intent_accuracy": acc("intent_ok"),
        "intent_accuracy_yue": acc("intent_ok", "yue"),
        "intent_accuracy_zh": acc("intent_ok", "zh"),
        "room_accuracy": acc("room_ok"),
        "slot_accuracy": acc("slots_ok"),
        "exact_accuracy": acc("exact_ok"),
        "exec_success_rate": round(100 * sum(r["exec_ok"] for r in rows) / len(rows), 1),
        "asr_ms_mean": round(statistics.mean(asr_times), 1),
        "asr_ms_p95": round(sorted(asr_times)[int(len(asr_times) * 0.95) - 1], 1),
        "wall_ms_mean": round(statistics.mean(wall_times), 1),
        "wall_ms_p95": round(sorted(wall_times)[int(len(wall_times) * 0.95) - 1], 1),
    }
    out = {"summary": summary, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(args.out).parent / "transcripts.json").write_text(
        json.dumps(transcripts, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
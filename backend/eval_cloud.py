"""云端契约链路评测：40 条语料 WAV → CloudEngine.understand → 契约字段比对。

与 scripts/run_eval.py（引擎直连）不同，本脚本覆盖适配层映射：
intent/slots.location/device/action(动词映射)/数值槽/dialect/confidence 门槛。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

from asr_nlu_pipeline import CloudEngine  # noqa: E402

ENGINE = (BACKEND / "../../dialect-smart-home").resolve()
ACTION_MAP = {"turn_on": "on", "turn_off": "off"}


def main() -> None:
    manifest = json.loads((ENGINE / "data/corpus/manifest.json").read_text(encoding="utf-8"))
    eng = CloudEngine(ENGINE)
    rows = []
    for rec in manifest:
        exp = rec["expected"]
        t0 = time.perf_counter()
        res = eng.understand(ENGINE / rec["wav"])
        ms = (time.perf_counter() - t0) * 1000
        slots = res["slots"]
        intent_ok = (res["intent"] == "device_control"
                     and slots.get("device") == exp["device"]
                     and slots.get("action") == ACTION_MAP.get(exp["action"], exp["action"]))
        room_ok = slots.get("location") == exp["room"]
        slots_ok = all(slots.get(k) == v for k, v in exp["slots"].items())
        dialect_ok = res["dialect"] == ("cantonese" if rec["lang"] == "yue" else "mandarin")
        gate_ok = res["confidence"] >= 0.7
        exact = intent_ok and room_ok and slots_ok and dialect_ok and gate_ok
        rows.append({"id": rec["id"], "lang": rec["lang"], "text": rec["text"],
                     "contract": res, "expected": exp, "intent_ok": intent_ok,
                     "room_ok": room_ok, "slots_ok": slots_ok, "dialect_ok": dialect_ok,
                     "gate_ok": gate_ok, "exact_ok": exact, "ms": round(ms, 1)})
        mark = "OK " if exact else "FAIL"
        print(f"[{mark}] {rec['id']} 「{rec['text']}」→ {res['intent']} {slots} "
              f"conf={res['confidence']} dialect={res['dialect']}", flush=True)

    def acc(key, subset=None):
        data = [r for r in rows if subset is None or r["lang"] == subset]
        return round(100 * sum(r[key] for r in data) / len(data), 1) if data else 0.0

    summary = {
        "path": "cloud adapter (CloudEngine.understand)",
        "corpus": "TTS 合成（zh-HK 粤语 + zh-CN 普通话），非真实录音",
        "n": len(rows),
        "intent_accuracy": acc("intent_ok"),
        "intent_accuracy_yue": acc("intent_ok", "yue"),
        "intent_accuracy_zh": acc("intent_ok", "zh"),
        "room_accuracy": acc("room_ok"),
        "slot_accuracy": acc("slots_ok"),
        "dialect_accuracy": acc("dialect_ok"),
        "gate_accuracy": acc("gate_ok"),
        "contract_exact_accuracy": acc("exact_ok"),
        "ms_mean": round(statistics.mean(r["ms"] for r in rows), 1),
    }
    out = {"summary": summary, "rows": rows}
    (BACKEND / "eval_results_cloud.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== 云端链路汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fails = [r["id"] for r in rows if not r["exact_ok"]]
    print("failures:", fails or "none")


if __name__ == "__main__":
    main()
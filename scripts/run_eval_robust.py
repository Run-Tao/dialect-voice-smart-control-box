"""NLU 鲁棒性/拒识评测（文本通道，隔离 ASR 变量）：模拟 ASR 同音噪声、语序、语气词、
数字汉字写法，以及域外闲聊必须拒识（expected=null → 不得产出可执行指令）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dialect_home.pipeline import Pipeline, load_settings  # noqa: E402


def main() -> None:
    settings = load_settings()
    settings["asr"]["backend"] = "mock"
    settings["tts"]["enabled"] = False
    pipe = Pipeline(settings)
    data = json.loads((ROOT / "data/corpus/robust_dev.json").read_text(encoding="utf-8"))
    rows = []
    for rec in data:
        exp = rec["expected"]
        res = pipe.process_text(rec["text"])
        cmd = res.command or {}
        if exp is None:
            ok = res.command is None
            kind = "reject"
        else:
            got_slots = cmd.get("slots", {})
            ok = (cmd.get("device") == exp["device"] and cmd.get("action") == exp["action"]
                  and cmd.get("room") == exp["room"]
                  and all(got_slots.get(k) == v for k, v in exp["slots"].items()))
            kind = "positive"
        rows.append({"id": rec["id"], "kind": kind, "text": rec["text"],
                     "normalized": res.normalized_text, "got": cmd or None,
                     "confidence": cmd.get("confidence", 0.0), "ok": ok})
        print(f"[{'OK ' if ok else 'FAIL'}] {rec['id']} ({kind}) 「{rec['text']}」"
              f"→「{res.normalized_text}」 {cmd.get('device')}.{cmd.get('action')} "
              f"@{cmd.get('room')} {cmd.get('slots')} conf={cmd.get('confidence', 0)}",
              flush=True)

    pos = [r for r in rows if r["kind"] == "positive"]
    neg = [r for r in rows if r["kind"] == "reject"]
    summary = {
        "n_positive": len(pos), "n_negative": len(neg),
        "robust_accuracy": round(100 * sum(r["ok"] for r in pos) / len(pos), 1),
        "reject_accuracy": round(100 * sum(r["ok"] for r in neg) / len(neg), 1),
    }
    out = {"summary": summary, "rows": rows}
    (ROOT / "data/results/eval_robust.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== 鲁棒性/拒识汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("failures:", [r["id"] for r in rows if not r["ok"]] or "none")


if __name__ == "__main__":
    main()
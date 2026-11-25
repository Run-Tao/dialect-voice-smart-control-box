from __future__ import annotations

import json
from pathlib import Path

import easyocr

src = Path(r"C:\Users\Taoli\.codex\visualizations\2026\11\13\01a1d380-839f-77c3-acd7-8bc6f6b658c6\hardware_pages72")
out = Path(r"C:\Users\Taoli\.codex\visualizations\2026\11\13\01a1d380-839f-77c3-acd7-8bc6f6b658c6\hardware_ocr")
out.mkdir(parents=True, exist_ok=True)
reader = easyocr.Reader(["ch_sim", "en"], gpu=False,
                        download_enabled=False, verbose=False)

all_pages = {}
for image in sorted(src.glob("page-*.png")):
    result = reader.readtext(str(image), detail=1, paragraph=False)
    rows = []
    for box, text, confidence in result:
        x = min(point[0] for point in box)
        y = min(point[1] for point in box)
        rows.append({
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "text": text,
            "confidence": round(float(confidence), 3),
        })
    rows.sort(key=lambda row: (row["y"], row["x"]))
    all_pages[image.name] = rows
    lines = [f"{row['y']:7.1f} {row['x']:7.1f} {row['confidence']:.3f} {row['text']}"
             for row in rows]
    (out / f"{image.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(image.name, len(rows), flush=True)

(out / "all.json").write_text(
    json.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8")

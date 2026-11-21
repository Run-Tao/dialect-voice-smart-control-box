"""Flask Web 服务：浏览器录音识别 + 设备状态面板 + 文本调试。"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dialect_home.audio import load_wav  # noqa: E402
from dialect_home.pipeline import Pipeline, load_settings  # noqa: E402


def create_app(settings: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    pipe = Pipeline(settings or load_settings())
    web_dir = ROOT / "web"

    @app.get("/")
    def index():
        return send_from_directory(web_dir, "index.html")

    @app.get("/web/<path:name>")
    def web_assets(name: str):
        return send_from_directory(web_dir, name)

    @app.post("/api/recognize")
    def recognize():
        if "audio" not in request.files:
            return jsonify({"error": "缺少 audio 文件字段"}), 400
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            request.files["audio"].save(fp.name)
            tmp = fp.name
        try:
            clip = load_wav(tmp)
            result = pipe.process_audio(clip)
            return jsonify(result.to_dict())
        finally:
            Path(tmp).unlink(missing_ok=True)

    @app.post("/api/text")
    def text_debug():
        text = (request.get_json(silent=True) or {}).get("text", "")
        return jsonify(pipe.process_text(text).to_dict())

    @app.get("/api/state")
    def state():
        snap = pipe.executor.snapshot()
        return jsonify({"state": snap, "executor": pipe.settings["executor"]["type"]})

    @app.post("/api/reset")
    def reset():
        if hasattr(pipe.executor, "reset"):
            pipe.executor.reset()
        return jsonify({"ok": True})

    return app


def main() -> None:
    settings = load_settings()
    app = create_app(settings)
    srv = settings.get("server", {})
    print(f"方言智居 Web 服务: http://{srv.get('host', '127.0.0.1')}:{srv.get('port', 8000)}")
    app.run(host=srv.get("host", "127.0.0.1"), port=srv.get("port", 8000), threaded=True)


if __name__ == "__main__":
    main()
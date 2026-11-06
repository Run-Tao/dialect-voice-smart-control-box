# -*- coding: utf-8 -*-
"""v3 本地 Web 演示台：离线页面 + JSON API（Flask）。

启动：
    python scripts/run_web_demo.py                       # 默认 127.0.0.1:8770
    python scripts/run_web_demo.py --port 8771 --threshold 0.5
    python scripts/run_web_demo.py --model data/models/tts_control_model_dtw_demo.pkl

页面完全离线（内联 CSS/JS，无 CDN），后端把每次点击转成
``V3ControlServer.process``，并返回识别结果、设备状态与审计日志。
"""
import argparse
import sys
import threading
from pathlib import Path

# 允许以 `python scripts/run_web_demo.py` 直接运行
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request  # noqa: E402

from src.control_server_v3 import (  # noqa: E402
    MockDeviceV3,
    V3ControlServer,
    load_predictor,
    state_to_text,
)
from web.demo_catalog import build_demo_catalog  # noqa: E402
from web.page import render_page  # noqa: E402

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8770
DEFAULT_MODEL = 'data/models/tts_control_model_dtw_demo.pkl'
DEFAULT_LOG = 'outputs/v3_demo_log.jsonl'
MAX_LOG_LIMIT = 100


def create_app(server, catalog, base_dir=None, log_path=None):
    """构建 Flask 应用（可在测试中用 mock predictor 独立构建）。

    参数：
        server  : 已构造的 ``V3ControlServer``。
        catalog : ``web.demo_catalog.build_demo_catalog`` 的返回值。
        base_dir: 项目根目录，用于解析 API 收到的相对音频路径。
        log_path: 页脚展示的日志文件路径。
    """
    base = Path(base_dir) if base_dir else PROJECT_ROOT
    app = Flask(__name__)
    app.json.ensure_ascii = False
    lock = threading.Lock()  # 串行化推理 + 日志追加，避免 dev server 多线程竞争

    def state_payload():
        return {
            'state': server.state,
            'state_text': state_to_text(server.state),
            'threshold': server.threshold,
            'model_name': server.model_name,
            'model_type': server.model_type,
        }

    @app.after_request
    def _no_cache(resp):
        if request.path.startswith('/api/'):
            resp.headers['Cache-Control'] = 'no-store'
        return resp

    @app.get('/')
    def index():
        page = render_page(
            catalog,
            default_threshold=server.threshold,
            model_name=server.model_name,
            log_path=log_path or getattr(server, 'log_path', '') or '',
        )
        return page, 200, {'Content-Type': 'text/html; charset=utf-8'}

    @app.get('/api/health')
    def health():
        return jsonify({'status': 'ok', 'model_name': server.model_name,
                        'threshold': server.threshold})

    @app.get('/api/catalog')
    def get_catalog():
        return jsonify(catalog)

    @app.get('/api/state')
    def get_state():
        with lock:
            return jsonify(state_payload())

    @app.post('/api/reset')
    def reset():
        with lock:
            server.reset()
            return jsonify(state_payload())

    @app.get('/api/log')
    def get_log():
        raw = request.args.get('limit', '10')
        try:
            limit = int(raw)
        except (TypeError, ValueError):
            return jsonify({'error': f'limit 必须是整数，收到：{raw}'}), 400
        limit = max(1, min(limit, MAX_LOG_LIMIT))
        with lock:
            records = server.read_log(limit)
        return jsonify({'records': records, 'limit': limit,
                        'log_path': str(log_path or getattr(server, 'log_path', '') or '')})

    @app.post('/api/predict')
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({'error': '请求体必须是 JSON 对象，例如 {"audio_path": "..."}'}), 400
        raw_audio = payload.get('audio_path')
        if not isinstance(raw_audio, str) or not raw_audio.strip():
            return jsonify({'error': '缺少 audio_path（音频文件路径）'}), 400

        audio = Path(raw_audio.strip())
        if not audio.is_absolute():
            audio = base / audio
        if not audio.exists():
            return jsonify({'error': f'音频文件不存在：{audio}'}), 404
        if not audio.is_file():
            return jsonify({'error': f'音频路径不是文件：{audio}'}), 400

        threshold = payload.get('threshold')
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                return jsonify({'error': 'threshold 必须是 0~1 之间的数字'}), 400
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                return jsonify({'error': f'threshold 必须在 0~1 之间，收到：{threshold}'}), 400
        try:
            with lock:
                result = server.process(audio, threshold=threshold)
        except FileNotFoundError as exc:
            return jsonify({'error': f'音频文件读取失败：{exc}'}), 404
        except ValueError as exc:
            return jsonify({'error': f'音频或参数错误：{exc}'}), 400
        except Exception as exc:  # pragma: no cover - 防御性兜底
            return jsonify({'error': f'处理失败：{exc}'}), 500
        return jsonify(result)

    return app


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='方言智控盒 v3 本地 Web 演示台（离线内联页面 + 模拟设备）')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'识别模型路径（默认 {DEFAULT_MODEL}；v3 模型或 v2 DTW pickle）')
    parser.add_argument('--base', default=str(PROJECT_ROOT), help='项目根目录（默认自动探测）')
    parser.add_argument('--host', default=DEFAULT_HOST, help=f'监听地址（默认 {DEFAULT_HOST}）')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'监听端口（默认 {DEFAULT_PORT}）')
    parser.add_argument('--threshold', type=float, default=None,
                        help='覆盖默认拒识阈值（默认使用模型自带阈值）')
    parser.add_argument('--log', default=DEFAULT_LOG,
                        help=f'操作日志 JSONL 路径（默认 {DEFAULT_LOG}）')
    parser.add_argument('--reset-log', action='store_true',
                        help='启动前清空日志文件（默认追加写入，保留审计记录）')
    args = parser.parse_args(argv)

    base = Path(args.base).resolve()
    model_path = _resolve(base, args.model)
    log_path = _resolve(base, args.log)

    print('=' * 68)
    print('方言智控盒 · 第三階段 Web 演示台')
    print('=' * 68)
    print(f'项目根目录：{base}')
    print(f'识别模型：{model_path}')
    try:
        predictor = load_predictor(model_path)
    except Exception as exc:
        print(f'\n[错误] 模型加载失败：{exc}', file=sys.stderr)
        print('提示：可先用 v2 DTW 演示模型：')
        print('      python scripts/run_web_demo.py --model '
              'data/models/tts_control_model_dtw_demo.pkl', file=sys.stderr)
        return 2
    threshold = float(args.threshold) if args.threshold is not None else None
    server = V3ControlServer(predictor, threshold=threshold,
                             device=MockDeviceV3(), log_path=log_path)
    if args.reset_log and log_path.exists():
        log_path.unlink()
    catalog = build_demo_catalog(base)
    app = create_app(server, catalog, base_dir=base, log_path=log_path)

    print(f'模型类型：{server.model_type} / {server.model_name}')
    print(f'拒識閾值：{server.threshold:.2f}'
          + ('（模型默认）' if args.threshold is None else '（命令行覆盖）'))
    print(f'操作日志：{log_path}')
    for item in catalog['commands']:
        print(f"  [按钮] {item['label']:<8} {item['command']:<14} "
              f"{Path(item['audio_path']).name}")
    if catalog['unknown']:
        print(f"  [未知] 應拒識          {Path(catalog['unknown']['audio_path']).name}")
    for warning in catalog['warnings']:
        print(f'  [提示] {warning}')
    print('-' * 68)
    print(f'浏览器打开：http://{args.host}:{args.port}/   （Ctrl+C 结束）')
    print('=' * 68)
    try:
        app.run(host=args.host, port=args.port, debug=False,
                use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print('\n演示台已停止。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

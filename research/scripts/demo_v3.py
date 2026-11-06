# -*- coding: utf-8 -*-
"""v3 命令行演示：语音指令 -> 识别/拒识 -> 模拟设备 -> 状态 -> 日志。

用法：
    python scripts/demo_v3.py --sequence                 # 六类命令 + 一条未知指令
    python scripts/demo_v3.py --audio data/xxx.wav       # 单条音频
    python scripts/demo_v3.py --interactive              # 逐条确认的交互演示
    python scripts/demo_v3.py --threshold 0.5            # 覆盖模型阈值

模型默认使用二期 v2 DTW 演示模型；如已拿到 v3 模型，用 --model 指向该文件即可
（``load_predictor`` 会优先尝试 src/v3_model.py 的 V3Predictor）。
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control_server_v3 import (  # noqa: E402
    COMMAND_LABELS,
    MockDeviceV3,
    V3ControlServer,
    load_predictor,
    state_to_text,
)
from web.demo_catalog import build_demo_catalog  # noqa: E402

DEFAULT_MODEL = 'data/models/tts_control_model_dtw_demo.pkl'
DEFAULT_LOG = 'outputs/v3_demo_log.jsonl'


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def _event_block(index, total, item, result, show_scores=False):
    """打印一条事件（含音频来源、识别、阈值、设备、状态、延迟）。"""
    audio = Path(result['audio_path'])
    label = result.get('command_label') or COMMAND_LABELS.get(
        result.get('predicted_intent'), '未知指令')
    predicted = result.get('predicted_intent') or '—'
    confidence = float(result['confidence'])
    threshold = float(result['threshold'])
    verdict = '接受' if result['accepted'] else '拒識'
    if result.get('rejection_reason') == 'unknown_command':
        verdict += '（未知指令）'
    elif result.get('rejection_reason') == 'low_confidence':
        verdict += '（低於閾值）'
    print(f'[{index}/{total}] 音頻：{audio.name}')
    if item:
        text = item.get('text') or ''
        meta = []
        if item.get('group'):
            meta.append(f"表達組 {item['group']}")
        if item.get('voice'):
            meta.append(f"音色 {item['voice']}")
        if text:
            meta.append(f'文本「{text}」')
        if meta:
            print('        ' + ' · '.join(meta))
    print(f'        識別：{label} ({predicted})  置信度 {confidence:.3f} '
          f'{">=" if confidence >= threshold else "<"} 閾值 {threshold:.2f} → {verdict}')
    print(f"        設備：{result.get('response')} ｜ 狀態：{result.get('state_text')} "
          f"｜ 延遲：{result['latency_sec'] * 1000:.0f} ms")
    if show_scores and result.get('scores'):
        top = sorted(result['scores'].items(), key=lambda kv: kv[1], reverse=True)[:3]
        print('        分數：' + '，'.join(f'{k}={v:.3f}' for k, v in top))


def _print_state(server, title='當前設備狀態'):
    print(f'{title}：{state_to_text(server.state)}  '
          f'(light={server.state["light"]}, curtain={server.state["curtain"]})')


def _display_width(text):
    """近似计算终端显示宽度（中文/全角字符按 2 列）。"""
    return sum(2 if ord(ch) > 127 else 1 for ch in str(text))


def _fit(text, width, align='left'):
    """按显示宽度截断/补空格，保证中文表格对齐。"""
    text = str(text)
    chars = list(text)
    out, used = [], 0
    for i, ch in enumerate(chars):
        ch_width = 2 if ord(ch) > 127 else 1
        remaining = len(chars) - i - 1
        # 若不是最后一个字符，需要为省略号预留 2 列
        limit = width if remaining == 0 else width - 2
        if used + ch_width > limit:
            out.append('…')
            used += 2
            break
        out.append(ch)
        used += ch_width
    pad = max(0, width - used)
    body = ''.join(out)
    return body + ' ' * pad if align == 'left' else ' ' * pad + body


def _print_summary(results):
    print()
    line = '-' * 103
    print(line)
    header = ' '.join((_fit('#', 3), _fit('表達組 / 文件', 24), _fit('識別命令', 14),
                       _fit('置信度', 9, 'right'), _fit('閾值', 7, 'right'),
                       _fit('結果', 7), _fit('設備反饋', 23),
                       _fit('延遲', 9, 'right')))
    print(header)
    print(line)
    for i, item in enumerate(results, 1):
        result = item['result']
        source = item['source'] or Path(result['audio_path']).name
        print(' '.join((
            _fit(i, 3),
            _fit(source, 24),
            _fit(result.get('predicted_intent') or '—', 14),
            _fit(f"{result['confidence']:.3f}", 9, 'right'),
            _fit(f"{result['threshold']:.2f}", 7, 'right'),
            _fit('接受' if result['accepted'] else '拒識', 7),
            _fit(result.get('response') or '', 23),
            _fit(f"{result['latency_sec'] * 1000:.0f}ms", 9, 'right'),
        )))
    print(line)


def _sequence_items(catalog):
    """组装演示序列：六类命令各一条 + 一条未知指令。"""
    items = []
    for sample in catalog.get('commands') or []:
        items.append(dict(sample, mode='command'))
    if catalog.get('unknown'):
        items.append(dict(catalog['unknown'], mode='unknown'))
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='方言智控盒 v3 命令行演示（模拟设备，TTS 测试音频）')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'识别模型路径（默认 {DEFAULT_MODEL}）')
    parser.add_argument('--audio', default=None, help='单条音频路径（.wav）')
    parser.add_argument('--sequence', action='store_true',
                        help='运行六类命令 + 一条未知指令的自动序列（默认）')
    parser.add_argument('--interactive', action='store_true',
                        help='逐条确认的交互演示（按 Enter 继续，q 退出）')
    parser.add_argument('--threshold', type=float, default=None,
                        help='覆盖拒识阈值（默认使用模型自带阈值）')
    parser.add_argument('--log', default=DEFAULT_LOG,
                        help=f'操作日志 JSONL 路径（默认 {DEFAULT_LOG}）')
    parser.add_argument('--base', default=str(PROJECT_ROOT), help='项目根目录')
    parser.add_argument('--show-scores', action='store_true', help='显示 Top-3 类别分数')
    parser.add_argument('--reset-log', action='store_true', help='开始前清空日志文件')
    args = parser.parse_args(argv)

    base = Path(args.base).resolve()
    model_path = _resolve(base, args.model)
    log_path = _resolve(base, args.log)

    print('=' * 96)
    print('方言智控盒 · 第三階段命令行演示')
    print('=' * 96)
    print(f'模型：{model_path}')
    try:
        predictor = load_predictor(model_path)
    except Exception as exc:
        print(f'[错误] 模型加载失败：{exc}', file=sys.stderr)
        return 2
    threshold = float(args.threshold) if args.threshold is not None else None
    server = V3ControlServer(predictor, threshold=threshold,
                             device=MockDeviceV3(), log_path=log_path)
    if args.reset_log and log_path.exists():
        log_path.unlink()
    print(f'模型類型：{server.model_type} / {server.model_name}')
    print(f'拒識閾值：{server.threshold:.2f}'
          + ('（模型默认）' if args.threshold is None else '（命令行覆盖）'))
    print(f'操作日誌：{log_path}（追加寫入，JSONL）')
    _print_state(server, '初始設備狀態')
    print()

    results = []
    if args.audio:
        audio = _resolve(base, args.audio)
        if not audio.exists():
            print(f'[错误] 音频文件不存在：{audio}', file=sys.stderr)
            return 2
        item = {'audio_path': str(audio), 'text': '', 'group': '', 'voice': '',
                'mode': 'single'}
        result = server.process(audio)
        print('單條音頻識別')
        print('-' * 96)
        _event_block(1, 1, item, result, show_scores=True)
        results.append({'source': audio.name, 'result': result})
    else:
        catalog = build_demo_catalog(base)
        if catalog.get('warnings'):
            for warning in catalog['warnings']:
                print(f'[提示] {warning}')
        if not catalog.get('commands'):
            print('[错误] 没有可用的测试音频，无法运行序列演示；'
                  '请检查 data/splits 与 data/tts_corpus。', file=sys.stderr)
            return 2
        items = _sequence_items(catalog)
        interactive = bool(args.interactive)
        print(f'演示序列：{len(items)} 條（六類命令 + 未知指令）'
              + ('，交互模式' if interactive else ''))
        print('-' * 96)
        aborted = False
        for index, item in enumerate(items, 1):
            if interactive:
                try:
                    answer = input(f'[{index}/{len(items)}] 按 Enter 處理「'
                                   f'{item.get("label") or "未知指令"}」，q 退出：')
                except (EOFError, KeyboardInterrupt):
                    print()
                    answer = 'q'
                if answer.strip().lower() in ('q', 'quit', 'exit'):
                    print('已提前結束演示。')
                    aborted = True
                    break
            try:
                result = server.process(item['audio_path'])
            except Exception as exc:
                print(f'[错误] 處理失敗：{exc}', file=sys.stderr)
                return 2
            _event_block(index, len(items), item, result, show_scores=args.show_scores)
            print()
            results.append({'source': item.get('group') or Path(item['audio_path']).name,
                            'result': result})
        if aborted:
            print('（部分命令未執行，日誌僅記錄已執行條目。）')

    if results:
        _print_summary(results)
    print()
    _print_state(server, '最終設備狀態')
    if log_path.exists():
        lines = [ln for ln in log_path.read_text(encoding='utf-8').splitlines() if ln.strip()]
        print(f'日誌文件：{log_path}（累計 {len(lines)} 條）')
    print()
    print('演示完成。Web 演示台：python scripts/run_web_demo.py')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

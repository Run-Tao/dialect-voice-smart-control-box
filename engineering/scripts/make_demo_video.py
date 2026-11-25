# -*- coding: utf-8 -*-
"""生成演示动画（GIF）：跑真实识别链路，把大字卡渲染成可播放的演示产物。

背景（项目审查 2026-11-19）：3 分钟演示视频是评分项，仓库里没有任何 mp4/gif。
本机 ffmpeg 与 cv2 视频编码不可用（缺 OpenH264），但 PIL 原生支持 GIF 且中文字体齐全，
故用 GIF 作为可播放的演示产物——它展示的是**真实识别结果**（复用 run_demo 的引擎与语料），
不是静态图，可直接插入 PPT 或用浏览器/看图软件播放。

用法：
    python scripts/make_demo_video.py --out docs/演示_六类命令.gif
    python scripts/make_demo_video.py --mode replay --audio <wav目录> --limit 6

说明：
    - 每条命令用 run_demo.one_per_command() 取一条样本，走真实 DialectEngine：
      模型 v32b + 双信号门限 + 状态先验三态决策，与部署同口径。
    - 帧内容 = 片头（口径声明）→ 逐条大字卡（真实 command/置信度/耗时/是否执行）→ 汇总。
    - 若需带声音的真人演示视频，请用 run_demo.py --mode demo 配合系统录屏工具录制；
      本脚本产出的是无声、可程序化复现的动画版本，适合作为答辩兜底材料。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

import run_demo  # noqa: E402  （同目录）

# 系统中文字体（Windows 自带；找不到则退回默认）
FONT_CANDIDATES = [
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/msyhbd.ttc',
    'C:/Windows/Fonts/simhei.ttf',
]

W, H = 960, 540
BG = (18, 22, 34)
FG = (240, 244, 252)
ACCENT = (86, 180, 233)
OK = (86, 200, 140)
REJECT = (230, 110, 110)


def load_font(size):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def centered(draw, y, text, font, fill=FG):
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (W - (bbox[2] - bbox[0])) / 2
    draw.text((x, y), text, font=font, fill=fill)


def frame(title, lines, badge=None, badge_color=ACCENT):
    """渲染一帧：标题 + 若干居中行 + 可选徽标。"""
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    f_title = load_font(46)
    f_body = load_font(30)
    f_badge = load_font(26)
    centered(d, 70, title, f_title, FG)
    d.line([(W * 0.2, 140), (W * 0.8, 140)], fill=ACCENT, width=3)
    y = 190
    for line in lines:
        centered(d, y, fit_text(line, f_body), f_body, FG)
        y += 62
    if badge:
        bbox = d.textbbox((0, 0), badge, font=f_badge)
        bw = bbox[2] - bbox[0] + 48
        x0 = (W - bw) / 2
        d.rounded_rectangle([x0, H - 150, x0 + bw, H - 88], radius=16, fill=badge_color)
        centered(d, H - 138, badge, f_badge, (255, 255, 255))
    return im


def fit_text(text, font, max_width=W - 80):
    """过长文本截断加省略号，避免两侧溢出画面。"""
    if not text:
        return text
    bbox = font.getbbox(text)
    if bbox[2] - bbox[0] <= max_width:
        return text
    out = text
    while out and font.getbbox(out + '…')[2] > max_width:
        out = out[:-1]
    return out + '…'


def intro_frame(model_name):
    return frame('方言语音智能家居指令识别', [
        '粤语家居命令 · 六类识别 · 双信号拒识',
        f'当前模型 {model_name}  |  2026-11-19',
        '全部数字为 edge-tts 合成粤语口径',
    ], badge='演示开始')


def summary_frame(total, accepted, mean_ms):
    return frame('演示汇总', [
        f'共 {total} 条命令',
        f'成功执行 {accepted} 条',
        f'平均端到端 {mean_ms:.0f} ms',
    ], badge=('演示结束' if accepted == total else '部分拒识'),
        badge_color=(OK if accepted == total else REJECT))


def collect_results(cfg, args):
    """复用 run_demo 的引擎与语料，返回 [(command, result, elapsed_ms)]。"""
    engine = run_demo.build_engine(cfg, args)
    from runtime.audio_io import load_wav
    from src.vad import EnergyVad

    out = []
    for command, path in run_demo.one_per_command():
        audio, sr = load_wav(path)
        vad = EnergyVad(fs=sr)
        segs = vad.feed(audio) + vad.flush()
        if not segs:
            continue
        t0 = time.perf_counter()
        result = engine.process_utterance(segs[0], sr)
        elapsed = (time.perf_counter() - t0) * 1000.0
        out.append((command, result, elapsed, path.name))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(ROOT / 'docs' / '演示_六类命令.gif'))
    ap.add_argument('--config', default=None)
    ap.add_argument('--frame-ms', type=int, default=1800, help='每帧停留毫秒')
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)
    # build_engine 需要 --no-speak 等字段
    args.no_speak = True

    results = collect_results(cfg, args)
    if not results:
        print('没有识别到任何样本，请先确认语料目录存在。')
        return 2

    frames = [intro_frame(Path(cfg.model_path).name)]
    total = len(results)
    accepted = 0
    elapsed_all = []
    for i, (command, result, elapsed, wav_name) in enumerate(results, 1):
        ok = bool(result.accepted)
        accepted += int(ok)
        elapsed_all.append(elapsed)
        pred = getattr(result, 'command', '') or '未识别'
        conf = float(getattr(result, 'confidence', 0.0))
        # 样本名缩短：取前 40 字符，避免长文件名溢出画面
        short_name = wav_name if len(wav_name) <= 40 else wav_name[:38] + '…'
        frames.append(frame(
            f'识别 {i}/{total}',
            [f'样本：{short_name}',
             f'识别结果：{pred}',
             f'置信度 {conf:.3f}   耗时 {elapsed:.0f} ms'],
            badge=('已执行' if ok else '已拒识'),
            badge_color=(OK if ok else REJECT),
        ))
    mean_ms = sum(elapsed_all) / len(elapsed_all)
    frames.append(summary_frame(total, accepted, mean_ms))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=args.frame_ms, loop=0, dispose=2)
    print(f'演示动画已生成: {out}（{len(frames)} 帧，{total} 条命令，成功 {accepted}）')
    print(f'模型 {Path(cfg.model_path).name}，平均耗时 {mean_ms:.0f} ms')
    print('如需带声音版本：python scripts/run_demo.py --mode demo 配合系统录屏工具录制')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

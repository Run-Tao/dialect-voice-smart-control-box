# -*- coding: utf-8 -*-
"""生成《方言智控盒 第三阶段汇报》PPTX（16:9，10–14 页，中文）。

数据来源（只读）：
    outputs/v3_experiments/main_v3_1/results.json
    outputs/v3_experiments/main_v3_1/rejection.json
    outputs/v3_experiments/main_v3_1/train_report.json
    outputs/v3_data_audit.json, outputs/v3_1_data_audit.json
若 results.json 尚不存在，结果类页面会显示「待补」，仍可预览版式。

用法::

    D:\\anaconda3\\python.exe docs/report_assets/build_v3_pptx.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / 'outputs' / 'v3_experiments' / 'main_v3_1'

NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x5C, 0x9A)
TEAL = RGBColor(0x1F, 0x7A, 0x73)
ORANGE = RGBColor(0xC5, 0x5A, 0x11)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = '微软雅黑'

W, H = Inches(13.333), Inches(7.5)


# --------------------------------------------------------------------------- 基础工具
def load(name):
    p = OUT_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _set_font(run, size=18, bold=False, color=GRAY, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    # 中文（eastAsia）字体
    from lxml import etree
    ns = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    for tag in ('ea', 'cs'):
        el = rPr.find(f'{{{ns}}}{tag}')
        if el is None:
            el = etree.SubElement(rPr, f'{{{ns}}}{tag}')
        el.set('typeface', name)


def add_text(slide, text, x, y, w, h, size=18, bold=False, color=GRAY,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.15):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    r = p.add_run()
    r.text = text
    _set_font(r, size, bold, color)
    return box


def add_bullets(slide, items, x, y, w, h, size=17, color=GRAY, gap=10):
    """items: list[str] 或 list[(level, str)]；level 0/1。"""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    first = True
    for it in items:
        if isinstance(it, tuple):
            level, text = it
        else:
            level, text = 0, it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = 1.18
        prefix = '▪ ' if level == 0 else '– '
        r = p.add_run()
        r.text = ('    ' if level else '') + prefix + text
        _set_font(r, size if level == 0 else size - 2, False, color if level == 0 else RGBColor(0x70, 0x70, 0x70))
    return box


def add_rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, shadow=False):
    shp = slide.shapes.add_shape(shape, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = shadow
    return shp


def new_slide(prs, title, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # 顶部色带
    add_rect(slide, 0, 0, W, Inches(0.16), NAVY)
    add_text(slide, title, Inches(0.62), Inches(0.42), Inches(11.2), Inches(0.9),
             size=28, bold=True, color=NAVY)
    if kicker:
        add_text(slide, kicker, Inches(0.62), Inches(1.18), Inches(12.0), Inches(0.5),
                 size=14, color=BLUE)
    # 页脚
    add_rect(slide, Inches(0.62), Inches(7.02), Inches(12.1), Emu(12700), RGBColor(0xD0, 0xD7, 0xE2))
    add_text(slide, '方言智控盒 · 第三阶段汇报 · 2026-10-21', Inches(0.62), Inches(7.1),
             Inches(9), Inches(0.3), size=10, color=RGBColor(0x8C, 0x8C, 0x8C))
    return slide


def add_kpi(slide, items, x, y, w_total, h=Inches(1.55), gap=Inches(0.22)):
    """items: list[(数值, 标签, 颜色)]"""
    n = len(items)
    w = int((w_total - gap * (n - 1)) / n)
    for i, (value, label, color) in enumerate(items):
        sx = x + i * (w + gap)
        add_rect(slide, sx, y, w, h, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_text(slide, value, sx, y + Inches(0.18), w, Inches(0.7), size=27, bold=True,
                 color=color, align=PP_ALIGN.CENTER)
        add_text(slide, label, sx + Inches(0.08), y + Inches(0.92), w - Inches(0.16), Inches(0.55),
                 size=12, color=GRAY, align=PP_ALIGN.CENTER)


def add_table(slide, rows, x, y, w, h, col_widths=None, size=12, header_size=12,
              header_fill=NAVY, first_col_bold=False):
    nrows, ncols = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nrows, ncols, x, y, w, h).table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            gt.columns[i].width = int(w * cw / total)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = gt.cell(ri, ci)
            cell.text = ''
            cell.margin_left = Emu(45720)
            cell.margin_right = Emu(45720)
            cell.margin_top = Emu(18288)
            cell.margin_bottom = Emu(18288)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if ci > 0 or ri == 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            if ri == 0:
                _set_font(r, header_size, True, WHITE)
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            else:
                _set_font(r, size, first_col_bold and ci == 0, GRAY)
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
    return gt


def add_chart(slide, categories, series, x, y, w, h, colors=None, title=None):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series:
        data.add_series(name, values)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, w, h, data)
    chart = gf.chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.font.size = Pt(12)
    chart.font.name = FONT
    if title:
        chart.has_title = True
        chart.chart_title.text_frame.text = title
        for p in chart.chart_title.text_frame.paragraphs:
            for r in p.runs:
                _set_font(r, 14, True, NAVY)
    if colors:
        for i, c in enumerate(colors):
            if i < len(chart.series):
                chart.series[i].format.fill.solid()
                chart.series[i].format.fill.fore_color.rgb = c
    try:
        chart.value_axis.has_major_gridlines = True
        chart.value_axis.minimum_scale = 0.0
        chart.value_axis.maximum_scale = 1.0
    except Exception:
        pass
    return chart


def fmt_acc(x):
    return f'{x*100:.1f}%' if isinstance(x, (int, float)) else '—'


def _ci(rej):
    ci = (rej or {}).get('far_ci') or []
    if len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
        return f'{ci[0]:.3f}, {ci[1]:.3f}'
    return '待补'


def fmt4(x):
    return f'{x:.4f}' if isinstance(x, (int, float)) else '待补'


# --------------------------------------------------------------------------- 数据提取
def extract_numbers(results, rejection, train_report):
    d = {}
    if train_report:
        d['temp'] = train_report.get('temperature')
        d['train_seconds'] = train_report.get('train_seconds')
        d['ensemble_val'] = (train_report.get('ensemble_val') or {}).get('accuracy')
        d['seeds'] = [s.get('seed') for s in train_report.get('seeds', [])]
        d['seed_val'] = [(s.get('seed'), s.get('best_val_accuracy'), s.get('best_epoch'))
                         for s in train_report.get('seeds', [])]
    if results:
        for key, short in [('E1_v2_dtw_baseline', 'E1'), ('E2_classical_logreg', 'E2'),
                           ('E3_cnn_ensemble', 'E3'), ('E4_cnn_tta', 'E4'),
                           ('E5_fusion', 'E5')]:
            item = results.get(key) or {}
            full = item.get('full_test') or {}
            matched = item.get('matched_v2_test') or {}
            val = item.get('val') or {}
            d[short] = {
                'acc': full.get('accuracy'), 'f1': full.get('macro_f1'),
                'matched_acc': matched.get('accuracy'), 'matched_f1': matched.get('macro_f1'),
                'val_acc': val.get('accuracy', item.get('val_accuracy')),
                'n': full.get('n'),
            }
        # E4 的 matched-v2 未存进 results.json，用预测 CSV 复算（与报告口径一致）
        if (d.get('E4') or {}).get('matched_acc') is None:
            _csv = OUT_DIR / 'predictions_E4_cnn_tta_final.csv'
            if _csv.exists():
                import csv as _c
                with open(_csv, encoding='utf-8', newline='') as _f:
                    _rows = list(_c.DictReader(_f))
                _v2 = {'text_morphology', 'rate_slow', 'legacy', 'baseline_v1', ''}
                _sel = [r for r in _rows if r['condition'] in _v2]
                if _sel:
                    d['E4']['matched_acc'] = sum(int(r['correct']) for r in _sel) / len(_sel)
        d['stats'] = results.get('E6_statistics') or {}
        d['e9'] = results.get('E9_latency_size') or {}
        d['e10'] = results.get('E10_error_analysis') or {}
        d['e4_candidates'] = (results.get('E4_cnn_tta') or {}).get('val_candidates') or {}
        d['e4_selected'] = (results.get('E4_cnn_tta') or {}).get('selected_on_val')
        d['e5_weight'] = ((results.get('E5_fusion') or {}).get('selected_on_val') or {}).get('cnn_weight')
    if rejection:
        tr = rejection.get('test_results') or {}
        d['rej'] = {
            'threshold': rejection.get('selected_threshold'),
            'coverage': tr.get('known_coverage'),
            'false_reject': tr.get('known_false_reject_rate'),
            'accepted_acc': tr.get('accepted_accuracy'),
            'no_thr_acc': tr.get('no_threshold_accuracy'),
            'far': tr.get('unknown_false_accept_rate'),
            'far_ci': tr.get('unknown_far_wilson95'),
            'rule3': tr.get('far_rule_of_three_upper95'),
            'n_unknown': tr.get('n_unknown_test'),
            'hard_far': tr.get('hard_negative_far'),
            'by_label': tr.get('by_label') or {},
            'by_type': tr.get('by_negative_type') or {},
            'per_text': rejection.get('unknown_test_per_text') or {},
            'val': rejection.get('selected_on_validation') or {},
        }
    return d


# --------------------------------------------------------------------------- 各页
def slide_cover(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(s, 0, 0, W, H, NAVY)
    add_rect(s, Inches(0.9), Inches(1.55), Inches(0.10), Inches(2.5), RGBColor(0xE8, 0xB0, 0x4B))
    add_text(s, '方言智控盒', Inches(1.25), Inches(1.45), Inches(10.5), Inches(1.1),
             size=54, bold=True, color=WHITE)
    add_text(s, '第三阶段汇报', Inches(1.25), Inches(2.55), Inches(10.5), Inches(0.9),
             size=40, bold=True, color=RGBColor(0xDC, 0xE6, 0xF2))
    add_text(s, '条件多样性扩充 · 判别式模型 · 端到端原型', Inches(1.25), Inches(3.55),
             Inches(11.0), Inches(0.6), size=22, color=RGBColor(0xB8, 0xCC, 0xE4))
    add_rect(s, Inches(1.25), Inches(4.45), Inches(6.2), Emu(19050), RGBColor(0x8F, 0xA9, 0xCC))
    add_text(s, '2026-10-21  ·  项目根目录  D:\\Run_Tao\\DialectVoice_Research',
             Inches(1.25), Inches(4.65), Inches(11.0), Inches(0.5), size=15,
             color=RGBColor(0xC9, 0xD6, 0xE6))
    add_text(s, '内部技术汇报 · 全部语料为 TTS 合成 / 合成变换，结论不外推到真人场景',
             Inches(1.25), Inches(5.15), Inches(11.5), Inches(0.5), size=13,
             color=RGBColor(0x9F, 0xB3, 0xCC))
    add_text(s, '数据 v3 / v3.1（2904 条）· log-mel+Δ + 1D-CNN（3 种子集成）· 纯 numpy 推理',
             Inches(1.25), Inches(6.35), Inches(11.5), Inches(0.5), size=13,
             color=RGBColor(0x9F, 0xB3, 0xCC))
    return s


def slide_summary(prs, d):
    s = new_slide(prs, '结论先行：分类突破，拒识与复杂句式未解决',
                  '所有数字来自 results.json / rejection.json；E3 为部署件（3 种子 CNN 集成）')
    e1, e3, e5 = d.get('E1'), d.get('E3'), d.get('E5')
    stats = d.get('stats', {})
    st = stats.get('E1_vs_E3_full') or {}
    p = st.get('mcnemar_exact_p_two_sided')
    rej = d.get('rej') or {}
    e10 = d.get('e10') or {}
    add_kpi(s, [
        (fmt_acc((e1 or {}).get('acc')), 'E1 二期 DTW 基线\n（full 483）', GRAY),
        (fmt_acc((e5 or {}).get('acc')), 'E5 报告主结果\n（full 483）', TEAL),
        (f'p={p:.0e}' if isinstance(p, (int, float)) else 'p=待补', 'E1 → E3 配对 McNemar\n（双侧精确检验）', ORANGE),
        (fmt_acc(e10.get('correction_register_accuracy')), 'correction 语域\n（test 98 条）', BLUE),
        (fmt_acc(rej.get('far')), '未知误接受率\n（test 36 条）', ORANGE),
    ], Inches(0.62), Inches(1.65), Inches(12.1))
    add_bullets(s, [
        f"闭集显著提升：E1 0.5901 → E3 0.9048（+31.47pp，p<1e-31）；与二期同音频的 matched-v2：0.5495 → 0.8829；",
        f"开集失败：未知指令误接受率 {fmt_acc(rej.get('far'))}（95%CI [{_ci(rej)}]）；覆盖率≥0.85 约束下验证集可达最低 FAR 仍约 36%–39%；",
        '数据消融 v3→v3.1（同种子）：full 0.8778→0.9130（+3.52pp，p=2.2e-4）；但两个窗帘 correction 组仍失败（0/19、2/15）；错误 100% 是对立对（窗帘 33/41）；',
        '部署件 E3：1.351 MB、前向约 16 ms/条（E5 虽为实验最佳，但为两模型融合，未部署）；TTA 在 val 赢、test 降。',
    ], Inches(0.62), Inches(3.55), Inches(12.1), Inches(2.9), size=15, gap=8)
    return s

def slide_v2(prs):
    s = new_slide(prs, '二期：一个被如实报告的负结果', '扩数据没有提升 → 先证明瓶颈不是数据量，再决定三期改什么')
    add_kpi(s, [
        ('0.5766', 'v1 历史数据\nTest Acc', GRAY),
        ('0.5495', '扩充到 600 条后\nTest Acc', GRAY),
        ('p=0.7359', '配对 McNemar\n（差异不显著）', ORANGE),
        ('80%', '错误集中在\n语义对立意图对', BLUE),
    ], Inches(0.62), Inches(1.65), Inches(12.1))
    add_bullets(s, [
        '数据规模曲线：训练样本 86→397，测试准确率 0.55–0.59 波动，无单调上升；模板策略 / 模板数也没有可靠增益；',
        '真正的瓶颈是方向词：CURTAIN_OPEN↔CURTAIN_CLOSE、LIGHT_ON↔LIGHT_OFF 占 80% 错误；',
        '声学佐证：跨意图最高能量包络余弦 0.9882（两句只差一个动词），整句 DTW 把局部差异淹没了；',
        '二期遗留的方法论缺口：条件多样性（语速/音高）「计划未实现」，验证集只有 92 条；',
        '所以三期不重复「再扩数据」，而是补条件多样性 + 换判别式模型 + 把统计检验口径固定。',
    ], Inches(0.62), Inches(3.55), Inches(12.1), Inches(3.0), size=15, gap=8)
    return s


def slide_route(prs):
    s = new_slide(prs, '三阶段路线：三条主线，各自可审计', '数据补条件、模型换判别式、评估与原型把结论钉死')
    cols = [
        ('A · 数据', BLUE, [
            'v3：+4 个条件变体（±18% 语速、±15 Hz 音高）',
            '网络失败率 >5% → 离线信号级增广（预登记回退）',
            'v3.1：+18 条 correction 表达，只进 train',
            '冻结 v2/v3 划分与哈希，测试集组数不动',
        ]),
        ('B · 模型', TEAL, [
            'log-mel(40)+Δ，整句 CMVN，256 帧',
            '1D-CNN（3 层卷积 + mean/max 池化）',
            '3 种子概率平均 + 验证集温度标定',
            '推理纯 numpy 导出，不依赖 PyTorch',
        ]),
        ('C · 评估与原型', ORANGE, [
            'E1–E6/E9/E10：同一 test 的配对对照',
            'McNemar + 按表达组自助法 95%CI',
            '拒识阈值只在 val + 未知校准集选',
            'CLI / Flask 演示台 + 全量单元测试',
        ]),
    ]
    x = Inches(0.62)
    for title, color, items in cols:
        add_rect(s, x, Inches(1.72), Inches(3.9), Inches(4.55), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_rect(s, x, Inches(1.72), Inches(3.9), Inches(0.62), color, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_text(s, title, x, Inches(1.82), Inches(3.9), Inches(0.45), size=18, bold=True,
                 color=WHITE, align=PP_ALIGN.CENTER)
        add_bullets(s, items, x + Inches(0.2), Inches(2.55), Inches(3.5), Inches(3.6), size=13.5, gap=9)
        x += Inches(4.12)
    return s


def slide_data_fallback(prs):
    s = new_slide(prs, '数据 v3：一次「按预登记规则」的回退', '目标 2088 条条件变体；实测 edge-tts 失败率 22%–42%，触发离线增广')
    add_table(s, [
        ['并发', '请求', '失败', '失败率', '吞吐（条/s）'],
        ['6', '24', '10', '41.7%', '0.121'],
        ['3', '9', '2', '22.2%', '0.078'],
        ['1', '9', '3', '33.3%', '0.037'],
    ], Inches(0.62), Inches(1.75), Inches(5.6), Inches(1.7), col_widths=[1, 1, 1, 1.2, 1.5], size=13, header_size=13)
    add_text(s, '规则：失败率 >5% 立即改为离线信号级增广（在跑正式批次前已预登记）',
             Inches(0.62), Inches(3.6), Inches(5.8), Inches(0.6), size=13, color=ORANGE)
    add_bullets(s, [
        '2088 条条件变体 → WSOLA 变速（保音高）/ 重采样变调 + WSOLA 还原时长，确定性、可复算；',
        '真实 TTS 只用于无法离线变换的 168 条：54 新表达 + 72 未知集 + 42 WanLung 缺口源；',
        '结果：主清单 2796 行 / 192 组 / 每类 466 条；离线增广 2142 条，offline_signal_aug 可逐条回溯源样本；',
        '代价（必须一起讲）：增广样本与源音频高度相关，有效样本量远小于 2904，不能当作独立采样。',
    ], Inches(6.55), Inches(1.75), Inches(6.15), Inches(4.4), size=14, gap=9)
    return s


def slide_data_cond(prs):
    s = new_slide(prs, '数据 v3：条件变体是「实测」不是「声明」', '逐条重读磁盘音频：时长比、f0 偏移、sha 与源相同数、格式违规')
    add_table(s, [
        ['条件', '条数', '期望时长比', '实测时长比', '2% 内', 'f0 偏移（Hz）'],
        ['rate_slow_18', '576', '1.21951', '1.2195', '576/576', '—'],
        ['rate_fast_18', '522', '0.84746', '0.8475', '522/522', '—'],
        ['pitch_low', '522', '1.00000', '1.0000', '522/522', '−14.61'],
        ['pitch_high', '522', '1.00000', '1.0000', '522/522', '+15.13'],
    ], Inches(0.62), Inches(1.8), Inches(7.4), Inches(2.4), col_widths=[1.6, 0.8, 1.2, 1.2, 1.0, 1.2],
        size=13, header_size=13)
    add_bullets(s, [
        '所有 4 个条件的时长比全部命中目标；变调条件 f0 中位趋势接近 ±15 Hz；',
        '「与源 sha 相同」恒为 0 → 变换确实产生新音频；格式违规 0；',
        '但这是客观信号指标：证明「变换发生且幅度正确」，不能证明变换后语音自然、可懂、等价于真人慢说/变调；',
        '该限制在报告 §3.6 与 §8.1 明确写出，避免把「条件多样性」说过头。',
    ], Inches(8.2), Inches(1.8), Inches(4.5), Inches(4.4), size=13.5, gap=9)
    add_text(s, '注：f0 用自相关逐条估计（60–400 Hz，取浊音帧中位数），个别清音/噪声帧会有离群值，故看 p05–p95 趋势。',
             Inches(0.62), Inches(4.45), Inches(7.4), Inches(0.8), size=11, color=RGBColor(0x80, 0x80, 0x80))
    return s


def slide_data_v31(prs):
    s = new_slide(prs, '数据 v3.1：只补训练集 correction 语域', 'v3 CNN 总体 0.913，但 6 个 correction 测试组只有 0.622，两个窗帘组 100% 判反')
    add_kpi(s, [
        ('0.913', 'v3 CNN\ntest 总体 Acc', GRAY),
        ('0.622', 'correction 语域\n（6 组 / 98 条）', ORANGE),
        ('6→24', 'train correction\n组数', TEAL),
        ('0.4444', '新文本 vs test\n最大 2-gram Jaccard', BLUE),
    ], Inches(0.62), Inches(1.65), Inches(12.1), h=Inches(1.5))
    add_bullets(s, [
        '18 条新 correction 表达（每类 3 条，含反方向干扰词）× 3 音色 × {真实 TTS baseline, rate_slow_18} = 108 条，只进 train；',
        '测试集 31 组 / 483 条一个样本都没动；v3 manifest、v3 划分、v2 划分三个冻结文件 sha256 复核 unchanged=true；',
        '新文本与 test 6 条 correction 文本 Jaccard 最大 0.4444 < 0.5（违反 0 条）→ 测试集仍是「未见表达」；',
        '代价：新增文本与 train 已有 correction 近似（最大 ≈0.625），属训练集内部冗余，不构成泄漏但会降低有效信息量。',
    ], Inches(0.62), Inches(3.55), Inches(12.1), Inches(2.9), size=14.5, gap=9)
    return s


def slide_model(prs):
    s = new_slide(prs, '模型：把「方向词」交给判别式模型', '特征 → CNN → 3 种子集成 → 温度标定 → 纯 numpy 推理导出')
    steps = [
        ('log-mel 40 + Δ', '整句 CMVN\n线性插值 256 帧\ndim=80'),
        ('1D-CNN', '3×Conv1d + BN\nmean/max 池化\n118k 参数/种子'),
        ('3 种子集成', 'seeds 42/7/2026\n概率平均\n降低方差'),
        ('温度标定', 'val 拟合 T\n只改置信度\n不改 argmax'),
        ('numpy 推理', 'npz 工件\n无 torch 依赖\n毫秒级前向'),
    ]
    x = Inches(0.62)
    for i, (title, desc) in enumerate(steps):
        color = [BLUE, TEAL, BLUE, ORANGE, NAVY][i]
        add_rect(s, x, Inches(1.8), Inches(2.2), Inches(1.95), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_text(s, title, x, Inches(1.95), Inches(2.2), Inches(0.5), size=15, bold=True,
                 color=color, align=PP_ALIGN.CENTER)
        add_text(s, desc, x + Inches(0.1), Inches(2.5), Inches(2.0), Inches(1.1), size=11.5,
                 color=GRAY, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            add_text(s, '→', x + Inches(2.16), Inches(2.5), Inches(0.32), Inches(0.5),
                     size=20, bold=True, color=RGBColor(0xB0, 0xB0, 0xB0))
        x += Inches(2.48)
    add_bullets(s, [
        '特征与二期同帧网格；features_v3.py 在 include_delta=False 时与旧 MFCC 数值等价（有测试），换特征不是黑箱；',
        '训练用 PyTorch（AdamW、cosine、90 epoch、patience=12、SpecAugment + 时间平移 + 随机增益）；推理零 PyTorch 依赖；',
        '温度标定在验证集上做，只让「概率」更像概率；拒识阈值读 v3_thresholds.json，推理时不选任何参数；',
        '模型保持小规模（单种子 0.118 M 参数）以适配离线部署，未做更深网络消融。',
    ], Inches(0.62), Inches(4.05), Inches(12.1), Inches(2.5), size=14, gap=8)
    return s


def slide_protocol(prs, d):
    s = new_slide(prs, '实验协议：同一固定 test，五个对照', 'E1 旧方法 / E2 经典判别式 / E3 CNN / E4 +TTA / E5 融合')
    e4sel = d.get('e4_selected')
    e4cand = d.get('e4_candidates') or {}
    e5w = d.get('e5_weight')
    rows = [
        ['实验', '模型', '训练数据', '选择（仅 val）'],
        ['E1', 'MFCC13+CMVN + DTW 全模板（max_frames=60）', 'v2 train 397 条', '固定协议，无选择'],
        ['E2', 'MFCC+Δ+ΔΔ 统计嵌入 + 多项逻辑回归', 'v3.1 train', 'l2 网格（按 val 选）'],
        ['E3', 'log-mel40+Δ + 1D-CNN ×3 seeds', 'v3.1 train', '早停看 val acc'],
        ['E4', 'E3 + TTA（时间平移）', 'v3.1 train', 'TTA 配置：' + (str(e4sel) if e4sel else '待补')],
        ['E5', 'E3 ⊕ E2 概率融合', 'v3.1 train', 'CNN 权重：' + (f'{e5w:.1f}' if isinstance(e5w, (int, float)) else '待补')],
    ]
    add_table(s, rows, Inches(0.62), Inches(1.75), Inches(12.1), Inches(2.75),
              col_widths=[0.7, 4.2, 1.9, 2.6], size=12.5, header_size=12.5, first_col_bold=True)
    add_bullets(s, [
        '两个测试视图：full = 483 条（含条件变体）；matched-v2 = 111 条（与二期同音频，可直接对比）；',
        '所有 test 数字都报告以便审计，但结论以 val 选择为准；若 test 更高的项未被选中，照实写出且不据此改选；',
        'E5 融合用的是无 TTA 的 CNN 概率 → 若 E5 低于 E4，只能说明「权重是在无 TTA 概率上选的」，不能说融合有害。',
    ], Inches(0.62), Inches(4.7), Inches(12.1), Inches(1.9), size=13.5, gap=8)
    return s


def slide_results(prs, d):
    s = new_slide(prs, '主结果：E1–E5（同一固定 test）', '表格与柱图数字均来自 results.json；full=483，matched-v2=111')
    rows = [['实验', 'Test Acc（full）', 'Test Macro-F1', 'Acc（matched-v2）', 'Val Acc']]
    labels = {'E1': 'E1 二期 DTW', 'E2': 'E2 经典 logreg', 'E3': 'E3 CNN 集成',
              'E4': 'E4 CNN+TTA', 'E5': 'E5 融合'}
    colors = {'E1': GRAY, 'E2': ORANGE, 'E3': TEAL, 'E4': TEAL, 'E5': BLUE}
    for k in ['E1', 'E2', 'E3', 'E4', 'E5']:
        it = d.get(k) or {}
        rows.append([labels[k], fmt_acc(it.get('acc')), fmt4(it.get('f1')),
                     fmt_acc(it.get('matched_acc')), fmt_acc(it.get('val_acc'))])
    add_table(s, rows, Inches(0.62), Inches(1.8), Inches(6.3), Inches(2.4),
              col_widths=[1.35, 1.25, 1.15, 1.45, 0.85], size=12, header_size=11.5)
    cats = [labels[k] for k in ['E1', 'E2', 'E3', 'E4', 'E5']]
    series = [('Test Acc（full 483）', [(d.get(k) or {}).get('acc') or 0 for k in ['E1', 'E2', 'E3', 'E4', 'E5']]),
              ('Test Acc（matched-v2 111）', [(d.get(k) or {}).get('matched_acc') or 0 for k in ['E1', 'E2', 'E3', 'E4', 'E5']])]
    add_chart(s, cats, series, Inches(6.95), Inches(1.65), Inches(5.8), Inches(2.85),
              colors=[TEAL, BLUE])
    e9 = d.get('e9') or {}
    temp = d.get('temp')
    train_s = d.get('train_seconds')
    infer_ms = (e9.get('inference_sec_per_sample') or 0) * 1000
    e9_txt = (f"延迟/体积（E9）：模型 {e9.get('model_size_mb', '待补')} MB，"
              f"CNN 前向 {infer_ms:.2f} ms/条（不含特征提取与 IO）")
    train_txt = (f"温度 T = {temp:.3f}" if isinstance(temp, (int, float)) else '温度 T = 待补')
    # 训练 wall time 受机器休眠影响，不在汇报中作为性能数字展示
    add_bullets(s, [
        e9_txt,
        train_txt,
        '解读纪律：先看 McNemar / CI（下一页），再看点估计；CI 重叠不宣称「更好」。',
    ], Inches(0.62), Inches(4.7), Inches(12.1), Inches(2.0), size=13.5, gap=8)
    return s


def slide_stats(prs, d):
    s = new_slide(prs, '统计检验与分层：提升是否真实', '配对 McNemar 精确检验 + 按表达组自助法 95%CI；分层看条件/语域/音色')
    stats = d.get('stats') or {}
    rows = [['比较', 'McNemar p', 'ΔAcc', 'ΔAcc 95%CI', '判定']]
    label = {'E1_vs_E3_full': 'E1→E3（full）', 'E1_vs_E3_matched_v2': 'E1→E3（matched-v2）',
             'E2_vs_E3_full': 'E2→E3（full）', 'E2_vs_E3_matched_v2': 'E2→E3（matched-v2）'}
    for key, name in label.items():
        st = stats.get(key)
        if not st:
            continue
        gb = st.get('group_bootstrap') or {}
        ci = gb.get('ci95') or [None, None]
        rows.append([name, fmt4(st.get('mcnemar_exact_p_two_sided')),
                     f"{gb.get('acc_diff_mean'):+.4f}" if isinstance(gb.get('acc_diff_mean'), (int, float)) else '待补',
                     f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if isinstance(ci[0], (int, float)) else '待补',
                     '显著' if (st.get('mcnemar_exact_p_two_sided') or 1) < 0.05 else '不显著'])
    add_table(s, rows, Inches(0.62), Inches(1.8), Inches(7.3), Inches(2.2),
              col_widths=[2.2, 1.0, 1.0, 1.9, 0.9], size=12, header_size=12)
    e10 = d.get('e10') or {}
    add_bullets(s, [
        f"correction 语域准确率：{fmt_acc(e10.get('correction_register_accuracy'))}（test 98 条 / 6 组）；",
        f"错误按语域分布：{e10.get('errors_by_register') or '待补'}；",
        '分层数字（条件 / 语域 / 音色）与逐类 P/R/F1 见报告 §5 表 11；',
        '若某条件显著低于 baseline，说明「条件多样性」引入了新的分布差异，必须如实报告而不是只报总体。',
    ], Inches(8.15), Inches(1.8), Inches(4.6), Inches(3.7), size=13, gap=9)
    return s


def slide_rejection(prs, d):
    s = new_slide(prs, '拒识与误识别风险：代价要一起报', '阈值只在 val 已知 + 未知校准集选；test 只评估一次')
    rej = d.get('rej') or {}
    add_kpi(s, [
        (f"{rej.get('threshold'):.2f}" if isinstance(rej.get('threshold'), (int, float)) else '待补',
         '选中阈值', NAVY),
        (fmt_acc(rej.get('coverage')), '已知命令\n覆盖率（test）', TEAL),
        (fmt_acc(rej.get('far')), '未知误接受率\n（36 条）', ORANGE),
        (fmt_acc(rej.get('hard_far')), '硬负例\n误接受率', ORANGE),
    ], Inches(0.62), Inches(1.65), Inches(12.1), h=Inches(1.45))
    top = []
    for text, v in sorted((rej.get('per_text') or {}).items(), key=lambda kv: -kv[1].get('max_prob', 0))[:4]:
        pred = max(v.get('pred_distribution') or {'-': 0}, key=(v.get('pred_distribution') or {'-': 0}).get)
        top.append(f'{text}（最高置信度 {v.get("max_prob")}，被接受 {v.get("n_accepted")}/{v.get("n")}，主预测 {pred}）')
    add_bullets(s, [
        f"测试集已知命令：覆盖率 {fmt_acc(rej.get('coverage'))}、误拒识率 {fmt_acc(rej.get('false_reject'))}、被接受样本准确率 {fmt_acc(rej.get('accepted_acc'))}（无阈值时 {fmt_acc(rej.get('no_thr_acc'))}）；",
        f"未知测试 36 条：误接受 {fmt_acc(rej.get('far'))}，Wilson 95%CI [{_ci(rej)}]；rule-of-three 上界约 8.3% → 0% 不等于安全；",
        '最危险的未知文本（置信度最高）：' + ('；'.join(top) if top else '待补'),
        'OOD 分数有可分性（AUC 0.85–0.95）但阈值迁移失败：cos_centroid val FAR 5.6% → test 36.1%；没有一种分数能在 coverage≥0.85 下把 FAR 压到 8% 以下。',
    ], Inches(0.62), Inches(3.4), Inches(12.1), Inches(3.1), size=13.5, gap=9)
    return s


def slide_demo(prs):
    s = new_slide(prs, '工程原型：完整链路可现场演示', 'TTS 测试音频 → 识别 → 阈值判定 → 模拟设备 → 中文反馈 → JSONL 审计日志')
    chain = ['TTS 测试音频', '识别器\nv3 CNN / v2 DTW', '阈值判定\n接受 / 拒识', '模拟设备\n灯 / 窗帘', 'JSONL\n审计日志']
    x = Inches(0.62)
    for i, t in enumerate(chain):
        color = [BLUE, TEAL, ORANGE, NAVY, GRAY][i]
        add_rect(s, x, Inches(1.75), Inches(2.2), Inches(1.1), color, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        add_text(s, t, x, Inches(1.75), Inches(2.2), Inches(1.1), size=13.5, bold=True,
                 color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        if i < len(chain) - 1:
            add_text(s, '→', x + Inches(2.16), Inches(2.0), Inches(0.32), Inches(0.5),
                     size=20, bold=True, color=RGBColor(0xB0, 0xB0, 0xB0))
        x += Inches(2.48)
    add_bullets(s, [
        'CLI scripts/demo_v3.py --sequence：六类命令各 1 条 + 未知 1 条；Web scripts/run_web_demo.py（127.0.0.1:8770）六按钮 + 未知按钮 + 阈值滑块；',
        '拒识时不动设备状态；未知命令即使置信度高也拒识；同一 seq 幂等，不重复改状态；接口含 /api/health、/api/predict、/api/log、/api/reset；',
        '已验证：全量单元测试 167 项通过（其中 test_v3_control.py 35 项）；CLI/Web 实测记录在 docs/demo_v3.md §9（当时用 v2 DTW 演示模型）；',
        'v3 模型即插即用（--model data/models/v3_cnn_ensemble.npz），接口不需要改演示台代码。',
    ], Inches(0.62), Inches(3.3), Inches(12.1), Inches(3.1), size=14, gap=9)
    return s


def slide_limits(prs):
    s = new_slide(prs, '局限与下一步', '先把「不能说的」说清楚，再谈下一步')
    add_rect(s, Inches(0.62), Inches(1.7), Inches(5.85), Inches(4.6), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text(s, '局限（必须与结论一起读）', Inches(0.85), Inches(1.85), Inches(5.4), Inches(0.5),
             size=17, bold=True, color=ORANGE)
    add_bullets(s, [
        '全部为合成 TTS / 合成变换，非真人粤语；文本仅规则审查，无母语者听测；',
        '离线增广 2142 条与源高度相关，有效样本量远小于 2904；',
        '拒识失败：未知误接受 36.11%（13/36）；correction 仍 0.6429；canto 音色仅 6 条（0.667）；',
        '未知测试仅 36 条，0% 误接受 ≠ 安全（上界约 8.3%）；',
        '模拟设备、TTS 测试音频、无 VAD/流式；无独立 OOD 检测器。',
    ], Inches(0.9), Inches(2.45), Inches(5.3), Inches(3.6), size=12.5, gap=8)
    add_rect(s, Inches(6.75), Inches(1.7), Inches(5.95), Inches(4.6), LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    add_text(s, '下一步（按预期收益排序）', Inches(6.98), Inches(1.85), Inches(5.5), Inches(0.5),
             size=17, bold=True, color=TEAL)
    add_bullets(s, [
        '采集真人粤语小样本（20–30 条/类）做初步外推验证；',
        '针对 correction / 方向词做对立成对判别或关键词切片；',
        '补独立 OOD 检测器并扩大未知测试集；',
        '实时前端（VAD + 分块）与真实 ESP32-S3 串口接入；',
        '不建议继续单纯扩合成数据（二期已证明不是杠杆）。',
    ], Inches(7.03), Inches(2.45), Inches(5.4), Inches(3.6), size=12.5, gap=8)
    return s


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(BASE / 'docs' / '方言智控盒_第三阶段汇报.pptx'))
    args = ap.parse_args()

    results = load('results.json')
    rejection = load('rejection.json')
    train_report = load('train_report.json')
    if train_report is None:
        _tp = BASE / 'outputs' / 'v3_experiments' / 'train_report.json'
        if _tp.exists():
            train_report = json.loads(_tp.read_text(encoding='utf-8'))
    d = extract_numbers(results, rejection, train_report)

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    slide_cover(prs)
    slide_summary(prs, d)
    slide_v2(prs)
    slide_route(prs)
    slide_data_fallback(prs)
    slide_data_cond(prs)
    slide_data_v31(prs)
    slide_model(prs)
    slide_protocol(prs, d)
    slide_results(prs, d)
    slide_stats(prs, d)
    slide_rejection(prs, d)
    slide_demo(prs)
    slide_limits(prs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f'OK {out}  {out.stat().st_size/1024:.1f} KB  slides={len(prs.slides)}')


if __name__ == '__main__':
    main()







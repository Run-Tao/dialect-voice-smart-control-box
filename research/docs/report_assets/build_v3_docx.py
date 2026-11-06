# -*- coding: utf-8 -*-
"""把 docs/v3_report.md 转成正式可交付的 Word 报告。

用法::

    D:\\anaconda3\\python.exe docs/report_assets/build_v3_docx.py
    D:\\anaconda3\\python.exe docs/report_assets/build_v3_docx.py --md docs/v3_report.md --out docs/xxx.docx

支持的最小 Markdown 子集：标题（#/##/###/####）、段落（自动合并软换行）、
无序/有序列表、引用块（>）、围栏代码块（```）、管道表格（含表题）、粗体/行内代码、
图片（![图题](相对路径)）。不引入额外依赖，只用 python-docx。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (保留扩展位)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = Path(__file__).resolve().parents[2]
FONT_CN = '宋体'
FONT_CN_HEAD = '黑体'
FONT_MONO = 'Consolas'
FONT_UI = '微软雅黑'
COLOR_HEAD = RGBColor(0x1F, 0x38, 0x64)
COLOR_GRAY = RGBColor(0x59, 0x59, 0x59)
CJK_RE = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')


# --------------------------------------------------------------------------- 字体/样式
def set_run_font(run, name=FONT_CN, size=10.5, bold=None, color=None, mono=False):
    name = FONT_MONO if mono else name
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), FONT_MONO if mono else name)


def style_font(style, name, size, bold=None, color=None):
    style.font.name = name
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), name)


def setup_styles(doc):
    normal = doc.styles['Normal']
    style_font(normal, FONT_CN, 10.5)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.space_before = Pt(0)

    for lvl, (size, before, after) in {1: (16, 16, 8), 2: (13.5, 12, 6), 3: (12, 10, 4)}.items():
        st = doc.styles[f'Heading {lvl}']
        style_font(st, FONT_CN_HEAD, size, bold=True, color=COLOR_HEAD)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.line_spacing = 1.25

    # 代码块样式
    try:
        code = doc.styles.add_style('CodeBlock', 1)  # WD_STYLE_TYPE.PARAGRAPH
        style_font(code, FONT_MONO, 8.5)
        code.paragraph_format.line_spacing = 1.1
        code.paragraph_format.space_after = Pt(6)
        code.paragraph_format.left_indent = Cm(0.4)
    except Exception:
        pass


def add_page_number_footer(section, left_text):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(left_text + '    第 ')
    set_run_font(r, FONT_CN, 9, color=COLOR_GRAY)
    # PAGE 域
    run = p.add_run()
    set_run_font(run, FONT_CN, 9, color=COLOR_GRAY)
    fld_begin = OxmlElement('w:fldChar')
    fld_begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = 'PAGE'
    fld_sep = OxmlElement('w:fldChar')
    fld_sep.set(qn('w:fldCharType'), 'separate')
    t = OxmlElement('w:t')
    t.text = '1'
    fld_end = OxmlElement('w:fldChar')
    fld_end.set(qn('w:fldCharType'), 'end')
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(t)
    run._r.append(fld_end)
    r2 = p.add_run(' 页')
    set_run_font(r2, FONT_CN, 9, color=COLOR_GRAY)


# --------------------------------------------------------------------------- 行内格式
def add_inline(paragraph, text, base_size=10.5, base_bold=False, color=None):
    """解析 **粗体** 与 `代码`，其余按普通文本追加。"""
    token_re = re.compile(r'(\*\*.+?\*\*|`[^`]+?`)')
    pos = 0
    for m in token_re.finditer(text):
        if m.start() > pos:
            r = paragraph.add_run(text[pos:m.start()])
            set_run_font(r, FONT_CN, base_size, bold=base_bold, color=color)
        tok = m.group(0)
        if tok.startswith('**'):
            r = paragraph.add_run(tok[2:-2])
            set_run_font(r, FONT_CN, base_size, bold=True, color=color)
        else:
            r = paragraph.add_run(tok[1:-1].strip('`'))
            set_run_font(r, FONT_MONO, base_size - 0.5, color=RGBColor(0x1F, 0x4E, 0x79))
        pos = m.end()
    if pos < len(text):
        r = paragraph.add_run(text[pos:])
        set_run_font(r, FONT_CN, base_size, bold=base_bold, color=color)


def join_lines(lines):
    """合并 Markdown 软换行：中文之间不补空格，西文之间补一个空格。"""
    out = ''
    for i, ln in enumerate(lines):
        ln = ln.strip()
        if not ln:
            continue
        if not out:
            out = ln
        elif CJK_RE.match(out[-1]) or CJK_RE.match(ln[0]):
            out += ln
        else:
            out += ' ' + ln
    return out


# --------------------------------------------------------------------------- 表格
def shade_cell(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def set_repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    tblHeader.set(qn('w:val'), 'true')
    trPr.append(tblHeader)


def set_cell_margins(table, top=40, bottom=40, left=70, right=70):
    tblPr = table._tbl.tblPr
    mar = OxmlElement('w:tblCellMar')
    for tag, val in (('top', top), ('left', left), ('bottom', bottom), ('right', right)):
        el = OxmlElement(f'w:{tag}')
        el.set(qn('w:w'), str(val))
        el.set(qn('w:type'), 'dxa')
        mar.append(el)
    tblPr.append(mar)


NUMERIC_RE = re.compile(r'^[\s\d.,%+\-±()（）/–—<>=~×x:：]*\d[\s\d.,%+\-±()（）/–—<>=~×x:：]*$')


def parse_table(lines):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c.replace(' ', '')) for c in cells if c != ''):
            continue  # 分隔行
        rows.append(cells)
    ncol = max(len(r) for r in rows)
    rows = [r + [''] * (ncol - len(r)) for r in rows]
    return rows


def _disp_len(s):
    s = str(s)
    return sum(2 if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f'
               or '\uff00' <= ch <= '\uffef' else 1 for ch in s)


TOKEN_RE = re.compile(r'[A-Za-z0-9_./:\\-]{4,}')


def compute_widths(rows, total_cm=15.9, min_cm=1.05):
    ncol = len(rows[0])
    maxlen = [max(_disp_len(r[c]) for r in rows) for c in range(ncol)]
    weights = [max(min_cm, (ml ** 0.72)) for ml in maxlen]
    s = sum(weights)
    widths = [total_cm * x / s for x in weights]
    # 不可断词的 ASCII token 需要最小宽度（如 general_unknown 在 Consolas 下不换行）
    need = [0.0] * ncol
    for c in range(ncol):
        longest = 0
        for r in rows:
            for tok in TOKEN_RE.findall(str(r[c])):
                longest = max(longest, len(tok))
        if longest:
            need[c] = min(0.165 * longest + 0.34, 0.45 * total_cm)
    fixed = {i: need[i] for i in range(ncol) if need[i] > widths[i]}
    if fixed:
        remain = total_cm - sum(fixed.values())
        free = [i for i in range(ncol) if i not in fixed]
        if free and remain > 0:
            base = sum(widths[i] for i in free) or 1.0
            for i in free:
                widths[i] = remain * widths[i] / base
        widths = [max(widths[i], fixed.get(i, 0)) for i in range(ncol)]
        scale = total_cm / sum(widths)
        widths = [w * scale for w in widths]
    return widths


def add_table(doc, rows, caption=None):
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        add_inline(p, caption, base_size=9)
    ncol = len(rows[0])
    body_size = 8.0 if ncol >= 7 else (8.5 if ncol >= 5 else 9.0)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_cell_margins(table, left=55, right=55)
    widths = compute_widths(rows)
    for ci, w in enumerate(widths):
        table.columns[ci].width = Cm(w)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.width = Cm(widths[ci])
            cell.text = ''
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.line_spacing = 1.1
            if ri == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(p, val, base_size=body_size, base_bold=True)
                shade_cell(cell, 'D9E2F3')
            else:
                if NUMERIC_RE.match(val) and len(val) <= 22:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                add_inline(p, val, base_size=body_size)
    set_repeat_header(table.rows[0])
    if len(rows) <= 14:
        for row in table.rows[:-1]:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement('w:cantSplit')
        cant.set(qn('w:val'), 'true')
        trPr.append(cant)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.line_spacing = 1.0
    for r in spacer.runs:
        pass
    return table


# --------------------------------------------------------------------------- 图片
def add_image(doc, path: Path, caption: str, width_cm=15.5):
    if not path.exists():
        p = doc.add_paragraph()
        add_inline(p, f'［缺图：{caption}（{path.name}）］', base_size=9)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    add_inline(cap, caption, base_size=9, color=COLOR_GRAY)


# --------------------------------------------------------------------------- 主流程
def build(md_path: Path, out_path: Path):
    lines = md_path.read_text(encoding='utf-8-sig').splitlines()
    doc = Document()
    setup_styles(doc)

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(2.4), Cm(2.2)
    sec.left_margin, sec.right_margin = Cm(2.6), Cm(2.4)
    add_page_number_footer(sec, '方言智控盒 · 第三阶段报告')

    # ---------- 封面 ----------
    for _ in range(4):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run('方言智控盒')
    set_run_font(r, FONT_UI, 30, bold=True, color=COLOR_HEAD)
    t2 = doc.add_paragraph()
    t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t2.add_run('第三阶段执行报告')
    set_run_font(r, FONT_UI, 24, bold=True, color=COLOR_HEAD)
    t3 = doc.add_paragraph()
    t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t3.add_run('条件多样性扩充 · 判别式模型 · 端到端原型')
    set_run_font(r, FONT_UI, 15, color=COLOR_GRAY)
    for _ in range(6):
        doc.add_paragraph()
    meta = [
        ('项目根目录', r'D:\Run_Tao\DialectVoice_Research'),
        ('报告日期', '2026-10-21'),
        ('主方案', 'log-mel(40)+Δ 帧级特征 + 1D-CNN（3 种子集成，温度标定，纯 numpy 推理）'),
        ('对照方案', 'v2 基线 MFCC13+CMVN+DTW（E1）、经典统计嵌入 + 多项逻辑回归（E2）'),
        ('数据版本', 'v3（2796 条 / 192 组）、v3.1（2904 条 / 210 组）'),
        ('评估纪律', '固定表达组划分；超参/阈值只在验证集与未知校准集选择；test 仅最终评估'),
        ('文档性质', '内部技术报告（合成语音实验，非真人场景性能证据）'),
    ]
    for k, v in meta:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(3.2)
        p.paragraph_format.line_spacing = 1.4
        r = p.add_run(f'{k}：')
        set_run_font(r, FONT_CN, 10.5, bold=True)
        r = p.add_run(v)
        set_run_font(r, FONT_CN, 10.5)
    doc.add_page_break()

    # ---------- 目 录（静态，由报告标题生成） ----------
    toc_items = []
    for ln in lines:
        m = re.match(r'^(#{1,3})\s+(.*)$', ln.strip())
        if m:
            toc_items.append((len(m.group(1)), m.group(2).strip()))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('目  录')
    set_run_font(r, FONT_UI, 18, bold=True, color=COLOR_HEAD)
    doc.add_paragraph()
    for lvl, title in toc_items:
        if lvl == 1:
            continue  # 文档大标题已在封面
        if lvl > 2:
            continue
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0 if lvl == 2 else 0.8)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(1)
        add_inline(p, title, base_size=10.5 if lvl == 2 else 10)
    doc.add_page_break()

    # ---------- 正文 ----------
    i = 0
    n = len(lines)
    in_code = False
    code_buf = []
    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        # 围栏代码块
        if stripped.startswith('```'):
            if not in_code:
                in_code, code_buf = True, []
            else:
                in_code = False
                for cl in code_buf or ['']:
                    p = doc.add_paragraph(style='CodeBlock' if 'CodeBlock' in [s.name for s in doc.styles] else None)
                    p.paragraph_format.line_spacing = 1.1
                    p.paragraph_format.keep_together = True
                    p.paragraph_format.keep_with_next = True
                    pPr = p._p.get_or_add_pPr()
                    shd = OxmlElement('w:shd')
                    shd.set(qn('w:val'), 'clear')
                    shd.set(qn('w:fill'), 'F2F2F2')
                    pPr.append(shd)
                    r = p.add_run(cl)
                    set_run_font(r, FONT_MONO, 8.5)
            i += 1
            continue
        if in_code:
            code_buf.append(raw)
            i += 1
            continue

        if not stripped:
            i += 1
            continue
        if stripped in ('---', '***', '___'):
            i += 1
            continue

        # 标题
        m = re.match(r'^(#{1,4})\s+(.*)$', stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                p = doc.add_paragraph()
                r = p.add_run(text)
                set_run_font(r, FONT_UI, 16, bold=True, color=COLOR_HEAD)
                p.paragraph_format.space_after = Pt(6)
            else:
                doc.add_heading(text, level=min(level - 1, 3))
            i += 1
            continue

        # 表格
        if stripped.startswith('|'):
            block = []
            while i < n and lines[i].strip().startswith('|'):
                block.append(lines[i])
                i += 1
            rows = parse_table(block)
            caption = None
            # 表题在表后（Markdown 习惯）：向前窥探
            j = i
            while j < n and not lines[j].strip():
                j += 1
            if j < n and re.match(r'^\*\*(附?表)\s*[\dA-Za-z一二三四五六七八九十]+\s*[：:]', lines[j].strip()):
                caption = lines[j].strip()
                i = j + 1
            add_table(doc, rows, caption)
            continue

        # 图片
        m = re.match(r'^!\[(.*?)\]\((.*?)\)\s*$', stripped)
        if m:
            cap, rel = m.group(1), m.group(2)
            add_image(doc, (md_path.parent / rel).resolve() if not Path(rel).is_absolute() else Path(rel), cap)
            i += 1
            continue

        # 引用块
        if stripped.startswith('>'):
            buf = []
            while i < n and lines[i].strip().startswith('>'):
                buf.append(re.sub(r'^>\s?', '', lines[i].strip()))
                i += 1
            for qi, qline in enumerate(buf):
                text = qline.strip()
                if not text:
                    continue
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.8)
                p.paragraph_format.right_indent = Cm(0.4)
                p.paragraph_format.space_before = Pt(0 if qi else 4)
                p.paragraph_format.space_after = Pt(0 if qi < len(buf) - 1 else 6)
                p.paragraph_format.line_spacing = 1.35
                pPr = p._p.get_or_add_pPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:fill'), 'F7F7F7')
                pPr.append(shd)
                add_inline(p, text, base_size=9.5, color=COLOR_GRAY)
            continue

        # 无序列表
        if re.match(r'^[-*]\s+', stripped):
            while i < n and re.match(r'^[-*]\s+', lines[i].strip()):
                text = re.sub(r'^[-*]\s+', '', lines[i].strip())
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.95)
                p.paragraph_format.first_line_indent = Cm(-0.45)
                p.paragraph_format.line_spacing = 1.45
                p.paragraph_format.space_after = Pt(1)
                r = p.add_run('• ')
                set_run_font(r, FONT_CN, 10.5)
                add_inline(p, text)
                i += 1
            continue

        # 有序列表（保留手写序号，避免跨节连续编号）
        if re.match(r'^\d+[.)]\s+', stripped):
            while i < n and re.match(r'^\d+[.)]\s+', lines[i].strip()):
                text = lines[i].strip()
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.95)
                p.paragraph_format.first_line_indent = Cm(-0.45)
                p.paragraph_format.line_spacing = 1.45
                p.paragraph_format.space_after = Pt(1)
                add_inline(p, text)
                i += 1
            continue

        # 普通段落（合并软换行）
        buf = []
        while i < n:
            s = lines[i].strip()
            if not s or s.startswith(('#', '|', '>', '```')) or re.match(r'^[-*]\s+', s) \
                    or re.match(r'^\d+[.)]\s+', s) or re.match(r'^!\[', s):
                break
            buf.append(s)
            i += 1
        text = join_lines(buf)
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(21)
        add_inline(p, text)

    # 文档属性
    cp = doc.core_properties
    cp.title = '方言智控盒 第三阶段执行报告'
    cp.author = '方言智控盒项目组'
    cp.subject = '条件多样性扩充、判别式模型与端到端原型'
    cp.comments = '由 docs/v3_report.md 自动生成；数字来源见报告“数字溯源约定”。'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=str(BASE / 'docs' / 'v3_report.md'))
    ap.add_argument('--out', default=str(BASE / 'docs' / '方言智控盒_第三阶段报告.docx'))
    args = ap.parse_args()
    out = build(Path(args.md), Path(args.out))
    size = out.stat().st_size
    print(f'OK {out}  {size/1024:.1f} KB')


if __name__ == '__main__':
    main()


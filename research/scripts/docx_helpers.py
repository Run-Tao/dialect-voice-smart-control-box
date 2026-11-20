# -*- coding: utf-8 -*-
"""Word 文档排版助手（供开题报告等构建脚本复用）。"""
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

BODY_FONT = '宋体'
HEAD_FONT = '黑体'
EN_FONT = 'Times New Roman'


def set_font(run, size=12, bold=False, cjk=BODY_FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = EN_FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), cjk)
    rfonts.set(qn('w:ascii'), EN_FONT)
    rfonts.set(qn('w:hAnsi'), EN_FONT)


def para(doc, text='', size=12, bold=False, align=None, first_indent=True,
         space_before=0, space_after=6, line=1.5, cjk=BODY_FONT):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if first_indent and text:
        pf.first_line_indent = Pt(size * 2)
    if align == 'center':
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == 'right':
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if text:
        r = p.add_run(text)
        set_font(r, size=size, bold=bold, cjk=cjk)
    return p


def heading(doc, text, level=1):
    sizes = {0: 20, 1: 16, 2: 14, 3: 12}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level <= 1 else 10)
    pf.space_after = Pt(8 if level <= 1 else 6)
    pf.line_spacing = 1.3
    pf.keep_with_next = True
    if level == 0:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_font(r, size=sizes.get(level, 12), bold=True, cjk=HEAD_FONT)
    ppr = p._element.get_or_add_pPr()
    outline = OxmlElement('w:outlineLvl')
    outline.set(qn('w:val'), str(max(level - 1, 0)))
    ppr.append(outline)
    return p


def bullet(doc, text, size=12, level=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(24 + level * 18)
    pf.first_line_indent = Pt(-12)
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.45
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run('• ' if level == 0 else '– ')
    set_font(r, size=size)
    r2 = p.add_run(text)
    set_font(r2, size=size)
    return p


def _set_tbl_layout_fixed(t):
    """强制固定列宽布局，否则 Word/LibreOffice 会按内容自动收缩导致列宽失效。"""
    tblPr = t._tbl.tblPr
    old = tblPr.find(qn('w:tblLayout'))
    if old is not None:
        tblPr.remove(old)
    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    tblPr.append(layout)
    t.autofit = False


def _set_cell_width(cell, cm):
    cell.width = Cm(cm)
    tcPr = cell._tc.get_or_add_tcPr()
    old = tcPr.find(qn('w:tcW'))
    if old is not None:
        tcPr.remove(old)
    w = OxmlElement('w:tcW')
    w.set(qn('w:type'), 'dxa')
    w.set(qn('w:w'), str(int(Cm(cm).twips)))
    tcPr.append(w)


def _repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement('w:tblHeader')
    el.set(qn('w:val'), 'true')
    trPr.append(el)


def _no_row_split(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement('w:cantSplit')
    el.set(qn('w:val'), 'true')
    trPr.append(el)


def table(doc, headers, rows, widths=None, size=9, align_center_cols=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_tbl_layout_fixed(t)
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ''
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(str(h))
        set_font(r, size=size, bold=True, cjk=HEAD_FONT)
        tcpr = hdr[i]._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:fill'), 'D9E2F3')
        tcpr.append(shd)
    _repeat_header(t.rows[0])
    _no_row_split(t.rows[0])
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            p = cells[i].paragraphs[0]
            center = align_center_cols is None or i in align_center_cols
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if center
                           else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.15
            r = p.add_run(str(v))
            set_font(r, size=size)
        _no_row_split(t.rows[-1])
    if widths:
        # 版心宽度 = 21.0 - 3.0(左) - 2.6(右) = 15.4cm；超宽按比例缩放
        total = float(sum(widths))
        scale = min(1.0, 15.3 / total) if total > 0 else 1.0
        widths = [w * scale for w in widths]
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
            for row in t.rows:
                _set_cell_width(row.cells[i], w)
    return t


def caption(doc, text, size=10):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    set_font(r, size=size, cjk=HEAD_FONT)
    return p


def note(doc, text, size=10):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(12)
    pf.space_before = Pt(0)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.3
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run('说明：')
    set_font(r, size=size, bold=True, cjk=HEAD_FONT)
    r2 = p.add_run(text)
    set_font(r2, size=size)
    return p


def add_page_numbers(doc):
    """正文节页脚插入“第 N 页”；封面节不显示。"""
    for idx, section in enumerate(doc.sections):
        footer = section.footer
        footer.is_linked_to_previous = False
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.text = ''
        if idx == 0:
            continue
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p.add_run('第 ')
        set_font(r1, size=9)
        run = p.add_run()
        b = OxmlElement('w:fldChar')
        b.set(qn('w:fldCharType'), 'begin')
        instr = OxmlElement('w:instrText')
        instr.set(qn('xml:space'), 'preserve')
        instr.text = ' PAGE '
        e = OxmlElement('w:fldChar')
        e.set(qn('w:fldCharType'), 'end')
        run._r.append(b)
        run._r.append(instr)
        run._r.append(e)
        set_font(run, size=9)
        r2 = p.add_run(' 页')
        set_font(r2, size=9)


def setup_page(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.6)
    style = doc.styles['Normal']
    style.font.name = EN_FONT
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), BODY_FONT)
    return sec

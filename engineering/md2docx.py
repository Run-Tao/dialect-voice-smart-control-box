# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from pathlib import Path

base = Path(r'D:\Run_Tao\DialectVoice_Engineering')
md = base / 'docs' / '项目建议书_正式版.md'
out = base / 'docs' / '项目建议书_正式版.docx'
doc = Document()
st = doc.styles['Normal']
st.font.name = '宋体'
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

def set_font(run, name='宋体', size=11, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)


def add_rich(p, text, name='宋体', size=11):
    """把 **加粗** 与 `代码` 渲染成对应的 run，而不是留下裸标记。"""
    import re
    for part in re.split(r'(\*\*.+?\*\*|`[^`]+`)', text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            set_font(p.add_run(part[2:-2]), name, size, True)
        elif part.startswith('`') and part.endswith('`'):
            set_font(p.add_run(part[1:-1]), 'Consolas', size)
        else:
            set_font(p.add_run(part), name, size)


def split_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def is_sep(line):
    body = line.strip().strip('|')
    return bool(body) and all(set(c.strip()) <= set('-: ') and c.strip()
                              for c in body.split('|'))


def add_table(rows):
    """rows[0] 是表头；分隔行已在收集阶段剔除。"""
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = 'Table Grid'
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ''
            p = cell.paragraphs[0]
            add_rich(p, row[j] if j < len(row) else '', '宋体', 10)
            if i == 0:
                for r in p.runs:
                    set_font(r, '黑体', 10, True)
    doc.add_paragraph()


lines = md.read_text(encoding='utf-8').splitlines()
i = 0
while i < len(lines):
    raw = lines[i].rstrip()
    line = raw.strip()
    if not line:
        i += 1
        continue

    # 表格：连续的 | 行，跳过 |---| 分隔行
    if line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            cur = lines[i].strip()
            if not is_sep(cur):
                rows.append(split_row(cur))
            i += 1
        if rows:
            add_table(rows)
        continue

    if line.startswith('> '):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(18)
        add_rich(p, line[2:], '楷体', 11)
    elif line.startswith('## '):
        p = doc.add_heading(line[3:], level=2)
        for r in p.runs:
            set_font(r, '黑体', 14, True)
    elif line.startswith('# '):
        p = doc.add_heading(line[2:], level=1)
        for r in p.runs:
            set_font(r, '黑体', 16, True)
    elif line.startswith('- '):
        p = doc.add_paragraph(style='List Bullet')
        add_rich(p, line[2:])
    elif len(line) > 2 and line[0].isdigit() and line[1:3] in ('. ', '、'):
        p = doc.add_paragraph(style='List Number')
        add_rich(p, line[2:].strip())
    else:
        p = doc.add_paragraph()
        if line == '项目建议书':
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(p.add_run(line), '黑体', 24, True)
        elif line.startswith('基于方言'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(p.add_run(line), '黑体', 17, True)
        else:
            add_rich(p, line)
    i += 1

doc.save(out)
print('正式版项目建议书已生成:', out)

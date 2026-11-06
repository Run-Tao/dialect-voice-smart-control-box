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

for raw in md.read_text(encoding='utf-8').splitlines():
    line = raw.rstrip()
    if not line:
        continue
    if line.startswith('## '):
        p = doc.add_heading(line[3:], level=2)
        for r in p.runs:
            set_font(r, '黑体', 14, True)
    elif line.startswith('# '):
        p = doc.add_heading(line[2:], level=1)
        for r in p.runs:
            set_font(r, '黑体', 16, True)
    elif line.startswith('- '):
        p = doc.add_paragraph(style='List Bullet')
        r = p.add_run(line[2:])
        set_font(r, '宋体', 11)
    else:
        p = doc.add_paragraph()
        r = p.add_run(line)
        set_font(r, '宋体', 11)
        if line == '项目建议书':
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(r, '黑体', 24, True)
        elif line.startswith('基于方言'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(r, '黑体', 17, True)

doc.save(out)
print('正式版项目建议书已生成:', out)

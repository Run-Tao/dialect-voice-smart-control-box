# -*- coding: utf-8 -*-
from pptx import Presentation
from pptx.util import Inches
from pathlib import Path

base = Path(r'D:\Run_Tao\DialectVoice_Engineering')
pages = sorted((base / 'docs' / 'beamer_pages').glob('page-*.png'))
out = base / 'docs' / '开题报告_智能音箱外接版_Beamer.pptx'

prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

for img in pages:
    slide = prs.slides.add_slide(blank)
    slide.shapes.add_picture(str(img), 0, 0, width=prs.slide_width, height=prs.slide_height)

prs.save(out)
print('已生成:', out, '页数:', len(pages))

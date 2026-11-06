# -*- coding: utf-8 -*-
"""生成可编辑版答辩 PPT（与 docs/开题报告_智能音箱外接版_Beamer.tex 内容一致）。"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pathlib import Path

base = Path(r'D:\Run_Tao\DialectVoice_Engineering')
assets = base / 'docs' / 'beamer_assets'
out = base / 'docs' / '开题报告_智能音箱外接版.pptx'

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def font(run, size=18, bold=False):
    run.font.name = '微软雅黑'
    run.font.size = Pt(size)
    run.font.bold = bold


def title_slide(title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = title
    s.placeholders[1].text = subtitle
    for sh in [s.shapes.title, s.placeholders[1]]:
        for p in sh.text_frame.paragraphs:
            for r in p.runs:
                font(r, 30 if sh == s.shapes.title else 18, sh == s.shapes.title)


def bullet_slide(title, items):
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = title
    tf = s.placeholders[1].text_frame
    tf.clear()
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        for r in p.runs:
            font(r, 18)
    for p in s.shapes.title.text_frame.paragraphs:
        for r in p.runs:
            font(r, 30, True)
    return s


def image_slide(title, images):
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = title
    for p in s.shapes.title.text_frame.paragraphs:
        for r in p.runs:
            font(r, 30, True)
    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(4.0)
    for img in images:
        s.shapes.add_picture(str(assets / img), left, top, width=width)
        left += Inches(4.3)


title_slide('基于方言语音识别的智能家居语音转换装置', '本科工程导论开题报告')
bullet_slide('目录', [
    '研究背景与问题', '前期研究基础', '项目目标', '系统方案与技术路线',
    '数据采集与实验设计', '硬件与进度', '风险与预期成果'
])
bullet_slide('研究背景', [
    '很多老年人习惯使用方言，而主流智能家居设备主要面向普通话',
    '方言口音会导致语音识别率下降，影响老人使用智能家居',
    '现有智能音箱多为封闭系统，普通开发者难以修改固件或接入厂商接口'
])
bullet_slide('问题提出', [
    '已有研究多关注普通话语音识别，方言识别资源较少',
    '前期研究已经完成方言指令识别模型、置信度拒识与串口控制链路',
    '本项目要把离线研究转化为一个可外接、可实时运行、可现场展示的装置',
    '工程化过程中发现并修正了一个关键问题：训练输入与线上输入分布不一致'
])
bullet_slide('前期研究基础', [
    '粤语家居指令语料库与合成语料生成流水线（2904 条）',
    'log-mel 与 MFCC 特征提取，整句 CMVN 归一化',
    '多条识别路线：DTW 模板匹配、回归映射、Softmax 回归、线性 SVM、1D-CNN',
    '温度标定置信度与阈值拒识',
    '串口协议、模拟设备与 Web 演示台'
])
bullet_slide('特征提取', [
    'Mel 刻度：Mel(f)=2595log10(1+f/700)',
    '加窗：w(n)=0.54-0.46cos(2πn/(N-1))',
    '短时傅里叶变换：X(m,ω)=Σx(n)w(n-m)e^-jωn',
    '整句 CMVN 消除信道与音量差异'
])
bullet_slide('一条被否决的技术路线', [
    '前期曾以「方言特征 → 普通话特征」的回归映射为主线：ŷ=β0+β1x',
    '特征级指标很好：RMSE 0.025，余弦相似度 0.999',
    '但映射后命令识别准确率 0.1667，与 6 类随机基线相同',
    '原因：统一回归矩阵未按命令类别条件化，类别判别边界被抹平',
    '结论：转换必须发生在语义层，不能发生在特征层'
])
image_slide('前期研究结果（已否决路线的实验产物）',
            ['Map.png', 'RMatrix.png', 'Weight_Matrix_Heatmap.png'])
bullet_slide('本项目的核心工程发现', [
    '模型原本在带尾部静音的整段音频上训练，线上 VAD 输出却紧贴语音',
    '整句 CMVN 下，尾部静音稀释了均值与方差，输入分布一变模型失效',
    '短命令受损最重：LIGHT_ON 准确率由 0.9893 掉到 0.1716',
    '处置：用部署同款 VAD 重生成训练语料，再重新训练',
    '原则：训练输入与部署输入必须同源'
])
bullet_slide('项目目标', [
    '做一个智能家居语音转换装置',
    '用户说方言，装置在端侧完成识别与置信度判断',
    '通过扬声器播放标准普通话指令，驱动智能音箱执行',
    '支持 6 条固定家居命令',
    '通过 OLED 显示识别结果与置信度',
    '保留 ESP32-S3 串口与继电器直控作为演示备用'
])
bullet_slide('系统总体方案', [
    '输入：麦克风采集方言语音',
    '计算：端侧完成 VAD、特征提取、分类与拒识',
    '输出：扬声器播放标准普通话指令，OLED 显示结果',
    '执行：智能音箱接收普通话指令并控制家居设备',
    '备用：串口 ESP32-S3 或继电器直接控制演示设备'
])
bullet_slide('技术路线', [
    '方言语音 → VAD → log-mel40+Δ → CNN 集成 → 置信度 → 拒识 → 普通话指令 → 智能音箱执行',
    '特征：40 维 log-mel 及一阶差分，整句 CMVN，插值到 256 帧',
    '模型：1D-CNN 三种子集成，纯 numpy 推理，约 1.35 MB',
    '拒识：温度标定最大概率，阈值只在验证集与未知校准集上选取',
    '语义层映射：命令标签 → 普通话指令 → 预录音频播放'
])
bullet_slide('VAD 端点检测', [
    '帧能量转 dB，自适应噪声底（最小值统计）',
    '判决门限：噪声底 + 门限余量',
    '静音挂起判定语音结束，保留前置上下文避免吃掉起始音',
    '输出前后裁剪过的语音段，作为模型输入',
    '与训练语料生成使用同一实现，保证训练与部署同源'
])
bullet_slide('智能音箱外接联动', [
    '不修改智能音箱固件，不依赖厂商 API',
    '装置播放预录的普通话命令音频',
    '智能音箱识别后执行家居控制',
    '若现场联动不稳定，改用 ESP32-S3 串口或继电器直接控制演示设备'
])
bullet_slide('命令词与数据采集', [
    '命令词：开灯、关灯、开窗帘、关窗帘、全部关闭、查询状态',
    '合成语料：2904 条，3 个粤语音色，含语速/音高条件与纠正否定句式',
    '真人语料：2-3 位方言使用者，每人每命令 30-50 次',
    '未知指令负样本：用于标定拒识阈值',
    '环境：安静环境加电视/风扇背景噪声',
    '划分：按表达组划分固定训练/验证/测试集，避免泄漏'
])
bullet_slide('实验与评价指标', [
    '对比方法：MFCC+DTW 基线 与 log-mel+CNN',
    '消融：VAD 前 / 后，验证端点检测对准确率与误接受率的影响',
    '测试条件：安静/噪声、近/远距离、已注册/未注册用户',
    '指标：识别准确率、接受样本准确率、未知指令误接受率、端到端延迟、执行成功率、误触发率、总成本'
])
bullet_slide('硬件与预算', [
    '树莓派 4B/Zero 2W：0-400 元',
    'USB 麦克风：50-150 元',
    '扬声器与功放：30-80 元',
    'ESP32-S3 开发板：60-90 元',
    'SG90 舵机 / LED / 按键：20-40 元',
    'OLED：15-25 元',
    '电源、线材、外壳：130-250 元',
    '合计约 305-1085 元，按 1000 元以内控制；价格为估算，未实时核价'
])
bullet_slide('进度安排', [
    '第 1 周：需求确认、技术路线修订、硬件清单与采购',
    '第 2 周：VAD、实时管线、普通话播报、双备用执行层打通',
    '第 3 周：真人数据采集、训练语料同源重生成与重训练',
    '第 4 周：拒识阈值标定与安全评估',
    '第 5 周：多人、噪声、距离测试与端到端延迟实测',
    '第 6 周：硬件联调与现场演示彩排',
    '第 7 周：成本核算、演示视频、答辩 PPT',
    '第 8 周：答辩与材料归档'
])
bullet_slide('风险与应对', [
    '合成与真人存在域差距：补真人语料，报告区分合成与实测数字',
    '未知指令误接受（最突出短板）：补采负样本、阈值仅在校准集选取、演示展示低置信度拒识',
    '智能音箱识别不稳定：预录普通话音频，准备串口与继电器两条备用通道',
    '真人语料不足：固定 6 条命令，2-3 位使用者反复采集，必要时用个性化注册模板补充',
    '端侧实时性不足：优先树莓派 4B，模型已纯 numpy 推理且体积 1.35 MB',
    '现场演示失败：准备预录视频、备用音频与继电器演示'
])
bullet_slide('预期成果', [
    '方言语音智能家居控制原型',
    '方言与普通话指令数据及标注（合成与真人分开标注来源）',
    '端侧推理程序与自包含运行时（纯 numpy，可离线运行）',
    '智能音箱联动演示与双备用通道演示',
    '测试报告、成本表、3 分钟演示视频与答辩 PPT',
    '技术路线修订说明：记录被否决方案及其实验依据'
])
bullet_slide('参考文献', [
    '申请人前期研究：方言家居指令识别与控制，实验记录与代码仓库',
    'Davis S, Mermelstein P. Comparison of parametric representations for '
    'monosyllabic word recognition in continuously spoken sentences. '
    'IEEE Transactions on Acoustics, Speech, and Signal Processing, 1980.',
    'MFCC、log-mel 特征、DTW、CNN 语音命令识别与智能家居语音交互相关文献'
])

prs.save(out)
print('可编辑PPT已生成:', out, '页数:', len(prs.slides))

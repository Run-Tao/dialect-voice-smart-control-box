# -*- coding: utf-8 -*-
"""生成工程型开题报告（两个月、交付导向）。

定位：两个月内完成一个可运行、可演示、可操作的粤语语音控制装置。
写作重点 = 工程效果与完善度（功能完整、时延可用、能实机闭环、一键运行、
文档齐全）；不追求算法创新、不做繁重统计检验、不做多方案消融。

现状数据取自本项目真实结果文件；验收阈值明确标注为待达到。
"""
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from docx_helpers import (  # noqa: E402
    HEAD_FONT, add_page_numbers, bullet, caption, heading, note, para,
    set_font, setup_page, table,
)

OUT = Path(r'D:\Run_Tao\DialectVoice_Research\docs\开题报告_粤语语音控制.docx')


def cover(doc):
    for _ in range(3):
        para(doc, '', first_indent=False, space_after=0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('本科毕业设计（论文）开题报告')
    set_font(r, size=15, bold=True, cjk=HEAD_FONT)

    para(doc, '', first_indent=False)

    for line, size in (('方言智控盒：面向智能家居的', 19),
                       ('低成本粤语语音控制装置设计与实现', 19)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(line)
        set_font(r, size=size, bold=True, cjk=HEAD_FONT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run('（工程设计类·两个月交付）')
    set_font(r, size=13, bold=True, cjk=HEAD_FONT)

    for _ in range(3):
        para(doc, '', first_indent=False, space_after=0)

    caption(doc, '项目建议书', size=14)

    for _ in range(3):
        para(doc, '', first_indent=False, space_after=0)

    info = [
        ('项目名称', '方言智控盒——低成本粤语语音控制装置'),
        ('项目类型', '工程设计 / 原型开发（交付导向）'),
        ('计划周期', '2 个月（8 周）'),
        ('交付形态', '可运行软件 + 串口执行器控制 + 演示与说明文档'),
        ('现有基础', '600 条粤语命令语料、识别与控制软件链路、65 项自动测试已就绪'),
        ('报告日期', '2026 年 10 月'),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for k, v in info:
        cells = t.add_row().cells
        p0 = cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r0 = p0.add_run(k + '：')
        set_font(r0, size=11, bold=True, cjk=HEAD_FONT)
        p1 = cells[1].paragraphs[0]
        r1 = p1.add_run(v)
        set_font(r1, size=11)
        cells[0].width = Cm(2.9)
        cells[1].width = Cm(11.5)

    doc.add_section(WD_SECTION.NEW_PAGE)


def chapter1(doc):
    heading(doc, '一、项目概述', level=1)

    heading(doc, '1.1 要解决的问题', level=2)
    para(doc, '家里说粤语的长辈用普通话语音助手很吃力，而市面上的语音方案大多依赖联网。'
              '本项目做一个离线的小装置：说一句粤语，它就执行灯和窗帘的动作，'
              '并能听懂查询状态。只用六类固定命令，不追求通用语音助手。')
    para(doc, '之所以选六类封闭命令，是因为这样可以在没有GPU、没有网络、'
              '成本很低的条件下把整条链路做通——这也正是本项目的价值所在：'
              '把「能听懂 → 能判断 → 能动作 → 有反馈」这条闭环做完整、做稳定。')

    heading(doc, '1.2 项目定位：交付完整装置，而非算法研究', level=2)
    para(doc, '本项目是工程设计类项目，评价标准是「东西能不能用、好不好用、'
              '能不能复现」，而不是算法是否新颖。具体而言：')
    bullet(doc, '算法上采用现成且够用的方案（MFCC + DTW 模板匹配），'
                '不做模型创新、不做复杂消融实验；')
    bullet(doc, '把时间花在工程完善上：交互反馈、异常处理、配置化、一键运行、'
                '实机联调、演示材料；')
    bullet(doc, '精度上以「不倒退、够演示」为准则，不为了追求指标而反复调参；')
    bullet(doc, '所有指标如实记录，不夸大成“已达商用水平”。')

    heading(doc, '1.3 两个月的工作边界', level=2)
    table(doc,
          ['范围', '内容'],
          [['做', '六类粤语命令识别；拒识（非命令不动作）；置信度提示；'
                  '串口驱动执行器；PC 端操作与演示程序；实机联调；'
                  '使用说明与演示材料'],
           ['不做', '连续语音听写、开放词表、多方言通用识别、云端大模型接入、'
                   '算法创新与论文级对比实验'],
           ['时间紧张时的取舍', '优先保证「软件闭环 + 串口动作 + 演示」完整；'
                                '实机硬件若延期，先用模拟设备交付，实机作为增量']],
          widths=[3.0, 12.8], size=9, align_center_cols={0})
    caption(doc, '表 1  项目边界')

    heading(doc, '1.4 已有基础（这是两个月能做完的前提）', level=2)
    para(doc, '本项目并非从零开始，前期的数据与软件链路已具备，'
              '两个月主要用于「完善工程」而不是「从无到有」：')
    table(doc,
          ['已具备', '现状'],
          [['粤语命令语料', '600 条，六类各 100 条，0 条质量排除、0 条重复'],
           ['表达库', '174 条粤语表达（六类各 29 条），覆盖五种语域'],
           ['音色', '4 个独立音色（edge-tts 3 个 + canto-tts 1 个）'],
           ['数据划分', '训练 397 / 验证 92 / 固定测试集 111，泄漏检查全为 0'],
           ['识别软件', 'MFCC、DTW、模板匹配、拒识、串口协议、模拟设备均已实现'],
           ['自动化测试', '65 项单元测试全部通过'],
           ['运行环境', '纯 Python + numpy，普通 CPU 可离线运行']],
          widths=[3.4, 12.4], size=9, align_center_cols={0})
    caption(doc, '表 2  前期已完成的基础')


def chapter2(doc):
    heading(doc, '二、需求分析', level=1)

    heading(doc, '2.1 使用场景', level=2)
    para(doc, '家庭客厅或卧室，环境噪声中等，说话距离 1–3 米。'
              '用户说一句简短粤语命令（2–9 个音节），装置在 1–2 秒内完成动作并给出反馈。'
              '装置离线放置，仅控制灯与窗帘两类执行器。')
    para(doc, '家庭里会有大量非命令语音（聊天、电视）。对这类语音，'
              '宁可“不动作”，也不要“误动作”——误开灯、误关窗帘会直接影响使用意愿。')

    heading(doc, '2.2 功能需求', level=2)
    table(doc,
          ['编号', '功能', '验收方式', '优先级'],
          [['FR1', '开灯 / 关灯', '说命令后继电器动作，指示灯变化', '必须'],
           ['FR2', '开窗帘 / 关窗帘', '说命令后舵机转动到对应位置', '必须'],
           ['FR3', '全部关闭', '一次动作同时关闭灯与窗帘', '必须'],
           ['FR4', '查询状态', '返回当前灯与窗帘状态并语音/文字提示', '必须'],
           ['FR5', '非命令拒识', '聊天语音不触发动作', '必须'],
           ['FR6', '低置信度二次确认', '置信度不足时提示“没听清，请再说一次”，不动作', '必须'],
           ['FR7', '串口协议', 'PC 与执行器报文正确、重复序号可去重', '必须'],
           ['FR8', '三种运行方式', '单文件识别 / 批量演示 / 实时麦克风', '应当'],
           ['FR9', '操作日志', '每次识别记录命令、置信度、是否拒识、动作结果', '应当']],
          widths=[1.6, 4.6, 7.4, 2.2], size=9, align_center_cols={0, 3})
    caption(doc, '表 3  功能需求')

    heading(doc, '2.3 工程指标（务实口径）', level=2)
    para(doc, '不追求漂亮的算法指标，只要求「能稳定演示、响应不拖沓」：')
    table(doc,
          ['编号', '指标', '要求', '说明'],
          [['P1', '六类识别准确率', '固定测试集不低于现有基线（0.5495）', '不倒退即可，不设激进目标'],
           ['P2', '单条识别时延', '≤0.5 秒（CPU）', '改用精简模板配置后实测可达'],
           ['P3', '端到端响应', '从说完到动作 ≤2 秒', '含串口通信与执行器行程'],
           ['P4', '连续稳定性', '连续 100 次识别不崩溃', '含异常输入'],
           ['P5', '离线可用', '断网下全部功能正常', '不依赖任何在线服务'],
           ['P6', '一键运行', '双击脚本即可启动演示', '降低演示与复现门槛'],
           ['P7', '成本', '整机物料控制在低成本区间', '具体金额待询价确认']],
          widths=[1.4, 4.0, 6.4, 4.0], size=9, align_center_cols={0, 1})
    caption(doc, '表 4  工程指标')

    heading(doc, '2.4 明确不做的事', level=2)
    para(doc, '（1）不做开放词表与连续语音识别；'
              '（2）不接云端大模型或付费 API；'
              '（3）不做多方言支持（其他方言仅用于拒识测试）；'
              '（4）不做算法创新与论文级对比实验；'
              '（5）不把合成语音的测试结果说成真实用户场景的准确率。')


def chapter3(doc):
    heading(doc, '三、现状与待改进项', level=1)

    heading(doc, '3.1 现有系统能跑到什么程度', level=2)
    para(doc, '软件链路已通：给定一条 16kHz WAV，可以输出六类命令之一及置信度；'
              '结果能通过网络/串口协议驱动模拟执行器，模拟控制器已实现完整的'
              '状态机与报文校验。也就是说，两个月的工作是“把它做完整、做快、做实”，'
              '不是“从头搭”。')

    heading(doc, '3.2 当前存在的问题（按对交付的影响排序）', level=2)
    table(doc,
          ['编号', '问题', '当前情况', '处理策略'],
          [['E1', '推理太慢', '全模板配置约 2.0 秒/条，不适合交互', '改用精简模板配置，见 3.3'],
           ['E2', '开/关类命令易混', '关窗帘常被判成开窗帘等；两类近似句子只差一个动词',
            '工程上不硬改算法，用「置信度不足时二次确认」兜底，并在演示中避开歧义句'],
           ['E3', '拒识阈值难调', '阈值高则拒掉太多真命令，阈值低则可能误动作',
            '采用保守策略 + 二次确认，优先保证不误动作'],
           ['E4', '没有实机', '目前只有模拟控制器', '第 5–6 周接入 ESP32-S3'],
           ['E5', '没有交互界面', '只有脚本命令', '增加简单的可视化/命令行交互与提示音'],
           ['E6', '部署门槛高', '需要按文档手动配环境', '打包一键启动脚本与依赖说明']],
          widths=[1.4, 3.4, 6.4, 4.6], size=9, align_center_cols={0})
    caption(doc, '表 5  当前问题与处理策略')

    heading(doc, '3.3 部署配置的工程选择（解决 E1）', level=2)
    para(doc, '前期已做过模板规模对照：每类模板数越多越慢。'
              '把「准确率」与「时延」放在一起看，选择精简配置作为部署默认值：')
    table(doc,
          ['配置', '模板总数', '测试集准确率', '单条时延', '取舍'],
          [['全模板（原默认）', 397, '0.5495', '约 2.0 秒', '精度略高但太慢，不适合交互'],
           ['精简每类 10 个（选用）', 60, '0.5766', '约 0.43 秒', '准确率没有下降、时延降约 4.7 倍'],
           ['每类 5 个', 30, '0.5315', '约 0.15 秒', '更快但准确率下降，作为低配备选']],
          widths=[4.4, 2.2, 2.8, 2.6, 3.8], size=9, align_center_cols={1, 2, 3})
    caption(doc, '表 6  部署配置对照（测试集固定 111 条）')
    note(doc, '准确率差异在少量样本上并不构成“提升”证据，但时延收益明确且对交互体验至关重要，'
              '因此以精简配置为部署默认，同时保留全模板配置供对照。'
              '这也体现本项目取向：优先保证工程可用，不追求纸面指标。')


def chapter4(doc):
    heading(doc, '四、系统设计', level=1)

    heading(doc, '4.1 总体结构', level=2)
    para(doc, '整体分三段：PC 端负责识别与决策，串口负责传输命令，'
              '单片机负责驱动执行器。识别算法与硬件解耦，换硬件不需要改算法。')
    table(doc,
          ['环节', '所在位置', '负责什么', '实现状态'],
          [['① 采集与识别', 'PC（Python）', '读取音频/麦克风、提取 MFCC、模板匹配、输出命令与置信度',
            '已实现'],
           ['② 决策与拒识', 'PC（Python）', '阈值判断、低置信度二次确认、命令到动作的映射',
            '已实现，待补二次确认'],
           ['③ 传输', 'USB 串口', 'CMD / ACK / STATE / ERR 文本报文', '协议已实现'],
           ['④ 执行', 'ESP32-S3', '解析命令、驱动继电器与舵机、回报状态', '模拟已实现，实机待接'],
           ['⑤ 反馈', 'PC 界面 + 蜂鸣器/LED', '提示识别结果、执行结果、失败原因', '待做']],
          widths=[2.6, 2.8, 7.4, 3.0], size=9, align_center_cols={0, 1, 3})
    caption(doc, '表 7  系统分段与实现状态')

    heading(doc, '4.2 串口协议（已实现，实机沿用）', level=2)
    table(doc,
          ['报文', '格式', '方向', '说明'],
          [['命令', 'CMD,<seq>,<INTENT>', 'PC → 单片机', 'INTENT 为六类命令之一'],
           ['应答', 'ACK,<seq>,OK|DUP', '单片机 → PC', 'DUP 表示重复序号'],
           ['状态', 'STATE,<seq>,LIGHT=on|off,CURTAIN=open|closed', '单片机 → PC', '查询命令返回'],
           ['错误', 'ERR,<seq>,<CODE>', '单片机 → PC', 'BAD_FORMAT / UNKNOWN_CMD 等']],
          widths=[2.0, 7.4, 2.8, 3.6], size=9, align_center_cols={0, 2})
    caption(doc, '表 8  串口协议定义')
    note(doc, '协议为纯文本行，便于用串口助手直接观察与排错；'
              '已通过重复序号去重、非法报文拒绝、未知命令拒绝等单元测试。')

    heading(doc, '4.3 三种运行方式', level=2)
    table(doc,
          ['方式', '用途', '输入', '输出'],
          [['单文件识别', '调试单条音频、定位错误', '一个 WAV 文件', '命令、置信度、耗时'],
           ['批量演示', '现场演示与验收', '演示音频序列或目录', '逐条结果与执行动作'],
           ['实时麦克风', '真实交互体验', '麦克风录音（按键触发）', '识别结果 + 动作 + 提示音']],
          widths=[3.2, 6.2, 3.2, 3.2], size=9, align_center_cols={0})
    caption(doc, '表 9  运行方式')

    heading(doc, '4.4 交互与反馈设计（工程完善度的重点）', level=2)
    para(doc, '这是两个月里最容易被忽略、但最影响“好不好用”的部分：')
    bullet(doc, '识别后立即给出可感知反馈：界面显示命令名 + 置信度，蜂鸣器一声表示已接受；')
    bullet(doc, '置信度不足时不动作，提示「没听清，请再说一次」，避免误动作；')
    bullet(doc, '执行失败（串口超时、无应答）时提示原因，并保持设备状态不变；')
    bullet(doc, '界面常驻显示当前灯与窗帘状态，与单片机回报保持一致；')
    bullet(doc, '所有识别与动作写入日志文件，便于演示后复盘。')

    heading(doc, '4.5 目录与配置', level=2)
    para(doc, '保持现有工程结构，新增部分集中在两个位置，避免打乱已有代码：'
              '识别与决策新增到 scripts/ 下的运行脚本，界面与演示新增到 app/ 目录；'
              '所有可调参数（模板路径、阈值、串口号、模板配置）集中到一个配置文件，'
              '换机器只需改配置。')


def chapter5(doc):
    heading(doc, '五、两个月实施计划', level=1)

    heading(doc, '5.1 周计划', level=2)
    table(doc,
          ['周次', '主要工作', '本周交付物'],
          [['第 1 周', '环境整理与固化；确认精简模板配置；跑通单文件识别与模拟串口',
            '可运行的单文件识别脚本 + 部署配置'],
           ['第 2 周', '识别服务封装（输入音频→输出命令/置信度）；配置文件化；操作日志',
            '识别服务模块 + 日志'],
           ['第 3 周', '拒识与低置信度二次确认；异常输入处理（空音频、噪声、超长）',
            '决策与拒识模块 + 异常用例记录'],
           ['第 4 周', 'PC 端交互：命令行交互 + 简易可视化界面；提示音与状态显示',
            '可交互的演示程序'],
           ['第 5 周', 'ESP32-S3 接线与固件：串口解析、继电器与舵机驱动、状态回报',
            '固件 v1 + 接线说明'],
           ['第 6 周', '端到端联调：真实说话触发真实动作；修复时延与稳定性问题',
            '实机可用原型 + 联调记录'],
           ['第 7 周', '一键启动脚本、依赖说明、演示脚本与演示视频录制',
            '可交付软件包 + 演示视频'],
           ['第 8 周', '缓冲周：补测试、修缺陷、整理文档、准备答辩演示',
            '使用说明书、测试记录、答辩材料']],
          widths=[1.8, 8.0, 6.0], size=9, align_center_cols={0})
    caption(doc, '表 10  八周实施计划')

    heading(doc, '5.2 进度控制方式', level=2)
    bullet(doc, '每周结束时必须有一个“可运行”的产物，不允许连续两周只有代码没有结果；')
    bullet(doc, '第 4 周末做第一次彩排：即使硬件未到，也要能用模拟设备完整演示一遍；')
    bullet(doc, '第 6 周末做第二次彩排：实机演示；若硬件延期，用模拟设备交付并书面说明；')
    bullet(doc, '第 8 周只做修复与整理，不追加新功能。')

    heading(doc, '5.3 硬件清单', level=2)
    table(doc,
          ['物料', '选型建议', '用途', '备注'],
          [['主控板', 'ESP32-S3 开发板', '接收命令、驱动执行器', '需支持 USB 串口'],
           ['麦克风', 'USB 麦克风或 I2S 数字麦克风', '采集语音', '先用 PC 自带麦克风也可'],
           ['继电器模块', '单路 5V 继电器', '模拟灯开关', '注意强电安全，演示用小灯泡'],
           ['舵机', 'SG90 或同类', '模拟窗帘开合', '行程需机械限位'],
           ['电源', '5V 电源模块', '整机供电', '舵机与主控分开供电更稳'],
           ['连线', '杜邦线、面包板', '原型连线', '—']],
          widths=[2.6, 5.6, 4.4, 3.2], size=9, align_center_cols={0})
    caption(doc, '表 11  硬件清单')
    note(doc, '未进行市场询价，因此不列具体金额，避免使用未经核实的报价；'
              '选型以“手边可得、能演示”为原则。')

    heading(doc, '5.4 软件环境', level=2)
    para(doc, '主环境 Python 3.13 + numpy；语料生成使用独立虚拟环境'
              '（canto-tts、edge-tts、soundfile）。'
              '全部依赖可离线安装，不涉及付费服务。'
              '交付时提供一份依赖清单与一键安装脚本。')


def chapter6(doc):
    heading(doc, '六、测试与演示', level=1)

    heading(doc, '6.1 功能测试清单', level=2)
    para(doc, '以“能不能用”为标准逐项过一遍，重点是异常路径：')
    table(doc,
          ['用例', '输入', '期望结果'],
          [['T1', '六类命令各若干条音频', '识别正确、动作正确、界面状态更新'],
           ['T2', '非命令语音（如「開冷氣」）', '不动作，提示未识别'],
           ['T3', '低置信度语音', '提示“没听清”，不动作'],
           ['T4', '空音频 / 极短音频 / 纯噪声', '不崩溃，提示无效输入'],
           ['T5', '串口未连接', '提示连接失败，不误报成功'],
           ['T6', '重复序号命令', '只执行一次'],
           ['T7', '非法报文', '返回错误码，设备状态不变'],
           ['T8', '连续 100 次识别', '不崩溃、无内存持续增长'],
           ['T9', '实机断电重启', '重新连接后可继续使用']],
          widths=[1.6, 5.2, 9.0], size=9, align_center_cols={0})
    caption(doc, '表 12  功能测试清单')

    heading(doc, '6.2 现场演示脚本', level=2)
    para(doc, '演示按固定顺序进行，保证稳定可复现：')
    bullet(doc, '第一步：显示当前状态（灯关、窗帘关）；')
    bullet(doc, '第二步：说「開燈」→ 灯亮，界面提示“开灯 已执行”；')
    bullet(doc, '第三步：说「打開窗簾」→ 舵机转动，窗帘打开；')
    bullet(doc, '第四步：说「查詢狀態」→ 播报/显示“灯亮、窗帘开”；')
    bullet(doc, '第五步：说「全部關閉」→ 两者同时复位；')
    bullet(doc, '第六步：说一句非命令（如「今日天氣點樣」）→ 提示未识别、不动作；')
    bullet(doc, '第七步：故意含糊发音 → 提示“没听清”，演示不误动作。')

    heading(doc, '6.3 记录与验收', level=2)
    table(doc,
          ['编号', '验收项', '标准', '记录方式'],
          [['A1', '六类命令可用', '每类至少 3 条不同说法能正确执行', '演示记录表'],
           ['A2', '非命令不动作', '连续 10 句非命令语音 0 次误动作', '测试记录'],
           ['A3', '单条时延', '≤0.5 秒', '程序打印耗时'],
           ['A4', '端到端响应', '说完到动作 ≤2 秒', '秒表/录屏'],
           ['A5', '连续稳定性', '100 次无崩溃', '压力测试日志'],
           ['A6', '实机闭环', '真实执行器动作正确', '录屏'],
           ['A7', '可复现', '照使用说明能在另一台机器跑起来', '他人试跑'],
           ['A8', '文档齐全', '使用说明、接线图、演示视频齐备', '交付清单核对']],
          widths=[1.4, 4.2, 6.4, 3.8], size=9, align_center_cols={0})
    caption(doc, '表 13  验收项与记录方式')
    note(doc, '准确率按固定测试集如实报告（当前 0.5495，精简配置 0.5766），'
              '不作为“达标”依据去修饰；验收以系统可用性与稳定性为主。')


def chapter7(doc):
    heading(doc, '七、交付物', level=1)
    table(doc,
          ['编号', '交付物', '形态', '完成判据'],
          [['D1', '识别与控制软件', '可运行程序 + 源码', '按使用说明可启动并完成演示'],
           ['D2', '一键启动脚本', '批处理/脚本文件', '双击即可运行演示'],
           ['D3', '实机原型', 'ESP32-S3 + 继电器 + 舵机', '真实语音触发真实动作'],
           ['D4', '使用说明书', '文档', '含环境安装、接线、操作、常见问题'],
           ['D5', '测试记录', '文档 + 日志', '覆盖表 13 全部验收项'],
           ['D6', '演示视频', '视频文件', '完整演示七步脚本'],
           ['D7', '设计说明书', '文档', '含需求、设计、实现、测试与局限'],
           ['D8', '数据与配置', '清单、模板、配置文件', '可随软件一起复制运行']],
          widths=[1.4, 4.0, 5.4, 5.0], size=9, align_center_cols={0})
    caption(doc, '表 14  交付物清单')


def chapter8(doc):
    heading(doc, '八、风险与应对', level=1)
    table(doc,
          ['编号', '风险', '影响', '应对措施'],
          [['R1', 'ESP32-S3 或器件到货延期', '实机无法按期演示',
            '先用模拟设备完成全部软件验收；第 4 周彩排不依赖硬件；到货后一周内补齐'],
           ['R2', '实机联调耗时超预期', '挤占文档与演示时间',
            '第 8 周设为纯缓冲周；若确实来不及，交付模拟版并附实机预研记录'],
           ['R3', '开/关类命令识别不稳', '演示时出现错误动作',
            '演示脚本避开歧义句；低置信度强制二次确认；界面显示候选供人工确认'],
           ['R4', '麦克风与环境噪声影响', '现场识别率下降',
            '演示用近场麦克风；先用录制好的音频演示，再用实时麦克风展示'],
           ['R5', '时间整体紧张', '交付不完整',
            '严格按周交付；优先保“软件闭环+串口动作+演示”，其余为增量'],
           ['R6', '精度偏低被质疑', '答辩压力',
            '如实报告固定测试集结果，同时强调这是合成语音条件下的原型、'
            '并说明真人数据是下一步工作']],
          widths=[1.4, 4.0, 4.2, 6.6], size=9, align_center_cols={0})
    caption(doc, '表 15  风险与应对')


def chapter9(doc):
    heading(doc, '九、参考技术资料', level=1)
    refs = [
        'Sakoe H, Chiba S. Dynamic programming algorithm optimization for spoken word '
        'recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing, '
        '1978, 26(1): 43–49.',
        'Davis S, Mermelstein P. Comparison of parametric representations for monosyllabic '
        'word recognition in continuously spoken sentences. IEEE Transactions on Acoustics, '
        'Speech, and Signal Processing, 1980, 28(4): 357–366.',
        'Yu T, Frieske R, Xu P, et al. Automatic Speech Recognition Datasets in Cantonese: '
        'A Survey and New Dataset. Proceedings of LREC 2022, Marseille, 2022: 6487–6494.',
        'Menon R, Kamper H, Quinn J, Niesler T. Fast ASR-free and almost zero-resource '
        'keyword spotting using DTW and CNNs for humanitarian monitoring. '
        'Proceedings of Interspeech 2018.',
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Pt(24)
        pf.first_line_indent = Pt(-24)
        pf.space_before = Pt(0)
        pf.space_after = Pt(5)
        pf.line_spacing = 1.35
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run('[%d] ' % i)
        set_font(r, size=10.5)
        r2 = p.add_run(ref)
        set_font(r2, size=10.5)

    note(doc, '文献均为与本项目直接相关的经典方法与粤语数据资源，已核实标题与出处。'
              '本项目为工程设计类，不做文献综述式论证；'
              '答辩前补充 ESP32-S3 数据手册与串口通信相关资料即可。')


def appendix(doc):
    heading(doc, '附录 A  现有资产与复现入口', level=1)
    table(doc,
          ['路径', '内容'],
          [['data/tts_corpus/manifest_yue_v2.csv', '600 行数据清单'],
           ['data/splits/track_a_expression_split.csv', '固定划分文件（含哈希）'],
           ['data/models/', '已训练的模板模型文件'],
           ['src/', '特征、距离、模板、拒识、串口协议等核心模块'],
           ['scripts/run_v2_experiments.py', '识别效果与时延的复现入口'],
           ['scripts/rejection_v2.py', '拒识评估'],
           ['outputs/v2_experiments/main_v2/results.json', '对照实验原始结果'],
           ['docs/v2_report.md', '前期完整执行报告'],
           ['tests/', '65 项自动化测试']],
          widths=[7.6, 8.2], size=9, align_center_cols=set())
    caption(doc, '表 16  现有资产')

    heading(doc, '附录 B  现状与两个月目标的对照', level=1)
    table(doc,
          ['维度', '现状', '两个月后目标'],
          [['识别准确率', '0.5495（全模板）/ 0.5766（精简配置）', '如实报告，不倒退'],
           ['单条时延', '约 2.0 秒（全模板）/ 约 0.43 秒（精简）', '≤0.5 秒'],
           ['交互界面', '无（仅脚本）', '命令行 + 简易界面 + 提示音'],
           ['拒识', '有阈值，无二次确认', '阈值 + 二次确认，优先不误动作'],
           ['执行器', '模拟控制器', 'ESP32-S3 实机驱动'],
           ['运行门槛', '需按文档手动配环境', '一键启动脚本'],
           ['文档', '技术报告已有', '使用说明 + 接线图 + 演示视频'],
           ['测试', '65 项单元测试', '增加端到端与稳定性测试']],
          widths=[3.0, 6.8, 6.0], size=9, align_center_cols={0})
    caption(doc, '表 17  现状与目标对照')

    para(doc, '', first_indent=False, space_after=0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('（现状数据均来自已运行并核对的结果文件；'
                  '目标为两个月内拟达到的工程状态。）')
    set_font(r, size=9.5, cjk=HEAD_FONT)


def build():
    doc = Document()
    setup_page(doc)
    cover(doc)
    chapter1(doc)
    chapter2(doc)
    chapter3(doc)
    chapter4(doc)
    chapter5(doc)
    chapter6(doc)
    chapter7(doc)
    chapter8(doc)
    chapter9(doc)
    appendix(doc)
    add_page_numbers(doc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print('saved', OUT)


if __name__ == '__main__':
    build()

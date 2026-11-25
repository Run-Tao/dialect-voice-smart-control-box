# -*- coding: utf-8 -*-
"""跨交付物一致性守卫：锁死 2026-11-16 的口径更正，防止评分文档静默回退。

只读文档与工件，不重算任何评估数字。背景：「覆盖率断崖到 0.0023」曾被四份评分
文档引用，实为网格取值假象；S5 判定、感知延迟工作点、硬件主控也曾与已下发配置
矛盾。这里把「更正后的不变量」写成测试，任何一条回退都会红。
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'

GRADED = [
    DOCS / '开题报告_智能音箱外接版_Beamer.tex',
    DOCS / '项目建议书_正式版.md',
    DOCS / '交付物清单.md',
    DOCS / 'v5方案设计.md',
    ROOT / 'build_integrated_pptx.py',
]
HISTORICAL = DOCS / '项目审查_2026-11-13.md'
ALL_DOCS = GRADED + [HISTORICAL]

# 出现「断崖」的行必须带否定/更正标记，否则就是把假结论当事实复述。
CLIFF_MARKERS = ('证伪', '假象', '不是', '不存在', '作废', '更正', '旧版', '~~', '不依赖')
# 出现优化前延迟数字的文档必须自带口径标注，否则读者会把它当当前生效值。
CALIBER_MARKERS = ('优化前', '120 ms', '时点')


def _text(path):
    return path.read_text(encoding='utf-8')


def _shipped_fast_silence_ms():
    m = re.search(r'fast_silence_ms:\s*(\d+)', _text(ROOT / 'config.yaml'))
    assert m, 'config.yaml 缺 fast_silence_ms'
    return int(m.group(1))


@pytest.mark.parametrize('path', ALL_DOCS, ids=lambda p: p.name)
def test_no_affirmative_false_cliff(path):
    for line in _text(path).splitlines():
        if '断崖' in line:
            assert any(m in line for m in CLIFF_MARKERS), (
                f'{path.name} 把已被证伪的「覆盖率断崖」当事实复述：{line.strip()}')


@pytest.mark.parametrize('path', ALL_DOCS, ids=lambda p: p.name)
def test_s5_failure_verdict_carries_time_label(path):
    for line in _text(path).splitlines():
        if '未达成' in line and ('S5' in line or '验收判据' in line):
            assert any(m in line for m in ('当时', '优化前')), (
                f'{path.name} 的 S5 未达成判定缺少时点标注：{line.strip()}')


@pytest.mark.parametrize('path', GRADED, ids=lambda p: p.name)
def test_s5_gain_matches_optimized_artifact(path):
    art = DOCS / 'dialog_eval_stream_optimized_v2.json'
    if not art.exists():
        pytest.skip('优化工件未生成')
    gain = (json.loads(_text(art)).get('confusable_pair') or {}).get('gain_vs_single_gate')
    if gain is None or gain <= 0:
        pytest.skip('工件增益非正，达成判定不适用')
    text = _text(path)
    if 'S5' in text:
        assert f'{gain:.4f}' in text, f'{path.name} 谈 S5 却未印工件增益 {gain:.4f}'


@pytest.mark.parametrize('path', ALL_DOCS, ids=lambda p: p.name)
def test_pre_optimization_latency_is_labeled(path):
    text = _text(path)
    if '142.7' not in text:
        pytest.skip('该文档不引用优化前延迟')
    assert any(m in text for m in CALIBER_MARKERS), (
        f'{path.name} 引用 142.7 ms 却无优化前/120 ms/时点口径标注')
    assert str(_shipped_fast_silence_ms()) in text, (
        f'{path.name} 引用优化前延迟却未给当前下发挂起值')


@pytest.mark.parametrize('path', ALL_DOCS, ids=lambda p: p.name)
def test_no_stale_raspberry_pi_preference(path):
    assert '优先树莓派' not in _text(path), (
        f'{path.name} 仍写「优先树莓派」，与已确认的 ESP32-S3 主控决策矛盾')


def test_historical_review_carries_point_in_time_note():
    assert '时点口径' in _text(HISTORICAL), '时点审查报告必须自带口径补注'


def test_optimized_bench_matches_shipped_config():
    art = DOCS / 'bench_perceived_latency_optimized.json'
    if not art.exists():
        pytest.skip('优化工件未生成')
    got = (json.loads(_text(art)).get('stream') or {}).get('fast_silence_ms')
    assert got == _shipped_fast_silence_ms(), '优化工件挂起值与 config.yaml 不一致'
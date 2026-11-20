# -*- coding: utf-8 -*-
"""报告生成脚本的回归测试：数字必须来自 JSON 工件，缺失真人数据时标注待采集。"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

DOCS = ROOT / 'docs'


@pytest.fixture(scope='module')
def generated_report(tmp_path_factory):
    """用真实工件生成一份报告到临时目录，返回 (markdown文本, 各工件dict)。"""
    import report_test

    test_eval = DOCS / 'pipeline_eval.json'
    val_eval = DOCS / 'pipeline_eval_val.json'
    bench = DOCS / 'bench_latency.json'
    ablation = DOCS / 'ablation_old_model.json'
    if not test_eval.exists():
        pytest.skip('评估工件未生成，先跑 evaluate_pipeline.py')

    out = tmp_path_factory.mktemp('report') / '测试报告.md'
    # 显式指向一个不存在的真人工件，保证「待采集」分支被测到
    report_test.main([
        '--test-eval', str(test_eval),
        '--val-eval', str(val_eval),
        '--bench', str(bench),
        '--ablation', str(ablation),
        '--human-eval', str(DOCS / 'no_such_human.json'),
        '--out', str(out),
    ])
    text = out.read_text(encoding='utf-8')
    artifacts = {
        'test': json.loads(test_eval.read_text(encoding='utf-8')),
        'val': json.loads(val_eval.read_text(encoding='utf-8')) if val_eval.exists() else None,
        'bench': json.loads(bench.read_text(encoding='utf-8')) if bench.exists() else None,
    }
    return text, artifacts


def test_test_set_accuracy_appears(generated_report):
    text, art = generated_report
    acc = art['test']['variants']['with_vad']['known']['argmax_accuracy']
    assert f'{acc:.4f}' in text


def test_sample_count_appears(generated_report):
    text, art = generated_report
    n = art['test']['variants']['with_vad']['known']['n']
    assert str(n) in text


def test_confusion_misread_marked(generated_report):
    text, art = generated_report
    conf = art['test']['variants']['with_vad']['known'].get('confusion') or {}
    off = [(t, p, c) for t, row in conf.items() for p, c in row.items() if p != t]
    if not off:
        pytest.skip('测试集无误判')
    truth, pred, _ = max(off, key=lambda x: x[2])
    # 最大误判流向应在结论里点出
    assert f'{truth}' in text and f'{pred}' in text


def test_bench_stages_present(generated_report):
    text, art = generated_report
    if art['bench'] is None:
        pytest.skip('bench 工件未生成')
    for label in ['VAD 切段', '特征提取', '推理', '用户感知等待']:
        assert label in text
    p50 = art['bench']['summary']['user_wait_ms']['p50']
    assert str(p50) in text


def test_human_section_placeholder(generated_report):
    text, _ = generated_report
    assert '待采集' in text

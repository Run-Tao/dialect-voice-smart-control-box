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


# --- 口径守卫：同一量在报告里只能有一个权威值 -------------------------

def test_coverage_bracket_contains_true_value():
    """粗网格曲线只能给区间；区间必须夹住真值，且不得把上邻当成该阈值的值。"""
    import report_test as rt

    curve = [{'threshold': 0.9883, 'coverage': 0.6159},
             {'threshold': 1.0, 'coverage': 0.0023}]
    lo, hi = rt.coverage_bracket(curve, 0.9924)
    assert lo['threshold'] == 0.9883 and hi['threshold'] == 1.0
    # 覆盖率随阈值单调不增 => 真值落在 [上邻, 下邻]
    assert hi['coverage'] <= 0.5568 <= lo['coverage']
    # 旧写法（向上取邻点）会给出 0.0023，与真值差两个数量级
    snapped = next(r['coverage'] for r in curve if r['threshold'] >= 0.9924 - 1e-9)
    assert snapped == 0.0023
    assert snapped != 0.5568


def test_coverage_bracket_exact_grid_hit():
    import report_test as rt

    curve = [{'threshold': 0.94, 'coverage': 0.82},
             {'threshold': 0.95, 'coverage': 0.80}]
    lo, hi = rt.coverage_bracket(curve, 0.95)
    assert lo['threshold'] == hi['threshold'] == 0.95
    assert rt.coverage_bracket([], 0.5) == (None, None)
    assert rt.coverage_bracket(curve, None) == (None, None)


def test_deployed_coverage_prefers_eval_artifact_over_recorded():
    """三个评估脚本一致的精确值优先；工件记录值只作冲突披露。"""
    import report_test as rt

    selective = {'baseline_maxprob': {
        'syn_known_coverage_at_deployed_threshold': 0.7864}}
    thresholds = {'known_coverage_on_validation': 0.8114, 'n_known_val': 440}
    exact, recorded = rt.deployed_coverage(selective, {'acceptance_rate': 0.7864},
                                           thresholds)
    assert exact == 0.7864 and recorded == 0.8114
    # 缺 selective 时退回验证集工件，仍不取工件记录值
    exact2, _ = rt.deployed_coverage(None, {'acceptance_rate': 0.7864}, thresholds)
    assert exact2 == 0.7864


def test_solve_table_discloses_grid_resolution(generated_report):
    """反解表不得把网格末点覆盖率当成反解阈值处的值。"""
    text, _ = generated_report
    rf = json.loads((DOCS / 'real_far.json').read_text(encoding='utf-8'))
    solved = rf.get('threshold_solve') or []
    if not solved:
        pytest.skip('real_far.json 无 threshold_solve')
    assert '分辨率披露' in text
    assert '网格区间' in text
    curve = (rf.get('risk_coverage') or {}).get('curve') or []
    import report_test as rt
    for item in solved:
        lo, hi = rt.coverage_bracket(curve, item['threshold'])
        if lo and hi and lo['threshold'] != hi['threshold']:
            # 该行必须以区间形式出现，而不是单独一个 snap 值
            assert f"{hi['coverage']:.4f}, {lo['coverage']:.4f}" in text


def test_deployed_coverage_conflict_is_disclosed(generated_report):
    text, _ = generated_report
    th = json.loads((ROOT / 'runtime' / 'models' / 'v3_thresholds_vad.json')
                    .read_text(encoding='utf-8'))
    sel = json.loads((DOCS / 'selective_eval.json').read_text(encoding='utf-8'))
    exact = sel['baseline_maxprob']['syn_known_coverage_at_deployed_threshold']
    recorded = th['known_coverage_on_validation']
    assert f'{exact:.4f}' in text
    if abs(recorded - exact) > 5e-4:
        assert '口径冲突待修' in text
        assert f'{recorded:.4f}' in text
        n_val = th.get('n_known_val')
        if n_val:
            n_rec, n_exact = round(recorded * n_val), round(exact * n_val)
            assert f'工件 {n_rec} 条 vs 实测 {n_exact} 条' in text
            assert f'差 {abs(n_rec - n_exact)} 条' in text


def test_aurc_reported_once_in_signal_section(generated_report):
    """§3 不再另印一个 AURC 数字，权威值只在 §4.1。"""
    text, _ = generated_report
    rf = json.loads((DOCS / 'real_far.json').read_text(encoding='utf-8'))
    own = (rf.get('risk_coverage') or {}).get('aurc_cross_domain')
    sel = json.loads((DOCS / 'selective_eval.json').read_text(encoding='utf-8'))
    authoritative = sel['signals']['maxprob']['aurc_cross_domain']
    assert f'{authoritative:.4f}' in text
    assert '第 4.1 节' in text
    # §3 区间内不得出现 real_far 自己那份 AURC 数值
    sec3 = text.split('## 3.')[1].split('## 4.')[0]
    if own is not None and abs(own - authoritative) > 1e-9:
        assert f'{own:.4f}' not in sec3


def test_hangover_comes_from_artifact_not_hardcoded(generated_report):
    """§6.3 的部署侧挂起必须等于工件 stream.fast_silence_ms。"""
    text, _ = generated_report
    b = json.loads((DOCS / 'bench_perceived_latency.json').read_text(encoding='utf-8'))
    hang = b['stream']['fast_silence_ms']
    sec63 = text.split('### 6.3')[1].split('### 6.4')[0]
    assert f'{hang:.0f} ms' in sec63
    assert 'stream.fast_silence_ms' in sec63


def test_optimization_section_reads_new_artifacts(generated_report):
    text, _ = generated_report
    for name in ('fast_endpoint_sweep_endpoint_only.json', 'red_soft_eval.json',
                 'bench_perceived_latency_optimized.json',
                 'dialog_eval_stream_optimized_v2.json'):
        if not (DOCS / name).exists():
            pytest.skip(f'{name} 未生成')
    assert '#### 6.4.1' in text and '#### 6.4.2' in text
    assert '#### 6.4.3' in text and '#### 6.4.4' in text
    opt = json.loads((DOCS / 'dialog_eval_stream_optimized_v2.json')
                     .read_text(encoding='utf-8'))
    assert f"{opt['dialog_session_success']:.4f}" in text
    hr = json.loads((DOCS / 'red_soft_eval.json').read_text(encoding='utf-8'))
    real = hr['high_risk']['groups']['real']
    assert f"{real['effective_dangerous_rate']:.4f}" in text
    bench = json.loads((DOCS / 'bench_perceived_latency_optimized.json')
                       .read_text(encoding='utf-8'))
    assert f"{bench['stream']['fast_silence_ms']:.0f}" in text


def test_s5_verdict_follows_gain_sign(generated_report):
    """S5 达成与否必须由工件增益判定，不能写死。"""
    text, _ = generated_report
    p = DOCS / 'dialog_eval_stream_optimized_v2.json'
    if not p.exists():
        pytest.skip('优化工件未生成')
    opt = json.loads(p.read_text(encoding='utf-8'))
    gain = (opt.get('confusable_pair') or {}).get('gain_vs_single_gate')
    sec11 = text.split('## 11.')[1].split('## 12.')[0]
    if gain is not None and gain > 0:
        assert '已达成' in sec11
        assert '仍未达成' not in sec11
    else:
        assert '仍未达成' in sec11


def test_synthetic_corpus_caveat_present(generated_report):
    text, _ = generated_report
    assert 'edge-tts 合成粤语' in text
    assert '真人**命令正样本为 0**' in text or '真人命令正样本为 0' in text


# --- 覆盖率复核：必须与工件同源，否则整列弃用 -------------------------

def test_coverage_recheck_degrades_without_cache(tmp_path):
    import report_test as rt

    exact, note = rt.coverage_recheck(str(tmp_path / 'nope.npz'), 0.945, 0.7864, [0.9924])
    assert exact is None
    assert '不存在' in note
    exact, note = rt.coverage_recheck(None, 0.945, 0.7864, [0.9924])
    assert exact is None


def test_coverage_recheck_rejects_mismatched_population(tmp_path):
    """缓存若在部署阈值上复现不了工件值，说明不是同一批样本，必须弃用而不是印出来。"""
    import numpy as np
    import report_test as rt

    p = tmp_path / 'fake.npz'
    # 全 0.99：在 0.945 上覆盖率 1.0，与工件 0.7864 不符
    np.savez(p, **{'syn_known/prob': np.full((10, 6), 0.99, dtype=np.float32)})
    exact, note = rt.coverage_recheck(str(p), 0.945, 0.7864, [0.9924])
    assert exact is None
    assert '不同源' in note


def test_coverage_recheck_uses_artifact_operator(tmp_path):
    """同源缓存下，复核值必须等于用 eval_real_far 的 searchsorted 算子直接算的结果。"""
    import numpy as np
    import report_test as rt

    rng = np.random.default_rng(0)
    prob = rng.random((440, 6)).astype(np.float32)
    p = tmp_path / 'ok.npz'
    np.savez(p, **{'syn_known/prob': prob})
    confs = np.sort(prob.max(axis=1).astype(np.float64))
    tau = 0.945
    expected = float((confs.size - int(np.searchsorted(confs, tau, side='left'))) / confs.size)
    exact, note = rt.coverage_recheck(str(p), tau, expected, [0.9924, None])
    assert exact is not None and None not in exact
    want = float((confs.size - int(np.searchsorted(confs, 0.9924, side='left'))) / confs.size)
    assert exact[0.9924] == round(want, 4)
    assert '不是工件字段' in note


def test_section4_does_not_repeat_false_cliff(generated_report):
    """§4 不得再把网格上邻 0.0023 当成反解阈值处的覆盖率。"""
    text, _ = generated_report
    sec4 = text.split('## 4.')[1].split('## 5.')[0]
    assert '此路不通' not in sec4
    rf = json.loads((DOCS / 'real_far.json').read_text(encoding='utf-8'))
    solved = {round(s['target_far'], 2): s for s in rf['threshold_solve']}
    s = solved.get(0.1)
    if s and s.get('reachable'):
        curve = rf['risk_coverage']['curve']
        import report_test as rt
        lo, hi = rt.coverage_bracket(curve, s['threshold'])
        if lo and hi and lo['threshold'] != hi['threshold']:
            # 只能以区间或复核精确值出现，不能单独印上邻那个 snap 值
            assert f"掉到 {hi['coverage']:.4f}——" not in sec4


def test_high_risk_table_separates_semantics(generated_report):
    """合成已知组的「直接动作」是正确执行，不能与真人误动作同标为危险率。"""
    text, _ = generated_report
    if '#### 6.4.3' not in text:
        pytest.skip('red-soft 工件未生成')
    sec = text.split('#### 6.4.3')[1].split('#### 6.4.4')[0]
    assert '直接动作率' in sec
    assert '语义不同' in sec
    assert '正确执行' in sec and '误动作' in sec


def test_degraded_solve_table_has_no_empty_duplicate_column(tmp_path):
    """缓存缺失时反解表只留 4 列，不得出现一列全是 — 的重复「网格区间」。"""
    import report_test as rt

    out = tmp_path / 'degraded.md'
    rt.main(['--coverage-cache', str(tmp_path / 'missing.npz'), '--out', str(out)])
    text = out.read_text(encoding='utf-8')
    tbl = text.split('**阈值反解**')[1].split('> **分辨率披露**')[0]
    header = next(l for l in tbl.splitlines() if l.startswith('| 目标 FAR'))
    assert header.count('|') == 5          # 4 列
    assert '复核精确值' not in header
    assert '| — |' not in tbl
    assert '精确值列不可用' in text


def test_section6_labels_pre_optimization_caliber(generated_report):
    """6.1–6.3 是优化前工件，必须在节首标明，否则读者会把旧挂起当成当前生效值。"""
    text, _ = generated_report
    old = DOCS / 'bench_perceived_latency.json'
    new = DOCS / 'bench_perceived_latency_optimized.json'
    if not (old.exists() and new.exists()):
        pytest.skip('端点工件未生成')
    h0 = json.loads(old.read_text(encoding='utf-8'))['stream']['fast_silence_ms']
    h1 = json.loads(new.read_text(encoding='utf-8'))['stream']['fast_silence_ms']
    if h0 == h1:
        pytest.skip('两工件端点相同，无需分段标注')
    sec6 = text.split('## 6.')[1].split('### 6.1')[0]
    assert '不要混引' in sec6
    assert f'{h0:.0f} ms' in sec6 and f'{h1:.0f} ms' in sec6
    assert '不是当前生效值' in sec6

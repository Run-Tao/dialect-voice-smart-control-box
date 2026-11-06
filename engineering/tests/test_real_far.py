# -*- coding: utf-8 -*-
"""真实域测试集构建与 FAR 评估的测试。

分两层：
- 纯函数层（Wilson 区间、阈值反解、风险-覆盖）不依赖任何语料，永远跑；
  这几个是统计口径，算错了整个「真实域 FAR」结论就不成立。
- 端到端层需要部署模型与一批 WAV，缺了就 skip，不伪造数据。
  端到端用合成语料当**替身输入**，只验证管线连通与产物结构，
  产出的 FAR 数字没有意义，测试里也不断言它的大小。
"""
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_real_testset as brt
import eval_real_far as erf
import report_test as rt

RESEARCH_CORPUS = Path(r'D:\Run_Tao\DialectVoice_Research\data\tts_corpus\wav16k_v3_1_vad')
MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'


# --------------------------------------------------------------------------
# 纯函数层
# --------------------------------------------------------------------------

def test_wilson_empty():
    assert erf.wilson(0, 0) == (None, None)


def test_wilson_zero_success_has_nonzero_upper():
    lo, hi = erf.wilson(0, 100)
    assert lo == 0.0
    assert 0.02 < hi < 0.06, hi


def test_wilson_half_is_symmetric_ish():
    lo, hi = erf.wilson(50, 100)
    assert lo < 0.5 < hi
    assert abs((lo + hi) / 2 - 0.5) < 0.01


def test_wilson_wide_on_small_sample():
    """10 条样本测出 0.1，区间必须宽——这正是报告里要带区间的原因。"""
    lo, hi = erf.wilson(1, 10)
    assert lo < 0.02 and hi > 0.3


def test_far_at_counts_greater_or_equal():
    a = np.array([0.1, 0.5, 0.5, 0.9])
    assert erf.far_at(a, 0.5) == 0.75
    assert erf.far_at(a, 0.9) == 0.25
    assert erf.far_at(a, 0.95) == 0.0
    assert erf.far_at(a, 0.0) == 1.0


def test_far_at_empty():
    assert erf.far_at(np.array([]), 0.5) == 0.0


def test_solve_thresholds_picks_smallest_sufficient():
    """0.9501 就能把 0.90/0.95 两条都挡掉（FAR=1/3），不需要抬到 0.99。"""
    solved = erf.solve_thresholds([0.90, 0.95, 0.99], [1 / 3 + 0.01])
    assert solved[0]['threshold'] == 0.9501
    assert solved[0]['far_at_threshold'] == pytest.approx(1 / 3, abs=1e-4)
    assert solved[0]['reachable'] is True


def test_solve_thresholds_unreachable_is_flagged():
    """置信度饱和到 1.0 时，任何 ≤1.0 的阈值都压不住 FAR——必须标不可达，
    而不是报一个 >1.0 的假阈值（那等于「一律拒识」，不是工作点）。"""
    solved = erf.solve_thresholds([1.0, 1.0], [0.0])
    assert solved[0]['reachable'] is False
    assert solved[0]['threshold'] is None
    assert '不是可部署工作点' in solved[0]['reason']


def test_solve_thresholds_never_exceeds_one():
    confs = list(np.linspace(0.5, 1.0, 30))
    for item in erf.solve_thresholds(confs, [0.5, 0.2, 0.1, 0.05, 0.0]):
        assert item['threshold'] is None or item['threshold'] <= 1.0


def test_solve_thresholds_report_is_self_consistent():
    """报出去的 (threshold, far_at_threshold) 必须能互相复算。

    踩过的坑：阈值取整方式不对时，写进报告的 FAR 和按该阈值重算的对不上，
    而这种不一致肉眼看不出来，只会让结论悄悄失真。
    """
    rng = np.random.default_rng(0)
    confs = list(np.round(rng.uniform(0.3, 1.0, 200), 6))
    a = np.sort(np.asarray(confs))
    for item in erf.solve_thresholds(confs, [0.5, 0.2, 0.1, 0.05, 0.0]):
        if not item['reachable']:
            continue
        actual = erf.far_at(a, item['threshold'])
        assert round(actual, 4) == item['far_at_threshold']
        assert actual <= item['target_far'] + 1e-12


def test_solve_thresholds_can_exclude_tied_confidences():
    """全部样本同分时，目标 FAR=0 需要一个严格大于该分数的阈值，
    不能因为「候选里没有」就误判成不可达。"""
    solved = erf.solve_thresholds([0.5] * 10, [0.0])
    assert solved[0]['reachable'] is True
    assert solved[0]['threshold'] > 0.5
    assert solved[0]['far_at_threshold'] == 0.0


def test_solve_thresholds_monotone_in_target():
    confs = list(np.linspace(0.5, 0.99, 50))
    solved = erf.solve_thresholds(confs, [0.20, 0.10, 0.05])
    thresholds = [s['threshold'] for s in solved]
    assert all(t is not None for t in thresholds)
    assert thresholds == sorted(thresholds), thresholds


def test_summarize_matches_manual_count():
    out = erf.summarize([0.99, 0.95, 0.10, 0.05], 0.945)
    assert out['n'] == 4 and out['accepted'] == 2 and out['far'] == 0.5
    assert out['ci95'][0] <= 0.5 <= out['ci95'][1]


def test_risk_coverage_requires_both_sides():
    assert erf.risk_coverage([], [0.9]) is None
    assert erf.risk_coverage([0.9], []) is None


def test_risk_coverage_curve_is_monotone():
    real = list(np.linspace(0.3, 0.98, 40))
    known = list(np.linspace(0.6, 1.0, 40))
    rc = erf.risk_coverage(real, known)
    # 阈值升序 → 覆盖率与 FAR 同时下降，所以按覆盖率排序后 FAR 也必须有序
    pairs = sorted((r['coverage'], r['far']) for r in rc['curve'])
    fars = [f for _, f in pairs]
    assert fars == sorted(fars)
    assert 0.0 <= rc['aurc_cross_domain'] <= 1.0
    assert '非同一数据集' in rc['note']


def test_risk_coverage_aurc_not_degenerate():
    """每档覆盖率要取「覆盖率最高且不超过该档」的工作点。

    踩过的坑：取覆盖率最低那点，每一档都落到 FAR=0，AURC 恒等于 0，
    看起来「风险为零」，其实只是算错了。
    """
    real = list(np.linspace(0.05, 0.99, 60))
    known = list(np.linspace(0.30, 1.00, 60))
    rc = erf.risk_coverage(real, known)
    assert rc['coverage_levels_used'] >= 15
    assert rc['aurc_cross_domain'] > 0.2, rc['aurc_cross_domain']


def test_coverage_at_picks_first_threshold_not_below():
    """取「不低于查询阈值的最小网格点」——阈值只会往上抬，覆盖率随之往下掉。"""
    curve = [{'threshold': 0.90, 'coverage': 0.95, 'far': 0.5},
             {'threshold': 0.95, 'coverage': 0.80, 'far': 0.3},
             {'threshold': 0.99, 'coverage': 0.40, 'far': 0.1}]
    assert rt.coverage_at(curve, 0.90) == 0.95
    assert rt.coverage_at(curve, 0.945) == 0.80   # 网格上 >=0.945 的第一点是 0.95
    assert rt.coverage_at(curve, 0.95) == 0.80
    assert rt.coverage_at(curve, 0.97) == 0.40
    assert rt.coverage_at(curve, 0.995) == 0.0    # 超出网格 → 一律拒识
    assert rt.coverage_at([], 0.9) is None
    assert rt.coverage_at(curve, None) is None


# --------------------------------------------------------------------------
# 端到端层（缺模型或缺语料则 skip）
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def stand_in_corpus(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip('部署模型缺失，先跑 sync_runtime.py')
    if not RESEARCH_CORPUS.exists():
        pytest.skip('合成语料缺失，无法构造替身输入')
    wavs = sorted(RESEARCH_CORPUS.rglob('*.wav'))[:6]
    if len(wavs) < 3:
        pytest.skip('合成语料样本不足')
    corpus = tmp_path_factory.mktemp('corpus')
    for i, src in enumerate(wavs):
        (corpus / f'standin_{i:03d}.wav').write_bytes(src.read_bytes())
    return corpus


def test_build_requires_license(stand_in_corpus, tmp_path):
    with pytest.raises(SystemExit):
        brt.main(['--source', f'fake={stand_in_corpus}', '--out', str(tmp_path / 'o')])


def test_build_end_to_end(stand_in_corpus, tmp_path):
    out = tmp_path / 'testset'
    prov = brt.main(['--source', f'fake={stand_in_corpus}',
                     '--license', 'fake=UNVERIFIED',
                     '--out', str(out)])
    manifest = out / 'manifest.csv'
    assert manifest.exists()
    rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig', newline='')))
    assert rows, '没有产出任何片段'
    assert prov['segments'] == len(rows)
    assert all(r['split'] == 'real_test' for r in rows)
    assert all(r['source_corpus'] == 'fake' for r in rows)

    for r in rows:
        p = Path(r['audio_path'])
        assert p.exists()
        assert p.name.startswith('real_fake_')
        with wave.open(str(p), 'rb') as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0

    provenance = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
    assert provenance['sources']['fake']['license'] == 'UNVERIFIED'
    assert provenance['vad_kwargs'], '同源纪律要求记录 VAD 参数'
    assert '不用于命令正样本训练' in provenance['usage']


def test_build_dry_run_writes_nothing(stand_in_corpus, tmp_path):
    out = tmp_path / 'dry'
    brt.main(['--source', f'fake={stand_in_corpus}', '--license', 'fake=UNVERIFIED',
              '--out', str(out), '--dry-run'])
    assert not (out / 'manifest.csv').exists()


def test_eval_real_far_end_to_end(stand_in_corpus, tmp_path):
    out = tmp_path / 'testset'
    brt.main(['--source', f'fake={stand_in_corpus}', '--license', 'fake=UNVERIFIED',
              '--out', str(out)])
    report_path = tmp_path / 'real_far.json'
    report = erf.main(['--manifest', str(out / 'manifest.csv'),
                       '--targets', '0.5,0.1',
                       '--out', str(report_path)])
    assert report_path.exists()

    head = report['headline']
    assert head['n'] > 0
    assert head['far'] is None or 0.0 <= head['far'] <= 1.0
    lo, hi = head['ci95']
    assert lo is not None and lo <= head['far'] <= hi

    assert report['threshold'] == pytest.approx(0.945, abs=1e-6)
    assert report['testset']['per_source']['fake']['n'] == head['n']
    for item in report['threshold_solve']:
        if item['reachable']:
            assert item['far_at_threshold'] <= item['target_far'] + 1e-9

    # 误接受落到哪个命令，必须只可能是已知命令之一
    assert set(report['false_accept_by_command']) <= set(report['classes'])
    assert '不代表真人命令识别率' in report['note']


def test_eval_real_far_window_filter(stand_in_corpus, tmp_path):
    out = tmp_path / 'testset2'
    brt.main(['--source', f'fake={stand_in_corpus}', '--license', 'fake=UNVERIFIED',
              '--out', str(out)])
    durations = sorted(float(r['duration_ms']) for r in csv.DictReader(
        (out / 'manifest.csv').open(encoding='utf-8-sig', newline='')))
    manifest = str(out / 'manifest.csv')

    full = erf.main(['--manifest', manifest, '--out', str(tmp_path / 'full.json')])
    assert full['testset']['rows'] == len(durations)
    assert full['testset']['window_ms'] is None

    # 窗口只放最短那一档，行数必须精确等于该档的条数
    lo = durations[0]
    narrow = erf.main(['--manifest', manifest, '--window-ms', f'{lo}-{lo}',
                       '--out', str(tmp_path / 'narrow.json')])
    assert narrow['testset']['rows'] == sum(1 for d in durations if d == lo)
    assert narrow['testset']['window_ms'] == [lo, lo]


def test_eval_real_far_empty_selection_exits(stand_in_corpus, tmp_path):
    """一条都选不出来时必须退出，不能报一个 n=0 的 FAR。"""
    out = tmp_path / 'testset3'
    brt.main(['--source', f'fake={stand_in_corpus}', '--license', 'fake=UNVERIFIED',
              '--out', str(out)])
    with pytest.raises(SystemExit):
        erf.main(['--manifest', str(out / 'manifest.csv'), '--window-ms', '0-1',
                  '--out', str(tmp_path / 'impossible.json')])

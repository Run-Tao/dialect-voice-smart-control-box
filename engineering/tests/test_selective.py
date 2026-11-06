# -*- coding: utf-8 -*-
"""选择性分类（runtime/selective.py）与逐种子概率接口的测试。

分两层，沿用 test_real_far.py 的口径：
- 纯函数层永远跑。这里锁的是**统计口径**：工作点反解必须是全局最优、
  AUC 的并列修正、AND 规则必须逐样本精确、AURC 不得退化成 0。
  这些算错了，「第二信号有没有用」的结论就不成立。
- 端到端层缺模型或语料就 skip，不伪造数据。端到端只验证结构与
  「重构后部署置信度逐位不变」，不断言任何 FAR/AUC 数值大小。

重构回归是本文件的重点：``predict_features`` 改成走 ``_forward`` 之后，
部署置信度必须与旧的「softmax(各种子 logits 均值)」路径完全一致，
否则线上判定会被静默改变。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from runtime.selective import (  # noqa: E402
    ACCEPT_SIGNALS, and_accept, and_workpoint, auc, aurc, build_pool, combo_min_z,
    combo_sum_z, domain_detector_gap, ece, entropy, js_divergence, knn_distance,
    l2_normalize, risk_coverage, seed_agreement, seed_std, signals, solve_workpoint,
    zapply, zstats,
)
from runtime.v3_cnn_numpy import prepare_input, softmax_np  # noqa: E402

MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'
THRESHOLDS = ROOT / 'runtime' / 'models' / 'v3_thresholds_vad.json'
RESEARCH_CORPUS = Path(r'D:\Run_Tao\DialectVoice_Research\data\tts_corpus\wav16k_v3_1_vad')
REAL_TESTSET = Path(r'D:\Run_Tao\DialectVoice_Research\data\real_domain_testset\manifest.csv')


def onehot(i, n=6):
    p = np.full(n, 1e-9)
    p[i] = 1.0
    return p / p.sum()


# --------------------------------------------------------------------------
# 纯函数层：信号
# --------------------------------------------------------------------------

def test_js_identical_is_zero():
    p = np.array([0.7, 0.2, 0.1])
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_disjoint_reaches_ln2():
    assert js_divergence(onehot(0), onehot(1)) == pytest.approx(np.log(2.0), abs=1e-6)
    assert js_divergence(np.eye(6)[0], np.eye(6)[1]) == pytest.approx(np.log(2.0), abs=1e-12)


def test_js_is_symmetric_and_bounded():
    rng = np.random.default_rng(7)
    for _ in range(20):
        p, q = rng.random(6), rng.random(6)
        p, q = p / p.sum(), q / q.sum()
        d = js_divergence(p, q)
        assert d == pytest.approx(js_divergence(q, p), abs=1e-12)
        assert -1e-12 <= d <= np.log(2.0) + 1e-12


def test_seed_std_zero_for_single_seed_and_for_identical_rows():
    assert seed_std(np.array([[0.9, 0.1]])) == 0.0
    assert seed_std(np.array([[0.9, 0.1], [0.9, 0.1]])) == 0.0


def test_seed_std_positive_when_confidence_spreads():
    assert seed_std(np.array([[0.99, 0.01], [0.55, 0.45]])) == pytest.approx(0.22)


def test_seed_std_blind_to_label_disagreement():
    """两个种子都满信心但**指向不同命令**时 seed_std = 0。

    这是 seed_std 的真实局限：它量的是置信度离散度，不是标签分歧。
    要用标签分歧就得看 seed_agree（基于 JS，能抓到这种情况）。
    """
    sp = np.stack([onehot(0), onehot(1)])
    assert seed_std(sp) == pytest.approx(0.0, abs=1e-6)
    assert seed_agreement(sp) == pytest.approx(0.0, abs=1e-6)


def test_seed_agreement_endpoints():
    assert seed_agreement(np.array([[0.9, 0.1], [0.9, 0.1]])) == pytest.approx(1.0)
    assert seed_agreement(onehot(0)[None, :]) == 1.0
    assert seed_agreement(np.stack([onehot(0), onehot(1)])) == pytest.approx(0.0, abs=1e-6)


def test_seed_agreement_in_unit_interval():
    rng = np.random.default_rng(3)
    for _ in range(20):
        sp = rng.random((3, 6))
        sp /= sp.sum(axis=1, keepdims=True)
        assert 0.0 - 1e-12 <= seed_agreement(sp) <= 1.0 + 1e-12


def test_l2_normalize_and_build_pool():
    pool = build_pool([np.array([3.0, 4.0]), np.array([0.0, 0.0])])
    assert pool.shape == (2, 2)
    assert np.linalg.norm(pool[0]) == pytest.approx(1.0)
    # 零向量不能被除成 nan，这是 kNN 在静音片段上的真实风险
    assert np.all(np.isfinite(pool[1]))


def test_build_pool_empty():
    assert build_pool([]).shape[0] == 0


def test_knn_distance_self_is_zero_and_none_on_empty_pool():
    pool = build_pool([np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])])
    assert knn_distance(np.array([2.0, 0.0, 0.0]), pool, k=1) == pytest.approx(0.0, abs=1e-9)
    assert knn_distance(np.array([1.0, 0.0, 0.0]), np.zeros((0, 3)), k=1) is None
    assert knn_distance(np.array([1.0, 0.0, 0.0]), None, k=1) is None


def test_knn_distance_orthogonal_is_one_and_k_is_clamped():
    pool = build_pool([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert knn_distance(np.array([1.0, 0.0]), pool, k=1) == pytest.approx(0.0, abs=1e-9)
    # k 超过池大小必须被夹住，否则 partition 越界
    assert knn_distance(np.array([1.0, 0.0]), pool, k=99) is not None


def test_knn_distance_grows_off_manifold():
    rng = np.random.default_rng(11)
    pool = build_pool([rng.normal(size=8) + np.array([5.0] * 8) for _ in range(30)])
    # k=1 且查询与池中某点同方向（余弦距离 0）；k>1 会把邻近点也算进来，不再是 0
    assert knn_distance(pool[0] * 3.0, pool, k=1) == pytest.approx(0.0, abs=1e-9)
    assert knn_distance(-pool[0], pool, k=1) > knn_distance(pool[0], pool, k=1)
    assert knn_distance(-pool[0], pool, k=5) > knn_distance(pool[0], pool, k=5)


def test_entropy_uniform_is_log_c():
    assert entropy(np.full(6, 1 / 6)) == pytest.approx(np.log(6.0), abs=1e-9)
    assert entropy(onehot(2)) == pytest.approx(0.0, abs=1e-6)


def test_signals_keys_and_consistency():
    sp = np.stack([np.array([0.8, 0.1, 0.1, 0, 0, 0]),
                   np.array([0.4, 0.5, 0.1, 0, 0, 0])])
    pool = build_pool([np.array([1.0] + [0.0] * 7)])
    row = signals(np.array([0.75, 0.15, 0.05, 0.02, 0.02, 0.01]), sp,
                  embedding=np.array([1.0] + [0.0] * 7), pool=pool, k=1)
    assert set(row) == set(ACCEPT_SIGNALS) | {'seed_std', 'knn_dist', 'entropy'}
    assert row['maxprob'] == pytest.approx(0.75)
    assert row['margin'] == pytest.approx(0.60)
    assert row['neg_entropy'] == pytest.approx(-row['entropy'])
    assert row['ood'] == pytest.approx(-row['knn_dist'])
    assert row['seed_agree'] == pytest.approx(seed_agreement(sp))


def test_signals_ood_none_without_pool():
    row = signals(np.array([0.9, 0.1]), np.array([[0.9, 0.1]]), embedding=np.array([1.0, 2.0]))
    assert row['ood'] is None and row['knn_dist'] is None


def test_signals_normalizes_unnormalized_input():
    row = signals(np.array([2.0, 2.0]), np.array([[0.5, 0.5]]))
    assert row['maxprob'] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 纯函数层：组合规则
# --------------------------------------------------------------------------

def test_zstats_std_zero_falls_back_to_one():
    s = zstats([0.5, 0.5, 0.5])
    assert s['std'] == 1.0 and s['mean'] == pytest.approx(0.5)
    assert zstats([])['n'] == 0
    assert zstats([None, None])['n'] == 0


def test_zapply_none_passthrough():
    assert zapply(None, {'mean': 0.0, 'std': 1.0}) is None


def test_combo_min_z_equals_min_of_zscores():
    stats = {'maxprob': {'mean': 0.5, 'std': 0.1}, 'ood': {'mean': -0.1, 'std': 0.05}}
    row = {'maxprob': 0.7, 'ood': -0.2}
    expected = min((0.7 - 0.5) / 0.1, (-0.2 + 0.1) / 0.05)
    assert combo_min_z(row, stats) == pytest.approx(expected)
    assert combo_sum_z(row, stats) == pytest.approx(2.0 + expected)


def test_combo_min_z_is_and_rule_in_score_space():
    """min-z 过单阈值 <=> 每个信号的 z 都过同一阈值。这是它能画风险-覆盖曲线的前提。"""
    stats = {'a': {'mean': 0.0, 'std': 1.0}, 'b': {'mean': 0.0, 'std': 1.0}}
    rng = np.random.default_rng(5)
    for _ in range(50):
        row = {'a': float(rng.normal()), 'b': float(rng.normal())}
        tau = float(rng.normal())
        assert (combo_min_z(row, stats) >= tau) == (row['a'] >= tau and row['b'] >= tau)


def test_combo_returns_none_when_a_signal_is_missing():
    stats = {'maxprob': {'mean': 0.5, 'std': 0.1}, 'ood': {'mean': 0.0, 'std': 1.0}}
    assert combo_min_z({'maxprob': 0.9, 'ood': None}, stats) is None


def test_and_accept_requires_every_signal_and_rejects_missing():
    th = {'maxprob': 0.9, 'ood': -0.05}
    assert and_accept({'maxprob': 0.95, 'ood': -0.01}, th) is True
    assert and_accept({'maxprob': 0.95, 'ood': -0.09}, th) is False
    assert and_accept({'maxprob': 0.10, 'ood': -0.01}, th) is False
    assert and_accept({'maxprob': 0.95, 'ood': None}, th) is False


def test_and_workpoint_reports_exact_per_sample_rates():
    """返回的 coverage/risk 必须能用 and_accept 逐样本复核出来。

    用「各信号单独接受率的 min」当近似是错的（对风险是上界、对覆盖率也是上界），
    这条测试就是防止再退回去。
    """
    rng = np.random.default_rng(17)
    n_pos, n_neg = 120, 200
    neg = {'a': list(rng.normal(0.0, 1.0, n_neg)), 'b': list(rng.normal(0.0, 1.0, n_neg))}
    pos = {'a': list(rng.normal(1.5, 1.0, n_pos)), 'b': list(rng.normal(1.5, 1.0, n_pos))}
    res = and_workpoint(neg, pos, ('a', 'b'), target_risk=0.05, n_grid=25)
    assert res['reachable'] is True
    assert res['risk'] <= 0.05 + 1e-12
    acc_neg = [and_accept({'a': a, 'b': b}, res['thresholds']) for a, b in zip(neg['a'], neg['b'])]
    acc_pos = [and_accept({'a': a, 'b': b}, res['thresholds']) for a, b in zip(pos['a'], pos['b'])]
    assert res['risk'] == pytest.approx(sum(acc_neg) / n_neg, abs=1e-4)
    assert res['coverage'] == pytest.approx(sum(acc_pos) / n_pos, abs=1e-4)


def test_and_workpoint_beats_single_signal_when_signals_are_complementary():
    """两个各带一半信息的信号，AND 规则应比任一单信号在同等 FAR 下覆盖更高。"""
    rng = np.random.default_rng(23)
    n = 400
    neg = {'a': list(rng.normal(0.0, 1.0, n)), 'b': list(rng.normal(0.0, 1.0, n))}
    pos = {'a': list(rng.normal(1.0, 1.0, n)), 'b': list(rng.normal(1.0, 1.0, n))}
    both = and_workpoint(neg, pos, ('a', 'b'), target_risk=0.05, n_grid=30)
    single = solve_workpoint(neg['a'], pos['a'], 0.05)
    assert both['reachable'] and single['reachable']
    assert both['coverage'] > single['coverage']


def test_and_workpoint_unreachable_and_mismatched():
    n = 30
    ones = {'a': [1.0] * n}
    assert and_workpoint(ones, {'a': [1.0] * n}, ('a',), target_risk=-1.0)['reachable'] is False
    bad = and_workpoint({'a': [1.0] * n, 'b': [1.0] * (n - 1)},
                        {'a': [1.0] * n, 'b': [1.0] * n}, ('a', 'b'), target_risk=0.5)
    assert bad['reachable'] is False and '样本数不一致' in bad['reason']
    with pytest.raises(ValueError):
        and_workpoint({}, {}, (), target_risk=0.1)


# --------------------------------------------------------------------------
# 纯函数层：风险-覆盖与工作点反解
# --------------------------------------------------------------------------

def _brute_workpoint(neg, pos, target):
    """穷举参考实现：候选阈值取两侧观测值，逐个算 FAR/覆盖率。

    并列时取**最高**阈值（风险随阈值单调不增，同覆盖率下高阈值严格占优），
    与 solve_workpoint 的打破平局规则保持一致。
    """
    cands = sorted(set(neg) | set(pos))
    best = None
    for t in cands:
        risk = sum(1 for v in neg if v >= t) / len(neg)
        if risk > target + 1e-12:
            continue
        cov = sum(1 for v in pos if v >= t) / len(pos)
        if best is None or cov > best['coverage'] + 1e-12 or (
                abs(cov - best['coverage']) <= 1e-12 and t > best['threshold']):
            best = {'threshold': t, 'risk': risk, 'coverage': cov}
    return best


@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_solve_workpoint_matches_brute_force(seed):
    """searchsorted 版本必须与穷举一致——反解错了整个工作点结论就废了。"""
    rng = np.random.default_rng(seed)
    neg = list(np.round(rng.beta(1.2, 1.2, 300), 6))
    pos = list(np.round(rng.beta(6.0, 1.5, 300), 6))
    for target in (0.20, 0.10, 0.05):
        got = solve_workpoint(neg, pos, target)
        want = _brute_workpoint(neg, pos, target)
        if want is None:
            assert got['reachable'] is False
            continue
        assert got['reachable'] is True
        # 输出四舍五入到 4 位小数，容差按这个口径给
        assert got['risk'] == pytest.approx(want['risk'], abs=1e-4)
        assert got['coverage'] == pytest.approx(want['coverage'], abs=1e-4)
        assert got['threshold'] == pytest.approx(want['threshold'], abs=1e-6)
        assert got['risk'] <= target + 1e-12


def test_solve_workpoint_tie_break_prefers_lowest_risk():
    """覆盖率并列时必须选 FAR 更低的那个阈值，否则白丢安全性。

    neg=[0.1,0.2,0.3,0.9]、pos=[0.55,0.56,0.7]：阈值 0.2 / 0.3 / 0.55 都给出
    coverage=1.0（三个正样本全过），risk 分别是 0.75 / 0.50 / 0.25。
    取最低阈值会白白报出 3 倍的风险，必须取 0.55。
    """
    res = solve_workpoint([0.1, 0.2, 0.3, 0.9], [0.55, 0.56, 0.7], target_risk=0.75)
    assert res['reachable'] is True
    assert res['coverage'] == pytest.approx(1.0, abs=1e-4)
    assert res['threshold'] == pytest.approx(0.55, abs=1e-6)
    assert res['risk'] == pytest.approx(0.25, abs=1e-4)


def test_solve_workpoint_unreachable_reports_floor():
    res = solve_workpoint([1.0, 1.0, 1.0], [0.5, 0.6], target_risk=0.0)
    assert res['reachable'] is False
    assert res['min_reachable_risk'] is not None
    assert '重训' in res['reason']


def test_solve_workpoint_empty_side():
    assert solve_workpoint([], [0.5], 0.1)['reachable'] is False
    assert solve_workpoint([0.5], [], 0.1)['reachable'] is False


def test_solve_workpoint_none_values_are_dropped():
    res = solve_workpoint([None, 0.1, 0.2], [None, 0.9], 0.5)
    assert res['n_neg'] == 2 and res['n_pos'] == 1


def test_risk_coverage_is_monotone_and_curves_are_aligned():
    rng = np.random.default_rng(9)
    neg = list(rng.beta(1.2, 3.0, 200))
    pos = list(rng.beta(5.0, 1.2, 200))
    rc = risk_coverage(neg, pos, n_points=40)
    cov = [r['coverage'] for r in rc['curve']]
    risk = [r['risk'] for r in rc['curve']]
    thr = [r['threshold'] for r in rc['curve']]
    assert thr == sorted(thr)
    # 阈值升高时正负样本的接受率**都**下降：覆盖率不增，风险同样不增。
    # 早先这里写成「FAR 随阈值上升」，方向反了，会掩盖真正的工作点选择错误。
    assert all(a >= b - 1e-9 for a, b in zip(cov, cov[1:]))
    assert all(a >= b - 1e-9 for a, b in zip(risk, risk[1:]))
    assert rc['n_pos'] == 200 and rc['n_neg'] == 200


def test_risk_coverage_none_when_a_side_is_empty():
    assert risk_coverage([], [0.5]) is None
    assert risk_coverage([0.5], []) is None


def test_aurc_not_degenerate():
    """AURC 必须取「coverage<=level 且覆盖率最高」那点。

    取最低覆盖率那点会让每档都落到 risk=0，AURC 恒等于 0（S0 踩过的坑）。
    """
    rng = np.random.default_rng(13)
    neg = list(rng.beta(1.5, 1.5, 400))
    pos = list(rng.beta(4.0, 1.5, 400))
    rc = risk_coverage(neg, pos, n_points=60)
    value = aurc(rc, tuple(round(1.0 - 0.05 * i, 2) for i in range(20)))
    assert value is not None and value > 0.02
    assert aurc(None, (1.0,)) is None


def test_aurc_tie_break_picks_highest_threshold_on_coverage_plateau():
    """覆盖率平台上的并列必须取最高阈值那点，否则 AURC 被读成平台上最松的值。

    neg=[0.0,0.01,0.02]、pos=[0.9,0.95,1.0] 完全可分：coverage=1.0 的最低阈值
    是 0.9，此时负样本全被拒（risk=0）。但曲线上 threshold<0.9 的点同样给出
    coverage=1.0 却 risk=1.0，取第一个就会把这一档读成 1.0。
    """
    rc = risk_coverage([0.0, 0.01, 0.02], [0.9, 0.95, 1.0], n_points=30)
    assert aurc(rc, (1.0, 0.5)) == 0.0


def test_aurc_is_positive_when_classes_overlap():
    rng = np.random.default_rng(29)
    neg = list(rng.beta(2.0, 2.0, 300))
    pos = list(rng.beta(2.5, 2.0, 300))
    value = aurc(risk_coverage(neg, pos, n_points=60), (1.0, 0.8, 0.5))
    assert value is not None and value > 0.05


def test_accept_rate_boundaries():
    from runtime.selective import accept_rate
    a = np.array([0.1, 0.5, 0.5, 0.9])
    assert accept_rate(a, 0.5) == pytest.approx(0.75)   # >= 判据，含并列
    assert accept_rate(a, 0.95) == 0.0
    assert accept_rate(a, 0.0) == 1.0
    assert accept_rate(np.array([]), 0.5) == 0.0


# --------------------------------------------------------------------------
# 纯函数层：诊断指标
# --------------------------------------------------------------------------

def test_auc_perfect_reversed_and_ties():
    assert auc([1.0, 2.0], [0.0, 0.5]) == 1.0
    assert auc([0.0, 0.5], [1.0, 2.0]) == 0.0
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auc([], [0.5]) is None


def test_auc_matches_direct_pairwise_count():
    rng = np.random.default_rng(19)
    pos = list(rng.normal(1.0, 1.0, 80))
    neg = list(rng.normal(0.0, 1.0, 90))
    wins = sum(1 for p in pos for n in neg if p > n) + \
        0.5 * sum(1 for p in pos for n in neg if p == n)
    assert auc(pos, neg) == pytest.approx(wins / (len(pos) * len(neg)), abs=1e-4)


def test_auc_skips_none():
    assert auc([1.0, None, 2.0], [0.0, None]) == 1.0


def test_domain_detector_gap_verdicts():
    assert domain_detector_gap(0.99, 0.52)['verdict'] == '域检测器'
    assert domain_detector_gap(0.99, 0.75)['verdict'] == '部分依赖域'
    assert domain_detector_gap(0.90, 0.88)['verdict'] == '内容判别器'
    assert domain_detector_gap(None, 0.9) is None
    d = domain_detector_gap(0.99, 0.52)
    assert d['gap'] == pytest.approx(0.47) and '不能外推' in d['note']


def test_ece_perfect_calibration_and_overconfidence():
    # 置信度 0.8 的样本里恰好 80% 正确 -> ECE≈0
    conf = [0.8] * 100
    corr = [True] * 80 + [False] * 20
    assert ece(conf, corr)['ece'] == pytest.approx(0.0, abs=1e-9)
    over = ece([1.0] * 100, [True] * 50 + [False] * 50)
    assert over['ece'] == pytest.approx(0.5, abs=1e-9)


def test_ece_tied_confidences_do_not_create_phantom_error():
    """置信度完全相同的样本必须落进同一箱。

    等样本数分箱会把并列值切开，各箱准确率凭空不同，ECE 报出一个纯粹由分箱
    方式造成的假误差——本项目置信度大量并列在 1.0，这个坑必踩。
    """
    corr = ([True] * 4 + [False]) * 40               # 整体 80% 正确，但正误交错
    conf = [0.85] * 200
    out = ece(conf, corr, bins=10)
    assert out['ece'] == pytest.approx(0.05, abs=1e-9)   # |0.85 - 0.80|
    assert len(out['bins']) == 1 and out['bins'][0]['n'] == 200


def test_ece_bins_are_equal_width_and_keep_confidence_one():
    conf = [1.0] * 90 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.95]
    corr = [True] * 90 + [False] * 10
    out = ece(conf, corr, bins=10)
    assert out['binning'] == 'equal_width'
    assert sum(b['n'] for b in out['bins']) == 100
    top = [b for b in out['bins'] if b['hi'] >= 1.0]
    assert top and top[0]['n'] == 91       # 90 个 1.0 加 1 个 0.95；1.0 不能被右开区间漏掉
    assert all(b['hi'] - b['lo'] == pytest.approx(0.1) for b in out['bins'])


def test_ece_rejects_mismatched_or_empty():
    assert ece([], []) is None
    assert ece([0.5, 0.6], [True]) is None
    assert ece([None, None], [True, False]) is None


# --------------------------------------------------------------------------
# 端到端层：逐种子概率接口与重构回归
# --------------------------------------------------------------------------

needs_model = pytest.mark.skipif(
    not MODEL.exists(), reason=f'缺部署模型 {MODEL}')
needs_corpus = pytest.mark.skipif(
    not RESEARCH_CORPUS.is_dir(), reason=f'缺合成语料 {RESEARCH_CORPUS}')


def _predictor():
    from runtime.v3_model import V3Predictor
    return V3Predictor.load(str(MODEL), str(THRESHOLDS))


def _sample_seq(predictor, n=3):
    files = sorted(RESEARCH_CORPUS.glob('*.wav'))[:n]
    return [(f, predictor.feature_sequence(f)) for f in files]


@needs_model
@needs_corpus
def test_predict_detailed_exposes_per_seed_probabilities():
    m = _predictor()
    _, seq = _sample_seq(m, 1)[0]
    pred, extras = m.predict_detailed(seq)
    sp = extras['seed_probs']
    assert sp.shape[0] == len(m._seed_params) * max(1, len(m.tta_shifts) or 1)
    assert sp.shape[1] == len(m.classes_)
    assert np.allclose(sp.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(sp >= 0.0)
    assert extras['embedding'].shape == (256,)
    assert pred.confidence == pytest.approx(m.predict_features(seq).confidence, abs=1e-12)


@needs_model
@needs_corpus
def test_deployed_confidence_unchanged_by_forward_refactor():
    """部署置信度必须逐位等于旧的 softmax(各种子 logits 均值) 路径。"""
    m = _predictor()
    for path, seq in _sample_seq(m, 4):
        x = prepare_input(seq, m.n_frames)
        legacy = softmax_np(m._raw_logits(x), m.temperature)[0]
        legacy = legacy / legacy.sum()
        got = m.predict_features(seq)
        assert got.confidence == pytest.approx(float(legacy.max()), abs=1e-12)
        assert got.command == m.classes_[int(np.argmax(legacy))]
        for i, c in enumerate(m.classes_):
            assert got.scores[c] == pytest.approx(float(legacy[i]), abs=1e-12)


@needs_model
@needs_corpus
def test_seed_mean_probability_is_not_the_deployed_confidence():
    """两种聚合方式不等价——这条测试钉住区别，防止有人拿逐种子均值替换线上判定。"""
    m = _predictor()
    _, seq = _sample_seq(m, 1)[0]
    pred, extras = m.predict_detailed(seq)
    seed_mean = extras['seed_probs'].mean(axis=0).max()
    assert seed_mean != pytest.approx(pred.confidence, abs=1e-9) or len(m._seed_params) == 1


@needs_model
@needs_corpus
def test_signals_from_real_forward_have_expected_ranges():
    m = _predictor()
    pool = build_pool([m.embed_features(s) for _, s in _sample_seq(m, 3)])
    for _, seq in _sample_seq(m, 3):
        pred, extras = m.predict_detailed(seq)
        prob = np.array([pred.scores[c] for c in m.classes_])
        row = signals(prob, extras['seed_probs'], extras['embedding'], pool, k=2)
        assert 0.0 <= row['maxprob'] <= 1.0
        assert 0.0 <= row['margin'] <= 1.0
        assert 0.0 <= row['seed_agree'] <= 1.0
        assert row['neg_entropy'] <= 0.0
        assert row['ood'] <= 0.0
        assert all(np.isfinite(row[k]) for k in ACCEPT_SIGNALS)


@needs_model
@needs_corpus
def test_predict_detailed_single_forward_pass_matches_two_calls():
    """predict_detailed 不得靠跑两遍网络拿信号（那样延迟翻倍，端侧受不了）。"""
    import time
    m = _predictor()
    _, seq = _sample_seq(m, 1)[0]
    m.predict_detailed(seq)
    t0 = time.perf_counter()
    for _ in range(5):
        m.predict_detailed(seq)
    t_detailed = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(5):
        m.predict_features(seq)
    t_plain = (time.perf_counter() - t0) / 5
    # 同一遍前向，允许噪声；若 detailed 明显更慢说明又跑了一次卷积
    assert t_detailed < t_plain * 1.6 + 5e-3


# --------------------------------------------------------------------------
# 端到端层：评估脚本产物结构（不断言数值大小）
# --------------------------------------------------------------------------

@needs_model
@needs_corpus
@pytest.mark.skipif(not REAL_TESTSET.exists(), reason=f'缺真实域测试集 {REAL_TESTSET}')
def test_eval_selective_end_to_end_structure(tmp_path):
    import eval_selective as es
    unknown = REAL_TESTSET.parent.parent / 'tts_corpus' / 'unknown_v3_vad'
    if not unknown.is_dir():
        pytest.skip(f'缺合成非命令目录 {unknown}')
    corpus = REAL_TESTSET.parent.parent / 'tts_corpus' / 'manifest_yue_v3_1_vad.csv'
    if not corpus.exists():
        pytest.skip(f'缺合成语料 manifest {corpus}')
    out = tmp_path / 'selective.json'
    report = es.main([
        '--manifest', str(REAL_TESTSET),
        '--corpus-manifest', str(corpus),
        '--unknown-dir', str(unknown),
        '--limit', '12',
        '--raw-cache', str(tmp_path / 'raw.npz'),
        '--out', str(out),
    ])
    assert out.exists()
    for key in ('baseline_maxprob', 'signals', 'and_rules', 'groups', 'caveats',
                'z_stats_calibrated_on'):
        assert key in report
    for name in es.SIGNAL_ORDER:
        s = report['signals'][name]
        for key in ('auc_cross_domain', 'auc_same_domain', 'domain_gap',
                    'workpoints_cross_domain', 'workpoints_same_domain',
                    'aurc_cross_domain', 'percentiles'):
            assert key in s, f'{name} 缺 {key}'
    # 基线必须与 S0 的判据同源：部署阈值下的真实域接受率是个合法比例
    base = report['baseline_maxprob']
    assert 0.0 <= base['real_far_at_deployed_threshold'] <= 1.0
    assert base['real_ci95'][0] <= base['real_far_at_deployed_threshold'] <= base['real_ci95'][1]
    assert base['ece_on_syn_known']['n'] == report['groups']['syn_known']['n']
    for key, res in report['and_rules'].items():
        assert res['target_risk'] == 0.10
        if res['calibrated_on_synthetic'].get('reachable'):
            ap = res['applied_to_real']
            assert 0.0 <= ap['real_far'] <= 1.0
            assert ap['real_ci95'][0] <= ap['real_far'] <= ap['real_ci95'][1]


@needs_model
@needs_corpus
def test_eval_selective_reuse_raw_cache_skips_inference(tmp_path):
    """改口径重算不应重跑推理——缓存往返必须无损。"""
    import eval_selective as es
    unknown = REAL_TESTSET.parent.parent / 'tts_corpus' / 'unknown_v3_vad'
    corpus = REAL_TESTSET.parent.parent / 'tts_corpus' / 'manifest_yue_v3_1_vad.csv'
    if not unknown.is_dir() or not corpus.exists():
        pytest.skip('缺合成语料')
    cache = tmp_path / 'raw.npz'
    common = ['--manifest', str(REAL_TESTSET), '--corpus-manifest', str(corpus),
              '--unknown-dir', str(unknown), '--limit', '6', '--raw-cache', str(cache)]
    first = es.main(common + ['--out', str(tmp_path / 'a.json')])
    assert cache.exists()
    second = es.main(common + ['--reuse-raw', '--out', str(tmp_path / 'b.json')])
    assert second['groups'] == first['groups']
    assert second['signals']['ood']['auc_cross_domain'] == \
        first['signals']['ood']['auc_cross_domain']
    a = json.loads((tmp_path / 'a.json').read_text(encoding='utf-8'))
    b = json.loads((tmp_path / 'b.json').read_text(encoding='utf-8'))
    assert a['baseline_maxprob'] == b['baseline_maxprob']

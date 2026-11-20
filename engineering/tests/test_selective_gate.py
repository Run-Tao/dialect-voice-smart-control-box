# -*- coding: utf-8 -*-
"""S4 双信号拒识门限的测试：``SelectiveGate``、交叉拟合标定、参照池构建与实时路径接线。

沿用既有两层口径：
- 纯函数层永远跑。这里锁的是**接线正确性**而不是指标好坏：``ood`` 的负号、
  缺件时的降级路径、交叉拟合的折划分必须真的互斥、分层子采样必须真的每类都在。
  这几件事错任何一个，线上跑的就不是实测过的那条规则，而 FAR 数字看起来仍然「还行」。
- 端到端层缺工件或语料就 skip，只验证结构与「下发的阈值 == 构建报告里的阈值」，
  不断言任何 FAR/覆盖率数值大小（唯一例外是 ``meets_target``，那是项目自己预先
  写死的判据，不是观测量）。
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
    SelectiveGate, and_accept, and_workpoint, build_pool, crossfit_and_rule,
    knn_distance, l2_normalize, single_threshold_gate)
from src.config import Config  # noqa: E402
from src.realtime import DEFAULT_GATE, DialectEngine  # noqa: E402

import eval_pool_size as eps  # noqa: E402

MODEL = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'
THRESHOLDS = ROOT / 'runtime' / 'models' / 'v3_thresholds_vad.json'
POOL = ROOT / 'runtime' / 'models' / 'v3_knn_pool.npz'
GATE_REPORT = ROOT / 'docs' / 'selective_gate.json'
RESEARCH_CORPUS = Path(r'D:\Run_Tao\DialectVoice_Research\data\tts_corpus\wav16k_v3_1_vad')


def prob_of(i, n=6, value=0.99):
    p = np.full(n, (1.0 - value) / (n - 1))
    p[i] = value
    return p


def unit(rng, dim=8):
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------
# 纯函数层：SelectiveGate 的判定与降级
# --------------------------------------------------------------------------
def test_gate_accepts_near_pool_and_rejects_far_embedding():
    rng = np.random.default_rng(0)
    v = unit(rng)
    gate = SelectiveGate(thresholds={'maxprob': 0.5, 'ood': -0.2},
                         pool=build_pool([v]), k=1, fallback_threshold=0.945)
    near = gate.decide(prob_of(0), v)
    assert near['rule'] == 'and' and near['accepted'] is True and near['reason'] == ''
    assert near['signals']['ood'] == pytest.approx(0.0, abs=1e-9)

    far = unit(rng)                       # 随机方向：8 维下并不正交，距离靠算不靠猜
    d = knn_distance(far, gate.pool, 1)
    assert d > 0.5                        # 确实离池远，才会被 ood 这一路拒掉
    out = gate.decide(prob_of(0), far)
    assert out['rule'] == 'and' and out['accepted'] is False
    assert out['signals']['ood'] == pytest.approx(-float(d), abs=1e-12)
    assert 'ood' in out['reason'] and 'maxprob' not in out['reason']


def test_gate_ood_sign_is_negated_distance():
    """``ood`` 必须是 **-kNN 距离**。

    这个负号漏掉过一次：AND 规则整体反向，「离池远」变成接受，满池 FAR 从
    0.0660 变成 0.2325，而数字看着仍然「还行」。所以直接钉死符号。

    注意 ``SelectiveGate`` 把池存成 float32（与下发的 f16→f32 工件一致），
    所以精确比对要用 ``gate.pool``，拿外部 float64 池比只能给 f32 级容差。
    """
    rng = np.random.default_rng(1)
    pool = build_pool([unit(rng) for _ in range(4)])
    gate = SelectiveGate(thresholds={'ood': -1.0}, pool=pool, k=2)
    assert gate.pool.dtype == np.float32
    emb = unit(rng)
    d32 = knn_distance(emb, gate.pool, 2)
    d64 = knn_distance(emb, pool, 2)
    got = gate.signals_of(prob_of(0), emb)['ood']
    assert got == pytest.approx(-float(d32), abs=1e-12)
    assert got == pytest.approx(-float(d64), abs=1e-6)
    assert got < 0.0                        # 符号本身：距离为正 => ood 为负
    # 池内点在 k=1 下距离为 0；k>1 会把邻居一起平均，所以这一点只在 k=1 上钉
    gate1 = SelectiveGate(thresholds={'ood': -1.0}, pool=pool, k=1)
    assert gate1.signals_of(prob_of(0), pool[0])['ood'] == pytest.approx(0.0, abs=1e-6)
    # k=2 时池内点仍必须显著比随机点更靠近池（ood 更大 = 更像命令）
    assert gate.signals_of(prob_of(0), pool[0])['ood'] > got


def test_gate_without_embedding_degrades_to_single_threshold():
    rng = np.random.default_rng(2)
    gate = SelectiveGate(thresholds={'maxprob': 0.3, 'ood': -0.2},
                         pool=build_pool([unit(rng)]), k=1, fallback_threshold=0.9)
    hi = gate.decide(prob_of(0, value=0.95), None)
    lo = gate.decide(prob_of(0, value=0.5), None)
    assert hi['rule'] == 'fallback_single' and hi['accepted'] is True
    assert lo['rule'] == 'fallback_single' and lo['accepted'] is False
    assert 'ood' in lo['reason']
    assert gate.degraded is False          # 池在，只是这一条没给嵌入


def test_gate_missing_pool_is_marked_degraded():
    gate = SelectiveGate(thresholds={'maxprob': 0.3, 'ood': -0.2}, pool=None,
                         fallback_threshold=0.9)
    assert gate.degraded is True and gate.degraded_reason == 'ood_pool_missing'
    out = gate.decide(prob_of(0, value=0.95), np.zeros(8))
    assert out['rule'] == 'fallback_single' and out['accepted'] is True


def test_gate_reject_fallback_refuses_everything():
    gate = SelectiveGate(thresholds={'maxprob': 0.3, 'ood': -0.2}, pool=None,
                         fallback_threshold=0.0, fallback_mode='reject')
    out = gate.decide(prob_of(0, value=1.0), None)
    assert out['accepted'] is False and out['rule'] == 'fallback_reject'


def test_maxprob_only_gate_does_not_degrade():
    """只有 maxprob 的门限（对照组）必须走 AND 路径，不能被当成「缺 ood」降级。

    降级路径用的是 fallback_threshold，与规则里的阈值可以不同；
    把单信号门限误判成缺件，会让对照组的判定口径悄悄变成另一个数。
    """
    gate = single_threshold_gate(0.5)
    out = gate.decide(prob_of(0, value=0.7), None)
    assert out['rule'] == 'and' and out['accepted'] is True
    assert gate.degraded is False
    assert single_threshold_gate(0.5).decide(prob_of(0, value=0.3), None)['accepted'] is False


def test_gate_rejects_unknown_signal_and_bad_fallback_mode():
    with pytest.raises(ValueError):
        SelectiveGate(thresholds={'nope': 0.5}, pool=None)
    with pytest.raises(ValueError):
        SelectiveGate(thresholds={}, pool=None)
    with pytest.raises(ValueError):
        SelectiveGate(thresholds={'maxprob': 0.5}, pool=None, fallback_mode='whatever')


def test_gate_normalizes_prob_and_rejects_illegal_input():
    gate = single_threshold_gate(0.5)
    assert gate.signals_of(np.array([0.0, 2.0, 2.0]))['maxprob'] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        gate.signals_of(np.zeros(3))
    with pytest.raises(ValueError):
        gate.signals_of(np.array([1.0, np.nan]))


def test_gate_load_roundtrip_pool_size_guard_and_missing_pool(tmp_path):
    rng = np.random.default_rng(3)
    emb = build_pool([unit(rng) for _ in range(3)])
    np.savez(str(tmp_path / 'pool.npz'), embedding=emb.astype(np.float32),
             label=np.array(['A', 'B', 'C']))
    rule = {'version': 1, 'rule': 'and', 'signals': ['maxprob', 'ood'],
            # 故意用整数化的假阈值：下发工件（500 条 f16 池）的数值不应出现在
            # 单元测试的夹具里，否则读代码的人会把这条 3 向量池当成真的那份
            'thresholds': {'maxprob': 0.3, 'ood': -0.02},
            'ood': {'kind': 'neg_knn_cosine', 'k': 5, 'pool': 'pool.npz',
                    'pool_array': 'embedding', 'pool_size': 3},
            'fallback': {'mode': 'single', 'threshold': 0.945}}
    rp = tmp_path / 'gate.json'
    rp.write_text(json.dumps(rule, ensure_ascii=False), encoding='utf-8')

    gate = SelectiveGate.load(rp, models_dir=tmp_path)
    assert gate.pool is not None and gate.pool.shape == (3, emb.shape[1])
    assert gate.k == 5 and gate.fallback_threshold == pytest.approx(0.945)
    assert gate.meta['pool_path'] and gate.degraded is False

    # 池条数与规则不符必须硬失败：阈值是按 3 条标定的，用 5 条池等于用错阈值
    rule['ood']['pool_size'] = 5
    rp.write_text(json.dumps(rule, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError):
        SelectiveGate.load(rp, models_dir=tmp_path)

    # 池文件不在 => 降级而不是抛异常（模型先到、池后到是正常下发状态）
    rule['ood']['pool_size'] = 3
    rule['ood']['pool'] = 'absent.npz'
    rp.write_text(json.dumps(rule, ensure_ascii=False), encoding='utf-8')
    degraded = SelectiveGate.load(rp, models_dir=tmp_path)
    assert degraded.pool is None and degraded.degraded is True
    assert degraded.decide(prob_of(0, value=0.99), None)['rule'] == 'fallback_single'


def test_gate_as_dict_is_serializable_and_omits_pool():
    gate = SelectiveGate(thresholds={'maxprob': 0.3, 'ood': -0.2},
                         pool=np.zeros((4, 8), dtype=np.float32), k=3)
    d = json.loads(json.dumps(gate.as_dict()))
    assert d['pool_size'] == 4 and d['ood_k'] == 3 and d['degraded'] is False
    assert 'embedding' not in json.dumps(d)


# --------------------------------------------------------------------------
# 纯函数层：交叉拟合标定
# --------------------------------------------------------------------------
def _separable(n_neg=200, n_pos=100, seed=0):
    rng = np.random.default_rng(seed)
    return ({'maxprob': list(rng.uniform(0.0, 0.4, n_neg)),
             'ood': list(rng.uniform(-1.0, -0.6, n_neg))},
            {'maxprob': list(rng.uniform(0.6, 1.0, n_pos)),
             'ood': list(rng.uniform(-0.3, 0.0, n_pos))})


def test_crossfit_folds_partition_the_negatives():
    neg, pos = _separable()
    xf = crossfit_and_rule(neg, pos, ('maxprob', 'ood'), target_risk=0.10,
                           n_folds=3, seeds=(0, 1))
    assert xf['reachable'] is True
    for seed in (0, 1):
        folds = [f for f in xf['folds'] if f['seed'] == seed and f['reachable']]
        assert len(folds) == 3
        assert sum(f['n_eval'] for f in folds) == len(neg['maxprob'])
        for f in folds:
            assert f['n_calib'] == len(neg['maxprob']) - f['n_eval']
    # 每次随机划分的合并 FAR 都必须独立报出，不能只报一次划分
    assert len(xf['pooled']) == 2


def test_crossfit_is_out_of_sample_and_hits_target_on_separable_data():
    neg, pos = _separable()
    xf = crossfit_and_rule(neg, pos, ('maxprob', 'ood'), target_risk=0.10,
                           n_folds=2, seeds=(0, 1, 2))
    assert xf['n_distinct_thresholds'] == 1
    assert xf['pooled_far_mean'] == 0.0 and xf['pooled_far_max'] == 0.0
    assert xf['pos_coverage_mean'] == 1.0 and xf['pos_coverage_min'] == 1.0


def test_workpoint_report_matches_the_thresholds_it_ships():
    """报出的 coverage/risk 必须能用它返回的 thresholds 复算出来。

    踩过：阈值按 6 位小数四舍五入后写进 JSON，而 coverage 是在未取整的网格上算的。
    可分离数据上报 coverage 1.0，拿下发的阈值一算是 0.99——最小那条正样本被舍入
    抬到阈值之上拒掉了。现在网格先按下发精度向下取整再搜，两者指同一条规则。
    """
    neg, pos = _separable()
    w = and_workpoint(neg, pos, ('maxprob', 'ood'), target_risk=0.10)
    th = w['thresholds']
    n_pos, n_neg = len(pos['maxprob']), len(neg['maxprob'])
    cov = sum(and_accept({n: pos[n][i] for n in th}, th) for i in range(n_pos)) / n_pos
    risk = sum(and_accept({n: neg[n][i] for n in th}, th) for i in range(n_neg)) / n_neg
    assert round(cov, 4) == w['coverage']
    assert round(risk, 4) == w['risk']
    assert risk <= w['target_risk']


def test_crossfit_agrees_with_full_fit_when_workpoint_is_unique():
    """工作点唯一时，交叉拟合的众数阈值必须等于全量拟合的阈值。

    下发的那对阈值来自全量拟合，报出的 FAR 来自交叉拟合；两者不是同一对数字时，
    报出的 FAR 就属于别的规则。``build_selective_gate.py`` 会检查这件事，
    这里把「什么时候应该相等」钉死。
    """
    neg, pos = _separable()
    xf = crossfit_and_rule(neg, pos, ('maxprob', 'ood'), target_risk=0.10,
                           n_folds=2, seeds=(0,))
    full = and_workpoint(neg, pos, ('maxprob', 'ood'), target_risk=0.10)
    modal = xf['threshold_modes'][0]['thresholds']
    assert modal == full['thresholds']


def test_crossfit_reports_unreachable_when_negatives_are_too_few():
    neg, pos = _separable(n_neg=3, n_pos=20)
    out = crossfit_and_rule(neg, pos, ('maxprob', 'ood'), target_risk=0.10, n_folds=2)
    assert out['reachable'] is False and '不足以' in out['reason']


def test_crossfit_reachable_guards_risk_only_not_usability():
    """``reachable`` 只保证「压得到目标 FAR」，不保证这条规则能用。

    两类完全重叠时，把阈值抬到负样本高分位数之上照样能满足 FAR ≤ 0.10，
    于是 ``reachable=True``——但覆盖率同时塌到判据以下。项目判据是两条
    （FAR ≤ 0.10 **且** 覆盖率 ≥ 0.80，见 ``eps.ACCEPT_*``），少看任何一条
    都会把「不可用」读成「达标」。这里把两者的分工钉死：函数报风险可达，
    调用方（``eval_pool_size`` / ``build_selective_gate``）负责合判覆盖率。
    """
    rng = np.random.default_rng(4)
    shared = list(rng.uniform(0.0, 1.0, 120))
    neg = {'maxprob': shared[:60], 'ood': [-0.5] * 60}
    pos = {'maxprob': shared[60:], 'ood': [-0.5] * 60}
    out = crossfit_and_rule(neg, pos, ('maxprob', 'ood'), target_risk=0.10, n_folds=2)
    assert out['reachable'] is True
    assert out['pooled_far_max'] <= 0.10
    assert out['pos_coverage_mean'] < eps.ACCEPT_MIN_COVERAGE   # 但不可用
    assert 'folds' in out                      # 失败也要留下逐折记录，便于排查


# --------------------------------------------------------------------------
# 纯函数层：参照池构建与批量 kNN
# --------------------------------------------------------------------------
def test_make_pool_idx_is_stratified_and_reproducible():
    rng = np.random.default_rng(5)
    classes = ['A', 'B', 'C', 'D', 'E', 'F']
    emb = rng.normal(size=(60, 8))
    labs = np.array([classes[i % 6] for i in range(60)])

    pool, idx = eps.make_pool_idx('subsample', emb, labs, 12, 0, classes)
    assert pool.shape[0] == 12 == len(idx)
    assert len(set(idx)) == len(idx)                       # 不放回
    assert all(0 <= i < 60 for i in idx)
    got = {str(labs[i]) for i in idx}
    assert got == set(classes), f'分层子采样漏了类别: {set(classes) - got}'
    assert np.allclose(pool, l2_normalize(emb[idx]).astype(np.float64), atol=1e-9)

    again = eps.make_pool_idx('subsample', emb, labs, 12, 0, classes)[1]
    assert again == idx                                    # 同种子可复现（下发要固定）
    other = eps.make_pool_idx('subsample', emb, labs, 12, 1, classes)[1]
    assert other != idx
    assert eps.make_pool('subsample', emb, labs, 12, 0, classes).shape == pool.shape


def test_make_pool_full_and_centroid_shapes():
    rng = np.random.default_rng(6)
    emb = rng.normal(size=(30, 8))
    labs = np.array([['A', 'B', 'C'][i % 3] for i in range(30)])
    full, idx = eps.make_pool_idx('full', emb, labs, 999, 0, ['A', 'B', 'C'])
    assert full.shape[0] == 30 and idx == list(range(30))
    cent, cidx = eps.make_pool_idx('centroid', emb, labs, 2, 0, ['A', 'B', 'C'])
    assert cent.shape[0] == 6 and cidx is None             # 质心没有源下标
    with pytest.raises(ValueError):
        eps.make_pool('nope', emb, labs, 5, 0, ['A'])


def test_batch_knn_matches_per_row_knn_distance():
    rng = np.random.default_rng(7)
    pool = build_pool([unit(rng, 16) for _ in range(25)])
    queries = np.stack([unit(rng, 16) for _ in range(9)])
    for k in (1, 3, 5, 99):                                # 99 会被夹到池大小
        fast = eps.batch_knn(queries, pool, k)
        slow = [knn_distance(queries[i], pool, k) for i in range(len(queries))]
        assert np.allclose(fast, np.asarray(slow, dtype=np.float64), atol=1e-12)


def test_accept_criteria_are_the_published_ones():
    """判据是预先写死的，不能看完结果再挪。改动必须是有意的、并同步改文档。"""
    assert eps.ACCEPT_MAX_FAR == 0.10
    assert eps.ACCEPT_MIN_COVERAGE == 0.80
    assert eps.PAIR == ('maxprob', 'ood')


# --------------------------------------------------------------------------
# 端到端层：下发工件与实时路径
# --------------------------------------------------------------------------
@pytest.mark.skipif(not (DEFAULT_GATE.exists() and POOL.exists()),
                    reason='缺下发工件，先跑 scripts/build_selective_gate.py + sync_runtime.py')
def test_shipped_gate_artifact_matches_build_report():
    gate = SelectiveGate.load(DEFAULT_GATE, models_dir=DEFAULT_GATE.parent)
    rule = json.loads(DEFAULT_GATE.read_text(encoding='utf-8'))
    assert gate.pool is not None and gate.degraded is False
    assert gate.pool.shape[0] == rule['ood']['pool_size']
    assert gate.k == rule['ood']['k']
    assert gate.thresholds == {k: float(v) for k, v in rule['thresholds'].items()}
    assert set(gate.thresholds) == {'maxprob', 'ood'}
    assert gate.fallback_mode in ('single', 'reject')

    if GATE_REPORT.exists():
        rep = json.loads(GATE_REPORT.read_text(encoding='utf-8'))
        # 下发的阈值必须就是构建报告里那对；否则线上跑的不是被评估过的规则
        assert gate.thresholds == {k: float(v)
                                   for k, v in rep['deployed_fit']['thresholds'].items()}
        assert rep['meets_target'] is True, f'工件未达判据: {rep["problems"]}'
        assert rep['crossfit']['n_distinct_thresholds'] == 1
        assert rule['ood']['pool_size'] == rep['pool']['shape'][0]


@pytest.mark.skipif(not DEFAULT_GATE.exists(), reason='缺规则文件')
def test_engine_build_gate_is_off_by_default_and_loud_when_requested():
    assert DialectEngine.build_gate(selective=None) is None
    assert DialectEngine.build_gate(selective=False) is None
    gate = DialectEngine.build_gate(selective=True)
    assert gate is not None and gate.pool is not None
    with pytest.raises(FileNotFoundError):
        DialectEngine.build_gate(gate_path=ROOT / 'runtime' / 'models' / 'absent.json',
                                 selective=True)


@pytest.mark.skipif(not MODEL.exists(), reason=f'缺部署模型 {MODEL}')
def test_process_utterance_records_which_rule_decided(tmp_path):
    """同一段音频、两条路径：结构必须都留下痕迹，但不断言接受与否。"""
    wavs = sorted(RESEARCH_CORPUS.rglob('*.wav')) if RESEARCH_CORPUS.exists() else []
    if not wavs:
        pytest.skip(f'缺合成语料 {RESEARCH_CORPUS}')
    from runtime.audio_io import load_wav
    audio, sr = load_wav(wavs[0])

    plain = DialectEngine.load(str(MODEL), str(THRESHOLDS), selective=False)
    r1 = plain.process_utterance(audio, sr)
    assert r1.rule == 'single_threshold'
    assert set(r1.signals) == {'maxprob'}
    assert 'rule' in r1.as_dict() and 'signals' in r1.as_dict()

    if not (DEFAULT_GATE.exists() and POOL.exists()):
        pytest.skip('缺下发工件，跳过双信号路径')
    dual = DialectEngine.load(str(MODEL), str(THRESHOLDS), selective=True)
    r2 = dual.process_utterance(audio, sr)
    assert r2.rule in ('and', 'fallback_single')
    assert 'maxprob' in r2.signals
    if r2.rule == 'and':
        assert 'ood' in r2.signals
    # 同一次前向的部署概率在两条路径下必须一致，否则比较毫无意义
    assert r2.confidence == pytest.approx(r1.confidence, abs=1e-12)
    assert r2.command == r1.command


def test_config_selective_defaults_off_and_validates_fallback_mode():
    assert Config(raw={}).selective_enabled is False
    assert Config(raw={}).selective_fallback_mode == 'single'
    cfg = Config(raw={'recognition': {'selective': {
        'enabled': True, 'gate': 'runtime/models/v3_selective_gate.json',
        'pool': 'runtime/models/v3_knn_pool.npz', 'fallback_mode': 'reject'}}})
    assert cfg.selective_enabled is True
    assert cfg.selective_fallback_mode == 'reject'
    assert cfg.selective_gate_path.name == 'v3_selective_gate.json'
    assert cfg.selective_pool_path.name == 'v3_knn_pool.npz'
    bad = Config(raw={'recognition': {'selective': {'fallback_mode': 'yolo'}}})
    with pytest.raises(ValueError):
        bad.selective_fallback_mode


def test_repo_config_enables_selective_with_shipped_artifacts():
    """仓库里的 config.yaml 必须指向真实存在的工件。

    这条测试的意义：有人改了池文件名或忘了 sync，配置就会指向空气，
    而 ``build_gate`` 在 selective=True 时是硬失败的——失败要发生在测试里，
    不是在答辩现场。
    """
    from src.config import load_config
    cfg = load_config()
    if not cfg.selective_enabled:
        pytest.skip('config.yaml 未启用双信号拒识')
    assert cfg.selective_gate_path.exists(), f'规则文件不存在: {cfg.selective_gate_path}'
    assert cfg.selective_pool_path is None or cfg.selective_pool_path.exists()
    gate = DialectEngine.build_gate(gate_path=str(cfg.selective_gate_path),
                                    pool_path=(str(cfg.selective_pool_path)
                                               if cfg.selective_pool_path else None),
                                    selective=True,
                                    fallback_mode=cfg.selective_fallback_mode)
    assert gate is not None and gate.degraded is False

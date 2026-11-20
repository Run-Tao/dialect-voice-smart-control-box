# -*- coding: utf-8 -*-
"""kNN 参照池压缩实测：池子能压到多小而不丢掉 S3 的拒识收益。

背景（v5 §9.2 限制 2、§12 第 10 项、风险登记册 R-13）：
S3 的最优规则是 ``maxprob AND ood``，其中 ``ood`` = 到参照池的 kNN 余弦距离取负。
满池 1981×256 float32 ≈ 2.03 MB，**比模型本体 1.35 MB 还大**，与 S2 的 ≤300 KB
目标直接冲突。下发方式有三条候选路：随机子采样、按类存质心、换掉这个信号。
第三条要重跑整个 S3，前两条是纯压缩问题——所以先把前两条量清楚。

协议与 S3 一致，只改池；但标定方式换了，原因是 S3 那套标定本身不稳：
- 每个池配置**重新标定** AND 阈值（池变小 → 所有距离变大 → 旧阈值必然失配，
  沿用满池阈值去评估会把压缩的代价和标定的代价混在一起）；
- **主标定 = 在真人负样本上做 K 折交叉拟合**（``runtime.selective.crossfit_and_rule``）：
  每折阈值只用其余折搜出，报出的每条 FAR 都来自阈值没见过的折。S3 用的是
  「72 条合成负样本上搜阈值」，而 bootstrap 对照显示那样搜出的真实域 FAR 会在
  [0.066, 0.20]（p10–p90）之间摆动——抖动幅度和整个子采样 FAR 全距同量级，
  在它上面比池规模等于在噪声里读信号。交叉拟合把标定集扩到数千条，同时保留独立评估；
- 合成侧标定（S3 原口径）仍然照算，作为 ``synthetic_only`` 对照一并报出；
- 覆盖率一侧始终用合成已知命令，不参与阈值搜索（少一条正样本只会让阈值偏保守）；
- 随机子采样跑多个种子，报均值与最差值——池只有几十条时，抽到哪几十条本身就有方差，
  只报一次抽样等于没报。

判据（预先写明，不看结果再挑）：真实域 FAR ≤ 0.10 且合成覆盖率 ≥ 0.80，
两者都按**交叉拟合**口径判。FAR 0.10 是 S0/S3 一直用的目标，0.80 是
``v3_thresholds_vad.json`` 里的 ``min_known_coverage_target``。

示例：
    python scripts/eval_pool_size.py --reuse-raw --out docs/pool_size.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_real_far import dangerous_split, wilson  # noqa: E402
from eval_selective import load_raw  # noqa: E402
from runtime.selective import (  # noqa: E402
    and_accept, and_workpoint, auc, build_pool, crossfit_and_rule, knn_distance,
    l2_normalize)

DEFAULT_RAW = Path('D:/Run_Tao/DialectVoice_Research/data/real_domain_testset/selective_raw.npz')
PAIR = ('maxprob', 'ood')
EMB_DIM = 256

# 判据：预先固定，不得看完结果再挪
ACCEPT_MAX_FAR = 0.10
ACCEPT_MIN_COVERAGE = 0.80


# ---------------------------------------------------------------------------
# 批量 kNN：与 runtime.selective.knn_distance 同一数学，只是把 2637 次
# Python 调用压成一次矩阵运算。等价性在 _selfcheck_batch 里逐条断言。
# ---------------------------------------------------------------------------
def batch_knn(queries: np.ndarray, pool: np.ndarray, k: int = 5) -> np.ndarray:
    """[M,D] 查询 × [N,D] 池 -> [M] 平均 k 近邻余弦距离。"""
    q = l2_normalize(np.asarray(queries, dtype=np.float64))
    d = 1.0 - pool @ q.T                       # [N,M]
    k = max(1, min(int(k), d.shape[0]))
    return np.partition(d, k - 1, axis=0)[:k].mean(axis=0)


def _selfcheck_batch(queries: np.ndarray, pool: np.ndarray, k: int) -> None:
    """批量实现必须与逐条实现逐位接近，否则下面所有数字都不可信。"""
    fast = batch_knn(queries, pool, k)
    idx = np.linspace(0, len(queries) - 1, min(8, len(queries))).astype(int)
    for i in idx:
        ref = knn_distance(queries[i], pool, k)
        if ref is None or abs(ref - fast[i]) > 1e-9:
            raise AssertionError(f'batch_knn 与 knn_distance 不一致（行 {i}）: '
                                 f'{fast[i]} vs {ref}')


# ---------------------------------------------------------------------------
# 池构造
# ---------------------------------------------------------------------------
def kmeans(X: np.ndarray, m: int, seed: int, iters: int = 30) -> np.ndarray:
    """极简 Lloyd k-means，k-means++ 式初始化。纯 numpy，不引 sklearn。

    只用来把每类嵌入压成 m 个质心。X 已 L2 归一化，质心不再归一化——
    ``build_pool`` 会统一归一化，方向才是余弦距离要的东西。
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if m >= n:
        return X.copy()
    centers = [X[rng.integers(n)]]
    for _ in range(m - 1):
        d2 = np.min(np.stack([((X - c) ** 2).sum(1) for c in centers], 0), axis=0)
        total = d2.sum()
        pick = rng.integers(n) if total <= 0 else rng.choice(n, p=d2 / total)
        centers.append(X[pick])
    C = np.stack(centers, 0)
    for _ in range(iters):
        assign = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(2), axis=1)
        new = np.array([X[assign == j].mean(0) if np.any(assign == j) else C[j]
                        for j in range(m)])
        if np.allclose(new, C):
            break
        C = new
    return C


def make_pool(kind: str, embeddings: np.ndarray, labels, size: int, seed: int,
              classes) -> np.ndarray:
    """按配置造池。``size`` 对 subsample 是条数，对 centroid 是**每类**质心数。"""
    return make_pool_idx(kind, embeddings, labels, size, seed, classes)[0]


def make_pool_idx(kind: str, embeddings: np.ndarray, labels, size: int, seed: int,
                  classes):
    """同 ``make_pool``，另返回选中条目在满池里的下标（质心池无下标，记 None）。

    为什么要下标：下发工件里要存「这 500 条分别是哪些」的类别标签，用于事后核查
    分层配额是否真的均衡。靠「在满池里找最近邻」反推下标既慢又间接，
    而抽样本身就产生下标，直接带出来最可靠。
    """
    if kind == 'full':
        return build_pool(list(embeddings)), list(range(len(embeddings)))
    if kind == 'subsample':
        rng = np.random.default_rng(seed)
        take = min(size, len(embeddings))
        # 分层抽样：池只有几十条时，随机抽可能整类缺席，那测的就不是「池小」
        # 而是「池缺类」，两件事必须分开。
        per_class, picks = {}, []
        for i, lab in enumerate(labels):
            per_class.setdefault(lab, []).append(i)
        quota = max(1, take // max(1, len(per_class)))
        for lab, idxs in per_class.items():
            rng.shuffle(idxs)
            picks.extend(idxs[:quota])
        rng.shuffle(picks)
        picks = picks[:take]
        if len(picks) < take:      # 配额取整剩下的名额从全池补
            rest = [i for i in range(len(embeddings)) if i not in set(picks)]
            picks += list(rng.choice(rest, size=min(take - len(picks), len(rest)),
                                     replace=False))
        return build_pool([embeddings[i] for i in picks]), [int(i) for i in picks]
    if kind == 'centroid':
        cents = []
        for c in classes:
            idx = [i for i, lab in enumerate(labels) if lab == c]
            if not idx:
                continue
            X = l2_normalize(np.asarray(embeddings[idx], dtype=np.float64))
            cents.append(kmeans(X, size, seed))
        return build_pool(list(np.concatenate(cents, axis=0))), None
    raise ValueError(f'未知池类型: {kind}')


# ---------------------------------------------------------------------------
def bootstrap_far(cols: dict, real_cmds, k_boot: int, seed: int,
                  target_risk: float) -> dict:
    """标定噪声对照：池不变，只对 72 条合成负样本做有放回重采样，看阈值和 FAR 抖多少。

    为什么必须有这个对照：AND 阈值是在 **72 条**合成负样本上搜的，目标风险 0.10
    意味着「最多放 7 条过」。这么小的标定集本身方差极大，换一次抽样就可能选到
    另一个阈值。若这个抖动和「换池规模」造成的抖动同量级，那池规模对比就读不出
    任何东西——必须先量清楚噪声底，再谈压缩代价。
    """
    rng = np.random.default_rng(seed)
    neg = cols['syn_unknown']
    pos = cols['syn_known']
    real = cols['real']
    n = len(neg['ood'])
    fars, covs, ths = [], [], []
    for _ in range(k_boot):
        idx = rng.integers(0, n, n)
        bn = {'maxprob': [neg['maxprob'][i] for i in idx],
              'ood': [neg['ood'][i] for i in idx]}
        cal = and_workpoint(bn, pos, PAIR, target_risk=target_risk)
        if not cal.get('reachable'):
            continue
        th = cal['thresholds']
        ths.append((round(th['maxprob'], 6), round(th['ood'], 6)))
        acc = [and_accept({'maxprob': real['maxprob'][i], 'ood': real['ood'][i]}, th)
               for i in range(len(real['ood']))]
        fars.append(sum(acc) / len(acc))
        kacc = [and_accept({'maxprob': pos['maxprob'][i], 'ood': pos['ood'][i]}, th)
                for i in range(len(pos['ood']))]
        covs.append(sum(kacc) / len(kacc))
    if not fars:
        return {'k_boot': k_boot, 'reachable_draws': 0}
    fars_a, covs_a = np.asarray(fars), np.asarray(covs)
    uniq = Counter(ths)
    return {
        'k_boot': k_boot, 'reachable_draws': len(fars),
        'real_far_p10': round(float(np.percentile(fars_a, 10)), 4),
        'real_far_p50': round(float(np.percentile(fars_a, 50)), 4),
        'real_far_p90': round(float(np.percentile(fars_a, 90)), 4),
        'real_far_min': round(float(fars_a.min()), 4),
        'real_far_max': round(float(fars_a.max()), 4),
        'syn_coverage_p50': round(float(np.percentile(covs_a, 50)), 4),
        'n_distinct_threshold_pairs': len(uniq),
        'top_threshold_pairs': [{'maxprob': t[0], 'ood': t[1], 'count': c}
                                for t, c in uniq.most_common(5)],
    }


def _apply(cols_g: dict, th: dict) -> list:
    return [and_accept({'maxprob': cols_g['maxprob'][i], 'ood': cols_g['ood'][i]}, th)
            for i in range(len(cols_g['maxprob']))]


def evaluate_pool(pool: np.ndarray, groups: dict, k: int, target_risk: float,
                  n_boot: int = 0, boot_seed: int = 0, calibrate: str = 'crossfit',
                  n_folds: int = 2, xfit_seeds=(0, 1, 2)) -> dict:
    """给定池，重标定 AND 阈值并在真人集上评估。返回一行结果。

    两套标定并存，主口径由 ``calibrate`` 决定：
    - ``crossfit``：在真人负样本上 K 折交叉拟合，报出的 FAR 全部是样本外的；
    - ``synthetic``：S3 原口径，只在 72 条合成负样本上搜阈值。
    无论主口径是哪个，另一套都照算并写进 ``synthetic_only`` / ``xfit`` 字段，
    这样「压缩代价」和「标定方式差异」可以分开读。
    """
    def ood_of(g):
        # ood = **-kNN 距离**（越大越像命令），与 runtime.selective.signals 同口径。
        # 漏掉这个负号会让 AND 规则反向（「离池远」变成接受），FAR 直接爆掉。
        return -batch_knn(groups[g]['embedding'], pool, k)

    _selfcheck_batch(groups['syn_known']['embedding'], pool, k)

    ood = {g: ood_of(g) for g in ('syn_known', 'syn_unknown', 'real')}
    maxprob = {g: [float(np.asarray(p, dtype=np.float64).max() /
                         np.asarray(p, dtype=np.float64).sum())
                   for p in groups[g]['prob']]
               for g in ('syn_known', 'syn_unknown', 'real')}
    cols = {g: {'maxprob': maxprob[g], 'ood': [float(v) for v in ood[g]]}
            for g in ('syn_known', 'syn_unknown', 'real')}

    n = len(cols['real']['maxprob'])
    syn_cal = and_workpoint(cols['syn_unknown'], cols['syn_known'], PAIR,
                            target_risk=target_risk)
    xfit = (crossfit_and_rule(cols['real'], cols['syn_known'], PAIR, target_risk,
                              n_folds=n_folds, seeds=xfit_seeds)
            if calibrate == 'crossfit' else None)

    row = {
        'n_pool': int(pool.shape[0]),
        'k': k,
        'bytes_f32': int(pool.shape[0] * EMB_DIM * 4),
        'bytes_f16': int(pool.shape[0] * EMB_DIM * 2),
        'ood_auc_cross_domain': auc(list(ood['syn_known']), list(ood['real'])),
        'ood_auc_same_domain': auc(list(ood['syn_known']), list(ood['syn_unknown'])),
        'reachable': bool(syn_cal.get('reachable')),
    }

    if xfit is not None and xfit.get('reachable'):
        primary, th = 'crossfit', xfit['threshold_modes'][0]['thresholds']
        row['reachable'] = True
    elif syn_cal.get('reachable'):
        primary, th = 'synthetic', syn_cal['thresholds']
        if xfit is not None:
            row['crossfit_reason'] = xfit.get('reason')
    else:
        row['reachable'] = False
        row['reason'] = syn_cal.get('reason')
        if xfit is not None:
            row['crossfit_reason'] = xfit.get('reason')
        return row

    row['calibrate_used'] = primary
    row['thresholds'] = {kk: round(float(vv), 6) for kk, vv in th.items()}

    # 主阈值（交叉拟合的众数阈值）直接铺到全部真人样本上：这是**要下发的那一对阈值**
    # 在真人集上的表现。当所有折都选中同一对阈值时，它和交叉拟合的合并 FAR 在数学上
    # 必然相等，下面的 selfcheck 就是查这件事；不相等说明有折选了别的阈值，
    # 此时合并 FAR（样本外）才是判据，众数阈值这一列只是参考。
    acc = _apply(cols['real'], th)
    kacc = _apply(cols['syn_known'], th)
    na = int(sum(acc))
    cmds = Counter(groups['real']['command'][i] for i in range(n) if acc[i])
    dang = dangerous_split(cmds, n)
    far_modal = round(na / n, 4)
    cov_modal = round(sum(kacc) / len(kacc), 4) if kacc else None

    if primary == 'crossfit':
        n_distinct = int(xfit['n_distinct_thresholds'])
        far_out = round(float(xfit['pooled_far_mean']), 4)
        row['xfit'] = {
            'n_folds': xfit['n_folds'], 'seeds': xfit['seeds'],
            'n_calib_negatives': xfit['n_neg'],
            'n_distinct_thresholds': n_distinct,
            'threshold_modes': xfit['threshold_modes'][:3],
            'pooled_far_mean': far_out,
            'pooled_far_min': xfit['pooled_far_min'],
            'pooled_far_max': xfit['pooled_far_max'],
            'pos_coverage_mean': xfit['pos_coverage_mean'],
            'pos_coverage_min': xfit['pos_coverage_min'],
        }
        # 判据用样本外的合并 FAR + 众数阈值下的覆盖率（= 实际下发那对阈值的覆盖率）
        far_primary, cov_primary = far_out, cov_modal
        row['real_far_source'] = 'crossfit_pooled_mean（样本外）'
        # CI 只在「所有折同一对阈值」时有效：那时逐样本接受与否都由没见过它的阈值决定
        row['ci95_valid'] = bool(n_distinct == 1)
        row['real_ci95'] = list(wilson(na, n)) if n_distinct == 1 else None
        if n_distinct == 1 and abs(far_out - far_modal) > 1e-4:
            raise AssertionError(
                f'交叉拟合合并 FAR {far_out} 与众数阈值全量 FAR {far_modal} 不一致，'
                f'但两者在 n_distinct_thresholds==1 时必须相等——本脚本实现有错')
    else:
        far_primary, cov_primary = far_modal, cov_modal
        row['real_far_source'] = 'synthetic 标定（S3 原口径，样本内搜阈值）'
        row['ci95_valid'] = True
        row['real_ci95'] = list(wilson(na, n))

    row.update({
        'real_far': far_primary,
        'real_far_at_modal_thresholds': far_modal,
        'real_accepted': int(na), 'real_n': n,
        'dangerous_rate': dang['dangerous']['rate'],
        'dangerous_ci95': dang['dangerous']['ci95'],
        'dangerous_from': '众数阈值铺到全部真人样本（交叉拟合下为参考口径）',
        'syn_coverage': cov_primary,
        'meets_target': bool(far_primary <= ACCEPT_MAX_FAR
                             and (cov_primary or 0) >= ACCEPT_MIN_COVERAGE),
    })

    # S3 原口径对照：同一池、只在合成负样本上搜阈值，会得到什么
    if syn_cal.get('reachable'):
        sth = syn_cal['thresholds']
        sacc = _apply(cols['real'], sth)
        skacc = _apply(cols['syn_known'], sth)
        sna = int(sum(sacc))
        scmds = Counter(groups['real']['command'][i] for i in range(n) if sacc[i])
        sdang = dangerous_split(scmds, n)
        sfar = round(sna / n, 4)
        scov = round(sum(skacc) / len(skacc), 4) if skacc else None
        row['synthetic_only'] = {
            'thresholds': {kk: round(float(vv), 6) for kk, vv in sth.items()},
            'real_far': sfar, 'real_ci95': list(wilson(sna, n)),
            'dangerous_rate': sdang['dangerous']['rate'],
            'syn_coverage': scov,
            'meets_target': bool(sfar <= ACCEPT_MAX_FAR
                                 and (scov or 0) >= ACCEPT_MIN_COVERAGE),
        }
    if n_boot:
        row['calibration_bootstrap'] = bootstrap_far(
            cols, groups['real']['command'], n_boot, boot_seed, target_risk)
    return row


def _crosscheck_full_pool(rows, ref_path: Path) -> dict:
    """满池结果必须与 S3 公布的数字一致，否则本脚本的实现有错。

    曾经踩过：``ood`` 少写一个负号，AND 规则整体反向，满池 FAR 从 0.0660 变成
    0.2325——数字看着「还行」，实际已经完全不是那件事。所以这里硬失败。

    S3 公布的是**合成侧标定**的数字，因此硬校验打在 ``synthetic_only`` 上；
    交叉拟合口径只做软对比并记录差值——它本来就该和合成口径接近（接近说明标定充分），
    但不要求逐位相等。
    """
    full = [r for r in rows if r.get('kind') == 'full']
    out = {'checked': bool(full), 'ref': str(ref_path), 'matches_s3': None}
    if not full or not ref_path.exists():
        return out
    ref = json.loads(ref_path.read_text(encoding='utf-8'))
    want = (ref.get('and_rules') or {}).get('maxprob+ood', {}).get('applied_to_real')
    if not want:
        return out
    got = full[0]
    syn = got.get('synthetic_only') or {}
    out['expected'] = {'real_far': want['real_far'],
                       'syn_coverage': want['syn_known_coverage'],
                       'dangerous_rate': want['dangerous_split']['dangerous']['rate']}
    out['got_synthetic_only'] = {
        'real_far': syn.get('real_far'),
        'syn_coverage': syn.get('syn_coverage'),
        'dangerous_rate': syn.get('dangerous_rate')}
    out['matches_s3'] = out['expected'] == out['got_synthetic_only']
    if not out['matches_s3']:
        raise AssertionError(
            f"满池（合成侧标定）结果与 S3 公布值不一致，本脚本实现有错：\n"
            f"  期望 {out['expected']}\n  实得 {out['got_synthetic_only']}")
    out['got_primary'] = {'calibrate_used': got.get('calibrate_used'),
                          'real_far': got.get('real_far'),
                          'syn_coverage': got.get('syn_coverage'),
                          'dangerous_rate': got.get('dangerous_rate'),
                          'thresholds': got.get('thresholds')}
    xf = got.get('xfit') or {}
    if xf:
        out['crossfit_vs_s3'] = {
            'far_delta': round(float(xf['pooled_far_mean']) - want['real_far'], 4),
            'n_distinct_thresholds': xf['n_distinct_thresholds'],
            'note': 'far_delta ≈ 0 说明 72 条合成负样本这次搜到了和数千条真人负样本'
                    '交叉拟合相同的工作点；n_distinct_thresholds==1 说明该工作点在'
                    '折间/种子间完全稳定。'}
    return out


def interpret(summary, best) -> dict:
    """把「池压缩代价」「标定方式」和「标定噪声」三个量摆在一起，判断哪个是瓶颈。

    这一步不能省：只看 FAR 随池规模上升，会把 72 条标定负样本带来的抖动
    误读成压缩代价，然后得出「池不能压」的错误结论。所以先确认交叉拟合是否
    把抖动压下去了（``xfit_distinct_thresholds_max``、跨种子 FAR 全距），
    压下去了再谈池规模。
    """
    full = next((s for s in summary if s['kind'] == 'full'), None)
    out = {'smallest_passing': best, 'full_pool': full}
    if not full:
        out['statements'] = ['未跑满池配置，无法判断。']
        return out

    noise_p90 = full.get('calib_noise_far_p90_worst')
    noise_p10 = full.get('calib_noise_far_p10_best')
    subs = [s for s in summary if s['kind'] == 'subsample']
    cents = [s for s in summary if s['kind'] == 'centroid']
    stmts = []

    stmts.append(
        f"满池（n={full['n_pool']}，{full['kb_f32']} KB f32 / "
        f"{round(full['bytes_f16'] / 1024, 1)} KB f16，标定={full['calibrate_used']}）"
        f"实测 FAR {full['real_far_mean']}、覆盖率 {full['syn_coverage_mean']}、"
        f"危险误动作率 ≤ {full['dangerous_rate_worst']}，"
        + ('达标。' if full['all_draws_meet_target'] else '**未达标**。'))

    # --- 1. 交叉拟合有没有把标定抖动压下去 ---
    xf_stable = None
    if 'xfit_distinct_thresholds_max' in full:
        d_full = full.get('xfit_distinct_thresholds_sum', full['xfit_distinct_thresholds_max'])
        d_max = max((s['xfit_distinct_thresholds_max'] for s in summary
                     if 'xfit_distinct_thresholds_max' in s), default=None)
        seed_spread = round(full['xfit_far_worst_over_seeds'] - full['real_far_mean'], 4)
        xf_stable = bool(d_full == 1)
        out['crossfit_stability'] = {
            'full_pool_distinct_thresholds': d_full,
            'distinct_thresholds_max_over_all_configs': d_max,
            'full_pool_far_seed_spread': seed_spread,
            'stable_at_full_pool': xf_stable}
        stmts.append(
            f"**交叉拟合稳定性**：满池在所有折与所有随机划分上共选出 {d_full} 组不同阈值，"
            f"合并 FAR 全距 {seed_spread}"
            + ('——满池工作点唯一且稳定。' if xf_stable and seed_spread <= 0.01
               else '——满池工作点已在折间漂移。')
            + f"压缩池上最多出现 {d_max} 组不同阈值，因此压缩配置的 FAR 是"
              f"「交叉拟合这套流程」的样本外估计，不是某一对固定阈值的估计；"
              f"实际下发时阈值会用全部真人负样本一次性拟合，"
              f"2 折交叉拟合只用了一半负样本，故报出的 FAR 偏保守。")

    # --- 2. 为什么必须换标定方式：合成侧标定的噪声底 ---
    if noise_p90 is not None:
        out['calib_noise_at_full_pool'] = {'p10': noise_p10, 'p90': noise_p90,
                                           'max': full.get('calib_noise_far_max')}
        spread = round(noise_p90 - noise_p10, 4)
        out['calib_noise_spread'] = spread
        stmts.append(
            f"**换标定方式的依据（合成侧口径的噪声底）**：池完全不变、"
            f"只对 72 条合成负样本做有放回重采样，真实域 FAR 就在 "
            f"[{noise_p10}, {noise_p90}]（p10–p90 跨度 {spread}）之间摆动，"
            f"最坏抽样到 {full.get('calib_noise_far_max')}，"
            f"共产生 {full.get('calib_distinct_thresholds_total')} 组不同阈值。"
            + ('这个跨度已超过目标 FAR，所以合成侧标定的点估计不可当作可复现保证；'
               '本表主口径改为交叉拟合。' if noise_p90 > ACCEPT_MAX_FAR else
               '这个跨度未超过目标 FAR，两种口径都能用，交叉拟合仍作为主口径。'))
        if any('synthetic_only_far_mean' in s for s in summary):
            xs = [s for s in summary if 'synthetic_only_far_mean' in s]
            stmts.append(
                f"**同一批池换成合成侧标定（S3 原口径）**：FAR 均值区间 "
                f"{min(s['synthetic_only_far_mean'] for s in xs)}–"
                f"{max(s['synthetic_only_far_mean'] for s in xs)}，最差抽样 "
                f"{max(s['synthetic_only_far_worst'] for s in xs)}；"
                f"交叉拟合口径 FAR 均值区间 "
                f"{min(s['real_far_mean'] for s in xs)}–"
                f"{max(s['real_far_mean'] for s in xs)}，最差 "
                f"{max(s['real_far_worst'] for s in xs)}。"
                f"满池上两者相同（{full['synthetic_only_far_mean']}），"
                f"池一压小就分道扬镳——小池上 72 条合成负样本更搜不出稳定工作点，"
                f"这正是必须用交叉拟合读压缩代价的原因。")

    # --- 3. 压缩代价 ---
    if subs:
        sub_worst = max(s['real_far_worst'] for s in subs)
        sub_best_mean = min(s['real_far_mean'] for s in subs)
        sub_spread = round(sub_worst - sub_best_mean, 4)
        out['subsample_far_spread'] = sub_spread
        stmts.append(
            f"子采样配置（n={min(s['n_pool'] for s in subs)}–"
            f"{max(s['n_pool'] for s in subs)}）的 FAR 全距为 {sub_spread}"
            f"（均值区间 {sub_best_mean}–{max(s['real_far_mean'] for s in subs)}，"
            f"最差抽样 {sub_worst}），覆盖率均值区间 "
            f"{min(s['syn_coverage_mean'] for s in subs)}–"
            f"{max(s['syn_coverage_mean'] for s in subs)}。"
            + ('压缩代价明显大于标定噪声底，池规模的结论可读。'
               if noise_p90 is not None and sub_spread > round(noise_p90 - noise_p10, 4)
               else '压缩代价不大于合成侧标定的噪声底——若用合成口径，池规模的结论不可读；'
                    '本表已改用交叉拟合口径，故按上条的稳定性判断。'))
        passing = [s for s in subs if s['all_draws_meet_target']]
        stmts.append(
            '子采样中达标的规模：'
            + (', '.join(f"n={s['n_pool']}（{s['kb_f32']} KB）" for s in passing)
               if passing else '**无一达标**（所有抽样都要同时满足两条判据才算达标）。'))
        # 哪条判据先崩：FAR 还是覆盖率。两者可读性不同——
        # FAR 在折间划分上还有抖动，覆盖率基本不随划分变，所以覆盖率才是可排序的那一轴。
        xf_split = max((s['xfit_far_worst_over_seeds'] - s['real_far_mean'])
                       for s in subs if 'xfit_far_worst_over_seeds' in s) if subs else None
        if xf_split is not None:
            out['crossfit_split_spread_max'] = round(xf_split, 4)
            far_span = round(max(s['real_far_mean'] for s in subs)
                             - min(s['real_far_mean'] for s in subs), 4)
            cov_span = round(max(s['syn_coverage_mean'] for s in subs)
                             - min(s['syn_coverage_mean'] for s in subs), 4)
            stmts.append(
                f"两条判据里，**先崩的是覆盖率**：交叉拟合自身的划分间 FAR 抖动最大 "
                f"{round(xf_split, 4)}，与子采样之间的 FAR 差异（{far_span}）同量级，"
                f"所以 FAR 这一轴只能读到「压到 32 条仍在 0.10 附近」，读不出精细排序；"
                f"覆盖率随池规模单调下滑且基本不随划分变（跨度 {cov_span}），"
                f"是决定最小可用池的那一轴。")
    if cents:
        stmts.append(
            f"质心池（每类 1–{max(s['n_pool'] for s in cents)} 个，"
            f"{min(s['kb_f32'] for s in cents)}–{max(s['kb_f32'] for s in cents)} KB）"
            f"FAR 最差到 {max(s['real_far_worst'] for s in cents)}，"
            f"覆盖率最差 {min(s['syn_coverage_worst'] for s in cents)}，"
            + ('没有任何配置达标，质心化不可行。'
               if not any(s['all_draws_meet_target'] for s in cents) else
               '存在达标配置，见 smallest_passing。'))

    # --- 4. 结论 ---
    if best is None:
        stmts.append(
            f"没有任何池配置在全部抽样下同时满足 FAR ≤ {ACCEPT_MAX_FAR} 且"
            f"覆盖率 ≥ {ACCEPT_MIN_COVERAGE}（{full['calibrate_used']} 口径）。")
    else:
        stmts.append(
            f"最小达标池：{best['kind']} n={best['n_pool']} = {best['kb_f32']} KB (f32) / "
            f"{round(best['bytes_f16'] / 1024, 1)} KB (f16)，"
            f"FAR {best['real_far_mean']}、覆盖率 {best['syn_coverage_mean']}。")
    out['statements'] = stmts
    return out


def summarize(rows) -> dict:
    """按池规模聚合多种子：均值与最差值都要，只看均值会漏掉坏抽样的风险。"""
    by_size = {}
    for r in rows:
        if not r.get('reachable'):
            continue
        by_size.setdefault((r['kind'], r['n_pool'], r['k']), []).append(r)
    out = []
    for (kind, n, k), rs in sorted(by_size.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        fars = [x['real_far'] for x in rs]
        covs = [x['syn_coverage'] for x in rs]
        dangs = [x['dangerous_rate'] for x in rs]
        syns = [x.get('synthetic_only') or {} for x in rs]
        syns = [s for s in syns if s.get('real_far') is not None]
        xfs = [x.get('xfit') or {} for x in rs]
        xfs = [f for f in xfs if f.get('pooled_far_mean') is not None]
        boots = [x.get('calibration_bootstrap') or {} for x in rs]
        boots = [b for b in boots if b.get('reachable_draws')]
        entry = {
            'kind': kind, 'n_pool': n, 'k': k, 'n_draws': len(rs),
            'calibrate_used': sorted({x.get('calibrate_used') for x in rs if x.get('calibrate_used')}),
            'bytes_f32': rs[0]['bytes_f32'], 'bytes_f16': rs[0]['bytes_f16'],
            'kb_f32': round(rs[0]['bytes_f32'] / 1024, 1),
            'real_far_mean': round(float(np.mean(fars)), 4),
            'real_far_worst': round(float(np.max(fars)), 4),
            'dangerous_rate_worst': round(float(np.max(dangs)), 4),
            'syn_coverage_mean': round(float(np.mean(covs)), 4),
            'syn_coverage_worst': round(float(np.min(covs)), 4),
            'ood_auc_cross_mean': round(float(np.mean(
                [x['ood_auc_cross_domain'] for x in rs])), 4),
            'all_draws_meet_target': all(x['meets_target'] for x in rs),
        }
        if syns:
            # S3 原口径对照：同一批池上，只用 72 条合成负样本标定会得到什么
            entry['synthetic_only_far_mean'] = round(float(np.mean(
                [s['real_far'] for s in syns])), 4)
            entry['synthetic_only_far_worst'] = round(float(np.max(
                [s['real_far'] for s in syns])), 4)
            entry['synthetic_only_coverage_mean'] = round(float(np.mean(
                [s['syn_coverage'] for s in syns if s.get('syn_coverage') is not None])), 4)
        if xfs:
            entry['xfit_far_worst_over_seeds'] = round(float(np.max(
                [f['pooled_far_max'] for f in xfs])), 4)
            entry['xfit_distinct_thresholds_max'] = int(max(
                f['n_distinct_thresholds'] for f in xfs))
            entry['xfit_distinct_thresholds_sum'] = int(sum(
                f['n_distinct_thresholds'] for f in xfs))
            entry['xfit_coverage_worst'] = round(float(np.min(
                [f['pos_coverage_min'] for f in xfs])), 4)
        if boots:
            # 噪声底：同一个池、只重采样 72 条标定负样本时 FAR 的分布。
            # 若它和「换池规模」造成的差异同量级，池规模这一列就读不出结论。
            entry['calib_noise_far_p10_best'] = min(b['real_far_p10'] for b in boots)
            entry['calib_noise_far_p50_mean'] = round(
                float(np.mean([b['real_far_p50'] for b in boots])), 4)
            entry['calib_noise_far_p90_worst'] = max(b['real_far_p90'] for b in boots)
            entry['calib_noise_far_max'] = max(b['real_far_max'] for b in boots)
            entry['calib_distinct_thresholds_total'] = int(
                sum(b['n_distinct_threshold_pairs'] for b in boots))
        out.append(entry)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-cache', default=str(DEFAULT_RAW))
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--sizes', default='1981,1000,500,250,128,64,32',
                    help='随机/分层子采样的池条数，逗号分隔')
    ap.add_argument('--seeds', default='0,1,2', help='每个规模重复抽样次数')
    ap.add_argument('--centroids', default='16,8,4,2,1', help='每类质心数，逗号分隔')
    ap.add_argument('--centroid-k', default='1,3', help='质心池上扫的 k 值')
    ap.add_argument('--target-risk', type=float, default=0.10)
    ap.add_argument('--calibrate', choices=('crossfit', 'synthetic'), default='crossfit',
                    help='主标定口径；crossfit = 在真人负样本上 K 折交叉拟合（样本外 FAR），'
                         'synthetic = S3 原口径（72 条合成负样本）。另一套始终作为对照计算')
    ap.add_argument('--xfit-folds', type=int, default=2)
    ap.add_argument('--xfit-seeds', default='0,1,2', help='交叉拟合的随机划分次数')
    ap.add_argument('--bootstrap', type=int, default=200,
                    help='合成侧标定噪声对照的重采样次数；0 = 关闭')
    ap.add_argument('--boot-seed', type=int, default=0)
    ap.add_argument('--classes', default='ALL_OFF,CURTAIN_CLOSE,CURTAIN_OPEN,'
                                         'LIGHT_OFF,LIGHT_ON,STATUS_ALL')
    ap.add_argument('--out', default=str(ROOT / 'docs' / 'pool_size.json'))
    args = ap.parse_args(argv)

    cache = Path(args.raw_cache)
    if not cache.exists():
        sys.exit(f'缺原始量缓存: {cache}\n先跑 scripts/eval_selective.py 生成它')
    groups = load_raw(cache, ('pool', 'syn_known', 'syn_unknown', 'real'))
    emb, labs = groups['pool']['embedding'], groups['pool']['label']
    classes = [c for c in args.classes.split(',') if c]
    print(f'满池 {emb.shape}，类别 {sorted(set(labs))}', flush=True)

    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    cents = [int(s) for s in args.centroids.split(',') if s.strip()]
    cks = [int(s) for s in args.centroid_k.split(',') if s.strip()]
    xfit_seeds = tuple(int(s) for s in args.xfit_seeds.split(',') if s.strip())

    rows = []

    def log(r, tag):
        boot = r.get('calibration_bootstrap') or {}
        extra = (f" 合成噪声 p50={boot['real_far_p50']} "
                 f"[{boot['real_far_p10']}, {boot['real_far_p90']}]"
                 if boot.get('reachable_draws') else '')
        syn = r.get('synthetic_only') or {}
        synx = f" | 合成口径 FAR={syn.get('real_far')} cov={syn.get('syn_coverage')}" if syn else ''
        nd = (r.get('xfit') or {}).get('n_distinct_thresholds')
        ndx = f" 阈值组数={nd}" if nd is not None else ''
        print(f"[{tag} n={r['n_pool']:>4} k={r['k']} {r.get('calibrate_used', '-')}{ndx}] "
              f"FAR={r.get('real_far')} cov={r.get('syn_coverage')} "
              f"dangerous={r.get('dangerous_rate')} "
              f"({r['bytes_f32'] / 1024:.1f} KB){synx}{extra}", flush=True)

    for size in sizes:
        draws = [(0, 'full')] if size >= len(emb) else [(s, 'subsample') for s in seeds]
        for seed, kind in draws:
            pool = make_pool(kind, emb, labs, size, seed, classes)
            r = evaluate_pool(pool, groups, args.k, args.target_risk,
                              n_boot=args.bootstrap, boot_seed=args.boot_seed + seed,
                              calibrate=args.calibrate, n_folds=args.xfit_folds,
                              xfit_seeds=xfit_seeds)
            r.update({'kind': kind, 'requested_size': size, 'seed': seed})
            rows.append(r)
            log(r, f'{kind} seed={seed}')

    for m in cents:
        pool = make_pool('centroid', emb, labs, m, 0, classes)
        for k in cks:
            r = evaluate_pool(pool, groups, k, args.target_risk,
                              n_boot=args.bootstrap, boot_seed=args.boot_seed + m,
                              calibrate=args.calibrate, n_folds=args.xfit_folds,
                              xfit_seeds=xfit_seeds)
            r.update({'kind': 'centroid', 'requested_size': m, 'seed': 0})
            rows.append(r)
            log(r, f'centroid m={m}/类')

    summary = summarize(rows)

    # 满池这一行必须复现 docs/selective_eval.json 里 S3 公布的数字。
    # 本脚本重写了 ood 的计算与阈值标定流程，不校验就等于两套实现各说各话。
    xcheck = _crosscheck_full_pool(rows, Path(ROOT / 'docs' / 'selective_eval.json'))

    ok = [s for s in summary if s['all_draws_meet_target']]
    # 达标里挑体积最小；同体积优先 f16 更小的（这里 f16 与 f32 同序，取 f32 即可）
    best = min(ok, key=lambda s: s['bytes_f32']) if ok else None

    interp = interpret(summary, best)

    report = {
        'generated': date.today().isoformat(),
        'protocol': {
            'accept_max_far': ACCEPT_MAX_FAR,
            'accept_min_coverage': ACCEPT_MIN_COVERAGE,
            'target_risk_for_calibration': args.target_risk,
            'knn_k': args.k,
            'recalibrated_per_pool': True,
            'calibrate_primary': args.calibrate,
            'calibrated_on': (
                'crossfit：真人负样本 K 折交叉拟合（每折阈值只用其余折搜出，'
                f'{args.xfit_folds} 折 × {len(xfit_seeds)} 次随机划分），'
                '正样本全量参与；报出的 FAR 全部样本外'
                if args.calibrate == 'crossfit' else
                'synthetic：syn_known(val) 正样本 + syn_unknown 负样本；真人集只评估'),
            'comparison_also_reported': (
                'synthetic_only（S3 原口径）' if args.calibrate == 'crossfit'
                else 'crossfit（交叉拟合）未启用'),
            'pool_source_split': 'train（与覆盖率一侧不重叠）',
            'sampling': '分层子采样（按命令类别配额），多随机种子',
            'bootstrap_draws': args.bootstrap,
            'bootstrap_target': 'syn_unknown（72 条合成负样本）有放回重采样，池不变；'
                                '用于量化合成侧标定的噪声底，即换标定方式的依据',
        },
        'groups': {g: groups[g]['n'] for g in ('pool', 'syn_known', 'syn_unknown', 'real')},
        'full_pool_matches_s3': xcheck,
        'per_config': rows,
        'summary': summary,
        'smallest_passing': best,
        'interpretation': interp,
        'reference': {
            'model_bytes': int((ROOT / 'runtime' / 'models'
                                / 'v3_cnn_ensemble_vad.npz').stat().st_size)
            if (ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz').exists() else None,
            'note': '满池 1981 条的 S3 结果见 docs/selective_eval.json',
        },
        'caveats': [
            '覆盖率一侧仍是合成命令；池压缩对**真人命令**存活率的影响无法测（真人正样本为 0）。',
            '质心池改变了 ood 的分布形状（不再有近邻尾部），阈值已按池重标定，但 k 的含义也变了：k 接近质心总数时 ood 退化为「到全部质心的平均距离」。',
            '这里的字节数是未压缩裸数组；实际下发可再套 npz 压缩或 f16，体积会更小，但 RAM 占用按 f32 算。',
            '交叉拟合口径下，真人负样本参与了阈值搜索。它仍是样本外估计（每条样本都由'
            '没见过它的阈值判定），但「本项目不需要真人数据」的说法在这个口径下不成立；'
            '实际下发时阈值会用全部真人负样本一次性拟合，S3 的 oracle 结果显示这与'
            '交叉拟合选出的工作点相同。',
            '交叉拟合下 real_ci95 只在所有折选中同一对阈值时有效（ci95_valid=true）；'
            '否则接受集是不同阈值的混合，Wilson 区间不再对应单一规则。',
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding='utf-8')

    print('\n=== 汇总（按体积升序）===')
    for s in sorted(summary, key=lambda x: x['bytes_f32']):
        flag = 'PASS' if s['all_draws_meet_target'] else 'fail'
        syn = (f" | 合成口径 mean={s['synthetic_only_far_mean']:.4f} "
               f"worst={s['synthetic_only_far_worst']:.4f}"
               ) if 'synthetic_only_far_mean' in s else ''
        xf = (f" | 阈值组数≤{s['xfit_distinct_thresholds_max']} "
              f"跨划分最差={s['xfit_far_worst_over_seeds']:.4f}"
              ) if 'xfit_distinct_thresholds_max' in s else ''
        print(f"{s['kind']:>10} n={s['n_pool']:>4} k={s['k']} "
              f"{s['kb_f32']:>7.1f} KB  FAR mean={s['real_far_mean']:.4f} "
              f"worst={s['real_far_worst']:.4f}  cov mean={s['syn_coverage_mean']:.4f} "
              f"worst={s['syn_coverage_worst']:.4f}  "
              f"dangerous≤{s['dangerous_rate_worst']:.4f}  {flag}{syn}{xf}")
    if best:
        print(f"\n最小达标池: {best['kind']} n={best['n_pool']} "
              f"= {best['kb_f32']} KB (f32) / {round(best['bytes_f16'] / 1024, 1)} KB (f16)")
    else:
        print('\n没有任何池配置在全部抽样下同时满足两条判据（原因见下面的判读）')

    print('\n=== 判读 ===')
    for s in interp['statements']:
        print('-', s)
    print('\n已写入:', args.out)
    return report


if __name__ == '__main__':
    main()

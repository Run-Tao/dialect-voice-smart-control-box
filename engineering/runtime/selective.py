# -*- coding: utf-8 -*-
"""选择性分类（selective classification）：多信号拒识、风险-覆盖工作点与诊断指标。

为什么需要它（S0 实测结论）
--------------------------
部署的单一 max-prob 阈值在真人粤语非命令语音上 FAR = 0.3110，且置信度
p90 = 0.9923、max = 1.0——约 10% 的域外负样本被 softmax **满信心**接受。
阈值反解显示 FAR 0.15 → 0.10 之间合成覆盖率从 0.6159 断崖到 0.0023。
结论：softmax 在域外饱和，**再抬阈值也换不来 FAR 下降**，必须引入不依赖
softmax 标定质量的第二信号。

本模块提供
----------
1. 信号计算：max-prob / margin / 负熵 / 集成一致性 / 嵌入 kNN 距离（OOD）；
2. 组合规则：min-z（等价于 AND 规则）、z 加和、双阈值 AND 网格搜索——
   形式全部**预先指定**，参数只在合成数据上标定，不在真实测试集上拟合；
3. 风险-覆盖：按阈值对齐两条曲线，反解满足目标 FAR 的最大覆盖率工作点；
4. 诊断：AUC、ECE，以及「这个信号到底在判别命令，还是只在判别 TTS 与真人」。

方向约定
--------
``ACCEPT_SIGNALS`` 里的每个信号都满足**越大越像真命令**，判据统一为
``score >= threshold``。``DIAGNOSTIC_SIGNALS`` 是原始量，只用于报告与排查，
不能直接当接受分数（方向或量纲不对）。

一个必须说清的口径问题
----------------------
本项目的正样本（真命令）只有合成 TTS，负样本有两套：合成非命令（同域）与
真人非命令（跨域）。任何能区分「TTS 与真人」的信号，在跨域配对上都会得到
漂亮的曲线，却完全不能说明它能拒掉**真人说的非命令**。所以评估必须同时给
同域 AUC 与跨域 AUC：跨域远高于同域，就说明该信号主要是域检测器。
``domain_detector_gap`` 就是为这件事准备的。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ACCEPT_SIGNALS = ('maxprob', 'margin', 'neg_entropy', 'seed_agree', 'ood')
DIAGNOSTIC_SIGNALS = ('seed_std', 'knn_dist', 'entropy')
LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# 信号计算
# ---------------------------------------------------------------------------

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 散度（自然对数，值域 [0, ln2]）。对称且有界，比 KL 稳。"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def seed_std(seed_probs: np.ndarray) -> float:
    """各种子 top-1 概率的标准差。集成越分歧，值越大。"""
    sp = np.asarray(seed_probs, dtype=np.float64)
    if sp.ndim != 2 or sp.shape[0] < 2:
        return 0.0
    return float(sp.max(axis=1).std())


def seed_agreement(seed_probs: np.ndarray) -> float:
    """1 - 平均两两 JS / ln2，归一到 [0,1]；1 = 完全一致。单种子时为 1.0。"""
    sp = np.asarray(seed_probs, dtype=np.float64)
    if sp.ndim != 2 or sp.shape[0] < 2:
        return 1.0
    vals = [js_divergence(sp[i], sp[j])
            for i in range(sp.shape[0]) for j in range(i + 1, sp.shape[0])]
    return float(1.0 - (sum(vals) / len(vals)) / LN2)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def build_pool(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """参照池：L2 归一化的命令嵌入矩阵 [N,D]，供 kNN 距离使用。"""
    if len(embeddings) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return l2_normalize(np.stack([np.asarray(e, dtype=np.float64).ravel()
                                  for e in embeddings], axis=0))


def knn_distance(embedding: np.ndarray, pool: np.ndarray, k: int = 5) -> Optional[float]:
    """到参照池 k 近邻的平均余弦距离 [0,2]。域外输入离命令流形远 → 值大。"""
    if pool is None or pool.size == 0 or pool.shape[0] == 0:
        return None
    e = l2_normalize(np.asarray(embedding, dtype=np.float64).ravel())
    d = 1.0 - pool @ e
    k = max(1, min(int(k), d.size))
    return float(np.partition(d, k - 1)[:k].mean())


def entropy(prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, None)
    return float(-np.sum(p * np.log(p)))


def signals(prob: np.ndarray, seed_probs: np.ndarray,
            embedding: Optional[np.ndarray] = None,
            pool: Optional[np.ndarray] = None, k: int = 5) -> Dict[str, Optional[float]]:
    """从一次前向的输出算全部信号。

    ``prob`` 是部署概率（各种子 logits 平均后 softmax），``seed_probs`` 是逐种子
    概率 [S,C]。``pool`` 为空时 ``ood`` / ``knn_dist`` 返回 None，调用方需过滤。
    """
    p = np.asarray(prob, dtype=np.float64)
    p = p / p.sum()
    order = np.argsort(-p)
    top = float(p[order[0]])
    second = float(p[order[1]]) if p.size > 1 else 0.0
    h = entropy(p)
    dist = knn_distance(embedding, pool, k) if embedding is not None else None
    return {
        'maxprob': top,
        'margin': top - second,
        'entropy': h,
        'neg_entropy': -h,
        'seed_std': seed_std(seed_probs),
        'seed_agree': seed_agreement(seed_probs),
        'knn_dist': dist,
        'ood': None if dist is None else -dist,
    }


# ---------------------------------------------------------------------------
# 组合规则（形式预先指定，参数只在合成数据上标定）
# ---------------------------------------------------------------------------

def zstats(values: Sequence[float]) -> Dict[str, float]:
    """z 分数的标定统计量。必须来自**部署可见**的数据（合成已知命令），
    不能用真实测试集，否则工作点就是照着测试集调出来的。"""
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if a.size == 0:
        return {'mean': 0.0, 'std': 1.0, 'n': 0}
    return {'mean': float(a.mean()), 'std': float(a.std()) or 1.0, 'n': int(a.size)}


def zapply(value: Optional[float], stats: Dict[str, float]) -> Optional[float]:
    if value is None:
        return None
    return (float(value) - stats['mean']) / max(stats['std'], 1e-12)


def _zvalues(row: Dict[str, Optional[float]],
             stats: Dict[str, Dict[str, float]]) -> Optional[List[float]]:
    """任一信号缺失就返回 None：宁可拒识，也不要拿半个规则打分。

    跳过缺失信号会让「参照池没建好」静默退化成纯 max-prob 规则，
    评估数字看着正常，实际少了第二信号——这种失效必须显式暴露。
    """
    zs = []
    for name in stats:
        v = row.get(name)
        if v is None:
            return None
        zs.append(zapply(v, stats[name]))
    return zs or None


def combo_min_z(row: Dict[str, Optional[float]],
                stats: Dict[str, Dict[str, float]]) -> Optional[float]:
    """min-z 组合：取各信号 z 分数的最小值。

    对它扫单一阈值**等价于**「每个信号都要过同一个 z 门槛」的 AND 规则，
    因此能直接画风险-覆盖曲线，而双阈值 AND 只能网格搜工作点。
    """
    zs = _zvalues(row, stats)
    return None if zs is None else float(min(zs))


def combo_sum_z(row: Dict[str, Optional[float]],
                stats: Dict[str, Dict[str, float]]) -> Optional[float]:
    """z 分数加和（等权线性规则）。一个信号极高可以补偿另一个极低。"""
    zs = _zvalues(row, stats)
    return None if zs is None else float(sum(zs))


def and_workpoint(neg: Dict[str, Sequence[float]], pos: Dict[str, Sequence[float]],
                  names: Sequence[str], target_risk: float,
                  n_grid: int = 40) -> Dict[str, object]:
    """双阈值 AND 规则网格搜索：接受当且仅当所有信号都过各自的阈值。

    在满足 FAR <= target_risk 的组合里取覆盖率最高的那一组。候选阈值取正样本
    分位数网格（覆盖率只在正样本取值处跳变），共 n_grid**len(names) 个组合。

    风险与覆盖率都按**逐样本 AND** 精确计算。用「各信号单独接受率的 min」当
    近似是错的：它对风险是上界（会漏掉可行组合），对覆盖率也是上界（会报出
    实际达不到的覆盖率），两个方向都不可靠。

    返回的 ``thresholds`` 已按 6 位小数取整（下发精度），``coverage``/``risk``
    就是用这组取整后的阈值算出来的，三者指同一条规则。
    """
    names = list(names)
    if not names:
        raise ValueError('and_workpoint 需要至少一个信号')

    vals, grids, neg_ok, pos_ok = [], [], [], []
    for n in names:
        nv = np.asarray([x for x in neg[n] if x is not None], dtype=np.float64)
        pv = np.asarray([x for x in pos[n] if x is not None], dtype=np.float64)
        if nv.size == 0 or pv.size == 0:
            return {'reachable': False, 'target_risk': target_risk,
                    'reason': f'信号 {n} 缺失或某一侧样本为空'}
        # 网格先按下发精度（6 位小数）**向下取整**再搜。阈值最终要写进 JSON 给端侧用，
        # 而取整会让边界样本翻转：实测四舍五入时最小那条正样本会落到阈值之下，
        # 未取整网格报的 coverage 1.0 与下发的阈值不是同一条规则（真实 0.99）。
        # 向下取整对 ``score >= threshold`` 是统一的「更宽松」方向（负阈值同样成立），
        # 所以覆盖率不会比未取整时更差，而风险是按取整后的值算的，目标约束仍然真实。
        # 和 f16 池必须先往返再标定是同一件事：标定要发生在**实际下发的那份数值**上。
        g = np.unique(np.floor(np.quantile(pv, np.linspace(0.0, 1.0, n_grid)) * 1e6) / 1e6)
        vals.append((nv.size, pv.size))
        grids.append(g)
        neg_ok.append(nv[None, :] >= g[:, None])
        pos_ok.append(pv[None, :] >= g[:, None])

    n_neg, n_pos = vals[0]
    if any(v != vals[0] for v in vals):
        return {'reachable': False, 'target_risk': target_risk,
                'reason': '各信号的样本数不一致，无法逐样本 AND'}

    best = None
    idx = np.array(np.meshgrid(*[np.arange(len(g)) for g in grids],
                               indexing='ij')).reshape(len(names), -1).T
    for combo in idx:
        acc_neg = neg_ok[0][combo[0]]
        acc_pos = pos_ok[0][combo[0]]
        for i in range(1, len(names)):
            acc_neg = acc_neg & neg_ok[i][combo[i]]
            acc_pos = acc_pos & pos_ok[i][combo[i]]
        risk = float(acc_neg.sum()) / n_neg
        if risk > target_risk + 1e-12:
            continue
        cov = float(acc_pos.sum()) / n_pos
        # 网格遍历顺序是任意的，同覆盖率必须显式按更低风险打破平局，
        # 否则报出来的工作点取决于 meshgrid 的排列。
        # best 里存**未取整**的值：拿取整后的 coverage 去和未取整的 cov 比，
        # 1e-12 的平局容差会被四舍五入的噪声吃掉。
        if (best is None or cov > best[0] + 1e-12
                or (abs(cov - best[0]) <= 1e-12 and risk < best[1] - 1e-12)):
            best = (cov, risk, {n: round(float(grids[i][combo[i]]), 6)
                                for i, n in enumerate(names)})
    if best is None:
        return {'reachable': False, 'target_risk': target_risk,
                'reason': '网格内没有任何组合能把 FAR 压到目标以下'}
    cov, risk, ths = best
    return {'reachable': True, 'target_risk': target_risk,
            'n_pos': n_pos, 'n_neg': n_neg, 'n_grid': n_grid,
            'thresholds': ths, 'coverage': round(cov, 4), 'risk': round(risk, 4)}


def and_accept(row: Dict[str, Optional[float]], thresholds: Dict[str, float]) -> bool:
    """按 and_workpoint 给出的阈值判定；任一信号缺失即拒识（保守）。"""
    for name, tau in thresholds.items():
        v = row.get(name)
        if v is None or float(v) < float(tau):
            return False
    return True


def crossfit_and_rule(neg: Dict[str, Sequence[float]],
                      pos: Dict[str, Sequence[float]],
                      names: Sequence[str], target_risk: float,
                      n_folds: int = 2, seeds: Sequence[int] = (0,),
                      n_grid: int = 40) -> Dict[str, object]:
    """在负样本上做 K 折交叉拟合：每折的阈值只用**其余折**搜出，只在本折上评估。

    为什么必须有这个：只在 72 条合成负样本上标定 AND 规则时，bootstrap 重采样
    显示真实域 FAR 会在 [0.066, 0.20]（p10–p90）之间摆动——那个点估计不是可复现
    的保证，瓶颈是标定集太小，不是信号选得不好。交叉拟合把标定集扩到真人负样本
    规模（数千条），同时**仍然保留独立评估**：报出的每一条 FAR 都来自阈值没见过
    的折，所以它既不是 oracle（在评估集上搜阈值，不可部署），也不是自欺
    （在同一批数据上搜又在其上报）。

    与 oracle 的区别必须讲清楚：``and_workpoint(neg=real_all, ...)`` 是能力上界，
    本函数是**可部署规则的无偏估计**。两者数字接近说明标定充分，差得远说明过拟合。

    正样本（覆盖率一侧）始终全量参与标定，因为覆盖率不是要防的错——
    少一条正样本只会让阈值偏保守。
    """
    names = list(names)
    n_neg = len(neg[names[0]])
    n_pos = len(pos[names[0]])
    if n_neg < n_folds * 2:
        return {'reachable': False, 'reason': f'负样本 {n_neg} 条不足以做 {n_folds} 折',
                'n_neg': n_neg, 'n_pos': n_pos, 'target_risk': target_risk}

    folds_out = []
    pooled = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(n_neg)
        chunks = np.array_split(order, n_folds)
        accepted_total = 0
        for j, hold in enumerate(chunks):
            calib = np.concatenate([c for i, c in enumerate(chunks) if i != j])
            cal = and_workpoint(
                {n: [neg[n][i] for i in calib] for n in names}, pos, names,
                target_risk=target_risk, n_grid=n_grid)
            if not cal.get('reachable'):
                folds_out.append({'seed': int(seed), 'fold': j, 'reachable': False,
                                  'reason': cal.get('reason'),
                                  'n_calib': int(calib.size), 'n_eval': int(hold.size)})
                continue
            th = cal['thresholds']
            ev = [and_accept({n: neg[n][i] for n in names}, th) for i in hold]
            na = int(sum(ev))
            accepted_total += na
            cov = [and_accept({n: pos[n][i] for n in names}, th) for i in range(n_pos)]
            folds_out.append({
                'seed': int(seed), 'fold': j, 'reachable': True,
                'n_calib': int(calib.size), 'n_eval': int(hold.size),
                'thresholds': th,
                'eval_accepted': na,
                'eval_far': round(na / hold.size, 4) if hold.size else None,
                'pos_coverage': round(sum(cov) / n_pos, 4) if n_pos else None,
                'calib_risk': cal.get('risk'),
            })
        pooled.append({'seed': int(seed), 'n_neg': n_neg,
                       'accepted': accepted_total,
                       'far': round(accepted_total / n_neg, 4)})

    ok = [f for f in folds_out if f.get('reachable')]
    if not ok:
        return {'reachable': False, 'reason': '所有折都搜不出满足目标风险的阈值',
                'n_neg': n_neg, 'n_pos': n_pos, 'target_risk': target_risk,
                'folds': folds_out, 'pooled': pooled}

    th_keys = [tuple(sorted(f['thresholds'].items())) for f in ok]
    uniq = Counter(th_keys)
    fars = [p['far'] for p in pooled]
    return {
        'reachable': True,
        'target_risk': target_risk,
        'n_folds': n_folds, 'seeds': list(seeds),
        'n_neg': n_neg, 'n_pos': n_pos, 'n_grid': n_grid,
        'folds': folds_out,
        'pooled': pooled,
        # 跨折/跨种子的阈值稳定性：只有 1 组说明工作点很稳，多组说明还在抖
        'n_distinct_thresholds': len(uniq),
        'threshold_modes': [{'thresholds': dict(t), 'count': c}
                            for t, c in uniq.most_common(5)],
        'pooled_far_mean': round(float(np.mean(fars)), 4),
        'pooled_far_min': round(float(np.min(fars)), 4),
        'pooled_far_max': round(float(np.max(fars)), 4),
        'pos_coverage_mean': round(float(np.mean(
            [f['pos_coverage'] for f in ok if f.get('pos_coverage') is not None])), 4),
        'pos_coverage_min': round(float(np.min(
            [f['pos_coverage'] for f in ok if f.get('pos_coverage') is not None])), 4),
    }


# ---------------------------------------------------------------------------
# 风险-覆盖与工作点反解
# ---------------------------------------------------------------------------

def accept_rate(sorted_asc: np.ndarray, tau: float) -> float:
    """score >= tau 的比例。sorted_asc 升序，用 searchsorted 避免 O(n) 扫描。"""
    if sorted_asc.size == 0:
        return 0.0
    idx = int(np.searchsorted(sorted_asc, tau, side='left'))
    return (sorted_asc.size - idx) / sorted_asc.size


def risk_coverage(neg_scores: Sequence[float], pos_scores: Sequence[float],
                  n_points: int = 60) -> Optional[Dict[str, object]]:
    """风险-覆盖曲线：覆盖率取正样本接受率，风险取负样本接受率（= FAR）。

    两侧可以来自不同数据集（例如正=合成已知命令、负=真人非命令），此时它是
    **按阈值对齐的跨域曲线**，不是同一批数据上的标准选择性分类曲线，报告里
    必须写明来源，否则会被当成标准 risk-coverage 误读。
    """
    neg = np.asarray([v for v in neg_scores if v is not None], dtype=np.float64)
    pos = np.asarray([v for v in pos_scores if v is not None], dtype=np.float64)
    if neg.size == 0 or pos.size == 0:
        return None
    lo = float(min(neg.min(), pos.min()))
    hi = float(max(neg.max(), pos.max()))
    taus = np.unique(np.concatenate([np.linspace(lo, hi, n_points), [hi + 1e-6]]))
    neg_s, pos_s = np.sort(neg), np.sort(pos)
    rows = [{'threshold': round(float(t), 6),
             'coverage': round(accept_rate(pos_s, t), 4),
             'risk': round(accept_rate(neg_s, t), 4)} for t in taus]
    return {'curve': rows, 'n_pos': int(pos.size), 'n_neg': int(neg.size)}


def aurc(rc: Optional[Dict[str, object]], levels: Sequence[float]) -> Optional[float]:
    """各覆盖率档位上的平均风险（Area Under Risk-Coverage curve）。

    档位工作点取「coverage <= level、覆盖率最高、且同覆盖率里阈值最高」那点。
    两个坑都踩过：
    - 取覆盖率**最低**那点，每档都落到 risk=0，AURC 恒为 0；
    - 覆盖率有平台（多个阈值给出同一覆盖率）时取**最低**阈值，会把该档风险
      读成平台上最松的那点。风险随阈值单调不增，同覆盖率下阈值最高 = 风险最低，
      才是部署者真正会选的工作点。
    """
    if not rc:
        return None
    vals = []
    for level in levels:
        ok = [r for r in rc['curve'] if r['coverage'] <= level + 1e-9]
        if not ok:
            continue
        vals.append(max(ok, key=lambda r: (r['coverage'], r['threshold']))['risk'])
    return round(sum(vals) / len(vals), 4) if vals else None


def solve_workpoint(neg_scores: Sequence[float], pos_scores: Sequence[float],
                    target_risk: float) -> Dict[str, object]:
    """反解：在 FAR <= target_risk 的前提下覆盖率最高的阈值（判据 score >= threshold）。

    候选阈值只需取两侧观测值的并集——覆盖率与 FAR 都只在观测值处跳变，
    所以这个候选集里的最优解就是全局最优，不必网格近似。
    """
    neg = np.asarray([v for v in neg_scores if v is not None], dtype=np.float64)
    pos = np.asarray([v for v in pos_scores if v is not None], dtype=np.float64)
    if neg.size == 0 or pos.size == 0:
        return {'target_risk': target_risk, 'reachable': False,
                'reason': '负样本或正样本为空，无法反解'}
    cands = np.unique(np.concatenate([neg, pos]))
    neg_s, pos_s = np.sort(neg), np.sort(pos)
    # 排序必须在循环外做一次；在推导里对每个候选重排是 O(M·N log N)，几千条就卡死。
    risk = (neg_s.size - np.searchsorted(neg_s, cands, side='left')) / neg_s.size
    cov = (pos_s.size - np.searchsorted(pos_s, cands, side='left')) / pos_s.size
    ok = np.nonzero(risk <= target_risk + 1e-12)[0]
    if ok.size == 0:
        return {'target_risk': target_risk, 'reachable': False,
                'min_reachable_risk': round(float(risk.min()), 4),
                'reason': '任何阈值都压不到目标 FAR 以下；需要补硬负样本重训，不是调参'}
    # 同覆盖率可能对应多个阈值（覆盖率是阶梯函数）。取最高的那个：风险随阈值
    # 单调不增，所以同覆盖率下高阈值严格占优——覆盖率一样、FAR 更低。
    best_cov = float(cov[ok].max())
    tied = ok[cov[ok] >= best_cov - 1e-12]
    best = int(tied[np.argmax(cands[tied])])
    return {'target_risk': target_risk, 'reachable': True,
            'threshold': round(float(cands[best]), 6),
            'risk': round(float(risk[best]), 4),
            'coverage': round(float(cov[best]), 4),
            'coverage_loss_vs_full': round(1.0 - float(cov[best]), 4),
            'n_pos': int(pos.size), 'n_neg': int(neg.size)}


# ---------------------------------------------------------------------------
# 诊断指标
# ---------------------------------------------------------------------------

def auc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> Optional[float]:
    """AUC（分数越大越像正样本），Mann-Whitney U 实现，含并列修正。0.5 = 无区分力。"""
    p = np.asarray([v for v in pos_scores if v is not None], dtype=np.float64)
    n = np.asarray([v for v in neg_scores if v is not None], dtype=np.float64)
    if p.size == 0 or n.size == 0:
        return None
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind='mergesort')
    sorted_vals = allv[order]
    ranks = np.empty(allv.size, dtype=np.float64)
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[:p.size].sum() - p.size * (p.size + 1) / 2.0
    return round(float(u / (p.size * n.size)), 4)


def domain_detector_gap(auc_cross: Optional[float], auc_same: Optional[float]) -> Optional[dict]:
    """判断一个信号是「命令判别器」还是「TTS/真人域检测器」。

    auc_cross：正=合成命令，负=真人非命令（跨域配对，域与内容同时变化）。
    auc_same ：正=合成命令，负=合成非命令（同域配对，只有内容变化）。
    同域 AUC 才是这个信号在真人场景下的能力上界证据；跨域远高于同域，
    说明它主要在检测「是不是 TTS」，那种漂亮的跨域曲线不能拿去承诺真人 FAR。
    """
    if auc_cross is None or auc_same is None:
        return None
    gap = auc_cross - auc_same
    if auc_same <= 0.55:
        verdict = '域检测器'
        note = '同域几乎无区分力，跨域成绩来自 TTS 与真人的声学差异，不能外推到真人非命令'
    elif gap > 0.15:
        verdict = '部分依赖域'
        note = '同域有区分力但明显弱于跨域，真人场景的实际效果会低于跨域曲线'
    else:
        verdict = '内容判别器'
        note = '同域与跨域接近，区分的是命令与非命令，而不是录音域'
    return {'auc_cross_domain': auc_cross, 'auc_same_domain': auc_same,
            'gap': round(gap, 4), 'verdict': verdict, 'note': note}


def ece(confidences: Sequence[float], correct: Sequence[bool],
        bins: int = 10) -> Optional[dict]:
    """期望标定误差（Expected Calibration Error），等宽分箱，空箱跳过。

    用等宽而不是等样本数分箱：本项目置信度高度集中在 1.0 附近且有大量并列值，
    等样本数分箱会把**置信度完全相同**的样本切进不同箱，各箱准确率凭空不同，
    ECE 会报出一个由分箱方式造成的假误差。等宽是标准定义，也更好解释。
    """
    conf = np.asarray([float(v) for v in confidences if v is not None], dtype=np.float64)
    corr = np.asarray([c for v, c in zip(confidences, correct)
                       if v is not None], dtype=bool)
    if conf.size == 0 or conf.size != corr.size:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, total = [], 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # 最后一箱闭区间，否则置信度恰好 1.0 的样本会全部漏掉——而本项目
        # 恰恰有大量 1.0（softmax 在合成命令上饱和）。
        mask = (conf >= lo) & (conf <= hi) if hi >= 1.0 else (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        c, y = conf[mask], corr[mask]
        rows.append({'lo': round(float(lo), 3), 'hi': round(float(hi), 3),
                     'n': int(c.size), 'mean_conf': round(float(c.mean()), 4),
                     'accuracy': round(float(y.mean()), 4),
                     'gap': round(abs(float(c.mean()) - float(y.mean())), 4)})
        total += int(c.size)
    if not rows:
        return None
    value = sum(r['n'] * r['gap'] for r in rows) / total
    return {'ece': round(value, 4), 'bins': rows, 'n': total,
            'binned': total, 'binning': 'equal_width'}


def column(rows: List[Dict[str, object]], name: str) -> List[Optional[float]]:
    """从逐样本信号字典列表里抽一列，缺失记 None。"""
    out = []
    for r in rows:
        v = r.get(name)
        out.append(None if v is None else float(v))
    return out


# ---------------------------------------------------------------------------
# 部署侧门限：把「阈值 + 参照池」绑成一个可加载、可降级、可测的判定对象
# ---------------------------------------------------------------------------
GATE_VERSION = 1


class SelectiveGate:
    """双信号 AND 拒识门限的部署实现（S4）。

    为什么单独成一个类，而不是把逻辑散在 ``realtime.py`` 里：

    1. **阈值和参照池是一对工件**。``ood`` = 到池的 kNN 余弦距离取负，池一换，
       所有距离的分布就变，旧阈值必然失配（``docs/pool_size.json`` 逐池重标定了
       29 个配置才读出压缩代价）。把两者绑在同一个加载器里，就不会出现
       「换了池忘了换阈值」这种最危险又最不容易被测出来的错。
    2. **缺件必须显式降级**。池文件没下发、或前向没给出嵌入时，不能崩溃，
       更不能默认接受；这里退回单阈值（默认）或全拒（可配），并把原因写进
       ``reason``，让日志能看出这一条走的是哪条路。
    3. **判定口径必须与评估口径逐字一致**。``ood`` 的负号、``maxprob`` 的归一化、
       缺失即拒识，都由本类统一负责；``scripts/eval_selective.py`` 与
       ``scripts/eval_pool_size.py`` 用的是同一套 ``and_accept``。
    """

    def __init__(self, thresholds: Dict[str, float], pool: Optional[np.ndarray],
                 k: int = 5, fallback_threshold: float = 0.945,
                 fallback_mode: str = 'single', meta: Optional[dict] = None):
        if not thresholds:
            raise ValueError('SelectiveGate 需要至少一个阈值')
        unknown = set(thresholds) - set(ACCEPT_SIGNALS)
        if unknown:
            raise ValueError(f'未知接受信号: {sorted(unknown)}')
        if fallback_mode not in ('single', 'reject'):
            raise ValueError("fallback_mode 只能是 'single' 或 'reject'")
        self.thresholds = {n: float(v) for n, v in thresholds.items()}
        self.pool = None if pool is None else np.asarray(pool, dtype=np.float32)
        self.k = int(k)
        self.fallback_threshold = float(fallback_threshold)
        self.fallback_mode = fallback_mode
        self.meta = dict(meta or {})
        # 池缺失 => ood 算不出来 => 只要规则里有 ood 就必须降级
        self.degraded = bool('ood' in self.thresholds and self.pool is None)
        self.degraded_reason = ('ood_pool_missing' if self.degraded else '')

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, rule_path, pool_path=None, models_dir=None) -> 'SelectiveGate':
        """从规则 JSON（+ 参照池 npz）加载门限。

        规则文件里记录了池的文件名、数组名、k 与标定元数据；池文件缺失时
        **不抛异常**，而是返回 ``degraded=True`` 的门限，由调用方决定是继续
        跑降级路径还是拒绝启动。这样做是因为端侧下发是分步的，模型先到、
        池后到是正常状态，不能让整个识别管线因此起不来。
        """
        rule_path = Path(rule_path)
        rule = json.loads(rule_path.read_text(encoding='utf-8'))
        ood_cfg = rule.get('ood') or {}
        pool = None
        pool_src = None
        if pool_path is None and ood_cfg.get('pool') and models_dir is not None:
            pool_path = Path(models_dir) / ood_cfg['pool']
        if pool_path is not None:
            pool_path = Path(pool_path)
            if pool_path.exists():
                with np.load(str(pool_path), allow_pickle=False) as z:
                    arr = ood_cfg.get('pool_array') or 'embedding'
                    if arr not in z:
                        raise KeyError(f'{pool_path} 里没有数组 {arr}（有 {list(z.keys())}）')
                    pool = np.asarray(z[arr], dtype=np.float32)
                pool_src = str(pool_path)
                want = ood_cfg.get('pool_size')
                if want is not None and int(want) != int(pool.shape[0]):
                    raise ValueError(
                        f'池条数与规则文件不符：规则写 {want}，实际 {pool.shape[0]}。'
                        f'阈值是按 {want} 条标定的，用错池等于用错阈值。')
        fb = rule.get('fallback') or {}
        return cls(
            thresholds=dict(rule['thresholds']),
            pool=pool,
            k=int(ood_cfg.get('k', 5)),
            fallback_threshold=float(fb.get('threshold', 0.945)),
            fallback_mode=str(fb.get('mode', 'single')),
            meta={'version': rule.get('version'), 'rule_path': str(rule_path),
                  'pool_path': pool_src, 'signals': list(rule.get('signals') or []),
                  'calibration': rule.get('calibration') or {},
                  'evidence': rule.get('evidence') or []},
        )

    # ------------------------------------------------------------------
    def signals_of(self, prob: np.ndarray,
                   embedding: Optional[np.ndarray] = None) -> Dict[str, Optional[float]]:
        """算出规则需要的信号；``ood`` 拿不到嵌入或池时记 None（缺失即拒识）。"""
        p = np.asarray(prob, dtype=np.float64).reshape(-1)
        s = float(p.sum())
        if not np.isfinite(s) or s <= 0:
            raise ValueError('prob 非法：和为 0 或含 NaN')
        p = p / s
        out: Dict[str, Optional[float]] = {'maxprob': float(p.max())}
        if self.pool is not None and embedding is not None:
            d = knn_distance(np.asarray(embedding, dtype=np.float64).reshape(-1),
                             self.pool, self.k)
            # 负号不能省：knn_distance 越小越像池内命令，而 ACCEPT_SIGNALS 统一是
            # 越大越像命令。signals() 里就是这个约定，两处必须一致。
            out['ood'] = None if d is None else -float(d)
        else:
            out['ood'] = None
        return out

    def decide(self, prob: np.ndarray,
               embedding: Optional[np.ndarray] = None) -> Dict[str, object]:
        """返回 ``{accepted, rule, signals, reason}``。

        ``rule`` 标明这一条走的是双信号还是降级路径，日志与测试都靠它区分；
        降级不是错误，但必须可见。
        """
        sig = self.signals_of(prob, embedding)
        # 只要规则需要的信号有一个算不出来就降级；不能只对 ood 特判，
        # 否则 maxprob-only 的门限（对照组）会被误当成缺件。
        missing = [n for n in self.thresholds if sig.get(n) is None]
        if missing:
            if self.fallback_mode == 'reject':
                return {'accepted': False, 'rule': 'fallback_reject', 'signals': sig,
                        'reason': 'signals_unavailable:' + ','.join(missing)
                                  + ':fallback_reject'}
            mp = sig.get('maxprob')
            ok = bool(mp is not None and mp >= self.fallback_threshold)
            return {'accepted': ok, 'rule': 'fallback_single', 'signals': sig,
                    'reason': ('signals_unavailable:' + ','.join(missing)
                               + (f'|below:{self.fallback_threshold:.4f}' if not ok else ''))}
        ok = and_accept(sig, self.thresholds)
        if ok:
            reason = ''
        else:
            fails = [n for n, tau in self.thresholds.items()
                     if sig.get(n) is None or float(sig[n]) < float(tau)]
            reason = 'and_reject:' + ','.join(fails)
        return {'accepted': ok, 'rule': 'and', 'signals': sig, 'reason': reason}

    # ------------------------------------------------------------------
    def as_dict(self) -> dict:
        """可序列化描述，用于写进结果日志（不含池本体）。"""
        return {
            'version': GATE_VERSION,
            'thresholds': {n: round(v, 6) for n, v in self.thresholds.items()},
            'ood_k': self.k,
            'pool_size': None if self.pool is None else int(self.pool.shape[0]),
            'pool_dim': None if self.pool is None else int(self.pool.shape[1]),
            'degraded': bool(self.degraded),
            'degraded_reason': self.degraded_reason,
            'fallback': {'mode': self.fallback_mode,
                         'threshold': round(self.fallback_threshold, 6)},
            'calibration': self.meta.get('calibration') or {},
        }


def single_threshold_gate(threshold: float, mode: str = 'single') -> SelectiveGate:
    """只有 max-prob 一个信号的门限：S3 之前的部署行为，用作对照组与兜底。"""
    return SelectiveGate(thresholds={'maxprob': float(threshold)}, pool=None,
                         fallback_threshold=float(threshold), fallback_mode=mode)

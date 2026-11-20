# -*- coding: utf-8 -*-
"""生成下发用的双信号拒识工件：参照池 npz + 规则 JSON（S4-2）。

为什么要有这个脚本，而不是手写两个文件
------------------------------------
``ood`` = 到参照池的 kNN 余弦距离取负，**阈值和池是一对**：池换了，距离分布就变，
旧阈值必然失配。手写文件最容易犯的错就是「换了池忘了重标定阈值」，而且这种错
在单元测试里看不出来（数字都还在合理范围内），只有 FAR 会悄悄变差。
所以这里把「建池 → 重标定 → 评估 → 落盘」串成一步，并把标定方式与实测数字
一起写进规则文件，谁用的时候都能看见这对阈值是在什么池上、用什么口径标定的。

标定口径（与 ``scripts/eval_pool_size.py`` 一致）
----------------------------------------------
- **下发阈值** = 在全部真人负样本 + 合成正样本上一次性拟合（``and_workpoint``）。
  这是部署时真正会用的那对数字。
- **性能估计** = 同一批数据的 K 折交叉拟合（``crossfit_and_rule``）：每折阈值只用
  其余折搜出，只在本折评估，因此报出的 FAR 是**样本外**的。
  两者必须一起报：只报前者是样本内自夸，只报后者又拿不到要下发的数字。
- 不用「72 条合成负样本」标定：``docs/pool_size.json`` 的 bootstrap 对照显示那样
  搜出的 FAR 在 [0.066, 0.20]（p10–p90）之间摆动，不可复现。

存储精度
--------
f16 能把池体积砍半（500 条：500 KB → 250 KB），直接决定 S2 的 ≤300 KB 预算能否满足。
但 f16 会改变距离的小数尾数，进而可能移动工作点，所以这里**不假设它无害**：
用 f16 往返后的池重新标定并交叉拟合，达不到判据就硬失败，让人改回 float32。

判据（与 pool_size 实验同一套，预先写明）：真实域 FAR ≤ 0.10 且合成覆盖率 ≥ 0.80。

示例：
    python scripts/build_selective_gate.py --pool-size 500 --pool-seed 0 --pool-dtype float16
    python scripts/build_selective_gate.py --dry-run          # 只看数字不落盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent / 'DialectVoice_Research'
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_pool_size import (  # noqa: E402
    ACCEPT_MAX_FAR, ACCEPT_MIN_COVERAGE, PAIR, batch_knn, make_pool_idx)
from eval_real_far import dangerous_split, wilson  # noqa: E402
from eval_selective import load_raw  # noqa: E402
from runtime.selective import and_accept, and_workpoint, crossfit_and_rule  # noqa: E402

DEFAULT_RAW = RESEARCH / 'data/real_domain_testset/selective_raw.npz'
DEFAULT_OUT_DIR = RESEARCH / 'data/models'
POOL_NAME = 'v3_knn_pool.npz'
GATE_NAME = 'v3_selective_gate.json'
CLASSES = ('ALL_OFF', 'CURTAIN_CLOSE', 'CURTAIN_OPEN', 'LIGHT_OFF', 'LIGHT_ON', 'STATUS_ALL')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cols_for(pool: np.ndarray, groups: dict, k: int) -> dict:
    """三组样本的 (maxprob, ood) 列。ood 的负号在这里统一处理。"""
    out = {}
    for g in ('syn_known', 'syn_unknown', 'real'):
        prob = np.asarray(groups[g]['prob'], dtype=np.float64)
        mp = prob.max(axis=1) / prob.sum(axis=1)
        # ood = **-kNN 距离**（越大越像命令）。漏掉负号会让 AND 规则整体反向。
        ood = -batch_knn(groups[g]['embedding'], pool, k)
        out[g] = {'maxprob': [float(v) for v in mp], 'ood': [float(v) for v in ood]}
    return out


def evaluate(th: dict, cols: dict, real_cmds) -> dict:
    """给定阈值，报真实域 FAR / 危险误动作率 / 合成覆盖率。"""
    acc = [and_accept({'maxprob': cols['real']['maxprob'][i],
                       'ood': cols['real']['ood'][i]}, th)
           for i in range(len(cols['real']['maxprob']))]
    kacc = [and_accept({'maxprob': cols['syn_known']['maxprob'][i],
                        'ood': cols['syn_known']['ood'][i]}, th)
            for i in range(len(cols['syn_known']['maxprob']))]
    n, na = len(acc), int(sum(acc))
    cmds = Counter(real_cmds[i] for i in range(n) if acc[i])
    dang = dangerous_split(cmds, n)
    return {
        'thresholds': {k: round(float(v), 6) for k, v in th.items()},
        'real_far': round(na / n, 4) if n else None,
        'real_ci95': list(wilson(na, n)) if n else None,
        'real_accepted': na, 'real_n': n,
        'dangerous_rate': dang['dangerous']['rate'],
        'dangerous_ci95': dang['dangerous']['ci95'],
        'dangerous_accepted': dang['dangerous']['accepted'],
        'benign_rate': dang['benign']['rate'],
        'syn_coverage': round(sum(kacc) / len(kacc), 4) if kacc else None,
        'false_accept_by_command': dict(sorted(cmds.items(), key=lambda kv: -kv[1])),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-cache', default=str(DEFAULT_RAW))
    ap.add_argument('--pool-size', type=int, default=500,
                    help='参照池条数；500 是 pool_size 实验里最小的全抽样达标配置')
    ap.add_argument('--pool-seed', type=int, default=0)
    ap.add_argument('--pool-dtype', choices=('float32', 'float16'), default='float16')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--target-risk', type=float, default=0.10)
    ap.add_argument('--xfit-folds', type=int, default=2)
    ap.add_argument('--xfit-seeds', default='0,1,2')
    ap.add_argument('--fallback-threshold', type=float, default=0.945,
                    help='池/嵌入缺失时退回的单阈值（= 现部署阈值）')
    ap.add_argument('--fallback-mode', choices=('single', 'reject'), default='single')
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR),
                    help='权威副本目录（Research data/models/）；runtime 由 sync_runtime.py 复制')
    ap.add_argument('--report', default=str(ROOT / 'docs' / 'selective_gate.json'))
    ap.add_argument('--dry-run', action='store_true', help='只算不写')
    args = ap.parse_args(argv)

    cache = Path(args.raw_cache)
    if not cache.exists():
        sys.exit(f'缺原始量缓存: {cache}\n先跑 scripts/eval_selective.py 生成它')
    groups = load_raw(cache, ('pool', 'syn_known', 'syn_unknown', 'real'))
    emb, labs = groups['pool']['embedding'], groups['pool']['label']
    xfit_seeds = tuple(int(s) for s in args.xfit_seeds.split(',') if s.strip())

    pool, pool_idx = make_pool_idx('subsample', emb, labs, args.pool_size,
                                   args.pool_seed, list(CLASSES))
    pool_labels = [str(labs[i]) for i in pool_idx]
    dtype_note = 'float32（原样）'
    if args.pool_dtype == 'float16':
        # 模拟「存成 f16 再读回来」的往返，而不是只在落盘时转换：
        # 标定必须发生在**实际下发的那份数值**上，否则阈值和线上用的距离不是同一套。
        pool = pool.astype(np.float16).astype(np.float32)
        dtype_note = 'float16 往返（标定与评估都用往返后的数值）'

    cols = cols_for(pool, groups, args.k)
    real_cmds = list(groups['real']['command'])
    n_real, n_pos = len(cols['real']['maxprob']), len(cols['syn_known']['maxprob'])

    deployed = and_workpoint(cols['real'], cols['syn_known'], PAIR,
                             target_risk=args.target_risk)
    if not deployed.get('reachable'):
        sys.exit(f'全量拟合搜不出满足目标风险 {args.target_risk} 的阈值：{deployed.get("reason")}')
    xf = crossfit_and_rule(cols['real'], cols['syn_known'], PAIR, args.target_risk,
                           n_folds=args.xfit_folds, seeds=xfit_seeds)
    if not xf.get('reachable'):
        sys.exit(f'交叉拟合搜不出阈值：{xf.get("reason")}')

    dep_eval = evaluate(deployed['thresholds'], cols, real_cmds)
    modal = xf['threshold_modes'][0]['thresholds']
    xf_eval = evaluate(modal, cols, real_cmds)

    far_out = float(xf['pooled_far_mean'])
    cov_out = float(xf['pos_coverage_mean'])
    meets = bool(far_out <= ACCEPT_MAX_FAR and cov_out >= ACCEPT_MIN_COVERAGE)
    th_same = {k: round(float(deployed['thresholds'][k]), 6)
               for k in PAIR} == {k: round(float(modal[k]), 6) for k in PAIR}

    print(f'池: {pool.shape} dtype={args.pool_dtype} ({dtype_note})，'
          f'{pool.nbytes / 1024:.1f} KB f32 内存占用', flush=True)
    print(f'负样本 {n_real} 条（真人非命令）/ 正样本 {n_pos} 条（合成已知命令）', flush=True)
    print(f'下发阈值（全量拟合）: {dep_eval["thresholds"]}', flush=True)
    print(f'  样本内 FAR {dep_eval["real_far"]} 危险 {dep_eval["dangerous_rate"]} '
          f'覆盖率 {dep_eval["syn_coverage"]}', flush=True)
    print(f'交叉拟合（{args.xfit_folds} 折 × {len(xfit_seeds)} 次划分）: '
          f'阈值组数 {xf["n_distinct_thresholds"]}，众数阈值 {xf_eval["thresholds"]}', flush=True)
    print(f'  **样本外** FAR {far_out}（min {xf["pooled_far_min"]} / max {xf["pooled_far_max"]}）'
          f' 覆盖率 {cov_out}（min {xf["pos_coverage_min"]}）', flush=True)
    print(f'  下发阈值与众数阈值{"一致" if th_same else "**不一致**"}', flush=True)
    print(f'判据 FAR ≤ {ACCEPT_MAX_FAR} 且覆盖率 ≥ {ACCEPT_MIN_COVERAGE}: '
          f'{"达标" if meets else "**未达标**"}', flush=True)

    problems = []
    if not meets:
        problems.append(
            f'交叉拟合的样本外指标未达判据（FAR {far_out} / 覆盖率 {cov_out}）。'
            f'若因 {args.pool_dtype} 存储导致，改用 --pool-dtype float32 重试；'
            f'若因池太小，改用 --pool-size 1000 或满池。')
    if xf['n_distinct_thresholds'] != 1:
        problems.append(
            f'交叉拟合在折间选出 {xf["n_distinct_thresholds"]} 组不同阈值，'
            f'工作点不稳定；样本外 FAR 是这套流程的估计，不是某一对固定阈值的估计。'
            f'（记录为警告，不阻断落盘）')
    if not th_same:
        problems.append(
            '下发阈值（全量拟合）与交叉拟合众数阈值不一致：报出的样本外 FAR '
            '严格来说属于众数阈值那一对。两者差距已记录在 report 里，需人工确认。')

    calibration = {
        'method': f'and_workpoint 全量拟合下发 + crossfit_and_rule '
                  f'{args.xfit_folds} 折 × {len(xfit_seeds)} 次划分做样本外估计',
        'target_risk': args.target_risk,
        'n_negatives': n_real,
        'n_positives': n_pos,
        'negatives_source': '真人粤语非命令语音（FLEURS yue_hant_hk，CC BY 4.0，'
                            'data/real_domain_testset/manifest.csv）',
        'positives_source': '合成 val 已知命令（TTS，非真人）',
        'n_distinct_thresholds_in_crossfit': int(xf['n_distinct_thresholds']),
        'deployed_equals_crossfit_mode': bool(th_same),
        'out_of_sample_far': far_out,
        'out_of_sample_far_range': [xf['pooled_far_min'], xf['pooled_far_max']],
        'out_of_sample_coverage': cov_out,
        'note': '真人负样本参与了阈值搜索，因此本项目在标定环节**需要**真人数据；'
                '交叉拟合保证每条样本都由没见过它的阈值判定，FAR 是样本外估计。',
    }

    gate = {
        'version': 1,
        'generated': date.today().isoformat(),
        'rule': 'and',
        'signals': list(PAIR),
        'thresholds': dep_eval['thresholds'],
        'ood': {
            'kind': 'neg_knn_cosine',
            'k': args.k,
            'pool': POOL_NAME,
            'pool_array': 'embedding',
            'pool_size': int(pool.shape[0]),
            'pool_dim': int(pool.shape[1]),
            'pool_dtype_stored': args.pool_dtype,
            'pool_bytes_stored': int(pool.shape[0] * pool.shape[1]
                                     * (2 if args.pool_dtype == 'float16' else 4)),
            'pool_source': f'合成 train 分层子采样（按 {len(CLASSES)} 个命令类别配额），'
                           f'size={args.pool_size} seed={args.pool_seed}',
            'pool_note': '池与阈值是一对工件：换池必须重跑本脚本重标定',
        },
        'fallback': {
            'mode': args.fallback_mode,
            'threshold': args.fallback_threshold,
            'artifact': 'v3_thresholds_vad.json',
            'note': '池文件缺失或前向未给出嵌入时走这条路径；已知代价是真实域 FAR '
                    '0.3110（docs/real_far.json），比双信号差 4.7 倍',
        },
        'calibration': calibration,
        'performance': {
            'shipped_thresholds_in_sample': dep_eval,
            'crossfit_mode_thresholds': xf_eval,
            'out_of_sample': {
                'real_far': far_out,
                'real_far_range_over_splits': [xf['pooled_far_min'], xf['pooled_far_max']],
                'syn_coverage': cov_out,
                'syn_coverage_min': xf['pos_coverage_min'],
                'meets_target': meets,
                'accept_max_far': ACCEPT_MAX_FAR,
                'accept_min_coverage': ACCEPT_MIN_COVERAGE,
            },
            'baseline_single_threshold': {
                'threshold': args.fallback_threshold,
                'real_far': 0.3110, 'dangerous_rate': 0.1437, 'syn_coverage': 0.7864,
                'source': 'docs/real_far.json（S0 实测，同一批真人负样本）',
            },
        },
        'evidence': ['docs/selective_eval.json', 'docs/pool_size.json',
                     'docs/real_far.json', 'docs/selective_gate.json'],
        'caveats': [
            '覆盖率一侧全是合成 TTS 命令；真人命令在这对阈值下的覆盖率未测（真人正样本为 0）。',
            'FAR 对应的负样本是 FLEURS 朗读式近讲粤语被 VAD 切成命令长度的片段，'
            '家庭远场/电视背景环境大概率更差，0.0808 一类数字不是保守上界。',
            '交叉拟合用 2 折，每折只用约一半负样本标定，因此样本外估计相对'
            '「全量拟合后部署」偏保守。',
        ],
    }

    report = {
        'generated': date.today().isoformat(),
        'built_by': 'scripts/build_selective_gate.py',
        'args': {k: v for k, v in vars(args).items()},
        'pool': {'shape': list(pool.shape), 'dtype_note': dtype_note,
                 'n_by_class': dict(sorted(Counter(pool_labels).items())),
                 'source_indices_sha256': hashlib.sha256(
                     json.dumps(pool_idx).encode('utf-8')).hexdigest()[:16],
                 'bytes_in_memory_f32': int(pool.nbytes)},
        'groups': {g: groups[g]['n'] for g in ('pool', 'syn_known', 'syn_unknown', 'real')},
        'deployed_fit': dep_eval,
        'crossfit': {
            'n_folds': xf['n_folds'], 'seeds': xf['seeds'],
            'n_distinct_thresholds': xf['n_distinct_thresholds'],
            'threshold_modes': xf['threshold_modes'],
            'pooled_far_mean': xf['pooled_far_mean'],
            'pooled_far_min': xf['pooled_far_min'],
            'pooled_far_max': xf['pooled_far_max'],
            'pos_coverage_mean': xf['pos_coverage_mean'],
            'pos_coverage_min': xf['pos_coverage_min'],
            'folds': xf['folds'],
        },
        'meets_target': meets,
        'problems': problems,
        'written': {},
    }

    if args.dry_run:
        print('\n[dry-run] 未写任何文件（含参照池、规则文件与构建报告）', flush=True)
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pool_path = out_dir / POOL_NAME
        gate_path = out_dir / GATE_NAME
        # 存的就是标定用的那份数值：float16 池按 f16 存，加载后转回 f32 与这里一致。
        # 若在这里改成压缩存储或换精度，标定与线上就不是同一套距离，必须重跑本脚本。
        store = (pool.astype(np.float16) if args.pool_dtype == 'float16'
                 else pool.astype(np.float32))
        np.savez(str(pool_path), embedding=store, label=np.asarray(pool_labels))
        gate['ood']['pool_sha256'] = sha256(pool_path)
        gate['ood']['pool_n_by_class'] = dict(sorted(Counter(pool_labels).items()))
        gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding='utf-8')
        report['written'] = {
            'pool': str(pool_path), 'pool_sha256': gate['ood']['pool_sha256'],
            'pool_bytes': int(pool_path.stat().st_size),
            'gate': str(gate_path), 'gate_sha256': sha256(gate_path),
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding='utf-8')
        print(f'\n已写入 {pool_path}（{pool_path.stat().st_size / 1024:.1f} KB）', flush=True)
        print(f'已写入 {gate_path}', flush=True)
        print(f'构建报告: {args.report}', flush=True)
        print('下一步：python scripts/sync_runtime.py 把两个工件复制进 runtime/models/',
              flush=True)

    if problems:
        print('\n=== 需要注意 ===')
        for p in problems:
            print('-', p)
    if not meets:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

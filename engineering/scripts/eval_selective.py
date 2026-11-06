# -*- coding: utf-8 -*-
"""选择性分类评估：第二信号能不能在保住覆盖率的前提下把真实域 FAR 压下来。

S0 的结论是单一 max-prob 阈值不可行（真实域 FAR 0.3110，FAR 0.15→0.10 之间
合成覆盖率从 0.6159 断崖到 0.0023，且 p90 置信度已达 0.9923）。本脚本检验
不依赖 softmax 标定质量的候选信号：集成一致性（逐种子概率分歧）、margin、
负熵、嵌入 kNN 距离（OOD），以及它们的 min-z / z 加和 / AND 双阈值组合。

三组数据，两种配对——这是本脚本最重要的设计
------------------------------------------
- ``syn_known``   合成已知命令（val 分割，带标签）→ 覆盖率一侧的正样本
- ``syn_unknown`` 合成非命令（edge-tts 念的非命令行）→ **同域**负样本
- ``real``        真人非命令（FLEURS 粤语，S0 测试集）→ **跨域**负样本

只报跨域配对会骗人：正样本全是 TTS、跨域负样本全是真人，任何能区分
「TTS 与真人」的信号都会得到漂亮曲线，却完全不能说明它能拒掉**真人说的
非命令**。所以每个信号同时给同域 AUC 与跨域 AUC，并由
``selective.domain_detector_gap`` 判定它是内容判别器还是域检测器。

标定纪律
--------
组合规则的阈值与 z 分数统计量只在合成数据上标定（``syn_known`` + ``syn_unknown``），
真实测试集只用来评估。另外单独给一份 **oracle**（阈值直接在真实负样本上搜）
作为能力上界，明确标注不可部署——两者的差距就是「缺真人负样本」的代价。
kNN 参照池取 **train** 分割，不含被评估的 val，否则自匹配会把距离压成 0。

示例：
    python scripts/eval_selective.py \
        --manifest "D:/Run_Tao/DialectVoice_Research/data/real_domain_testset/manifest.csv" \
        --corpus-manifest "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" \
        --unknown-dir "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/unknown_v3_vad" \
        --out docs/selective_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_real_far import dangerous_split, load_rows, percentiles, wilson  # noqa: E402
from runtime.audio_io import load_wav  # noqa: E402
from runtime.features_v3 import extract_sequence  # noqa: E402
from runtime.selective import (  # noqa: E402
    ACCEPT_SIGNALS, and_accept, and_workpoint, auc, aurc, build_pool, combo_min_z,
    combo_sum_z, domain_detector_gap, ece, risk_coverage, signals, solve_workpoint,
    zstats,
)
from runtime.v3_model import V3Predictor  # noqa: E402
from src.config import load_config  # noqa: E402
from src.vad import EnergyVad  # noqa: E402

COVERAGE_LEVELS = tuple(round(1.0 - 0.05 * i, 2) for i in range(20))
SIGNAL_ORDER = ACCEPT_SIGNALS + ('combo_min_z_maxprob_ood', 'combo_min_z_maxprob_seed',
                                 'combo_sum_z_maxprob_ood')


# ---------------------------------------------------------------------------
# 一次前向，存原始量；信号计算留到后面，改口径不用重跑推理
# ---------------------------------------------------------------------------

def extract_group(predictor, cfg, items, use_vad=True, tag=''):
    """items: [(path, label_or_None)] -> 原始概率/逐种子概率/嵌入/预测命令。"""
    probs, seeds, embs, labels, correct, cmds, skipped = [], [], [], [], [], [], []
    want = int(predictor.feature_spec.sample_rate)
    t0 = time.perf_counter()
    for i, (path, label) in enumerate(items, 1):
        audio, sr = load_wav(path)
        if use_vad:
            vad = EnergyVad(fs=sr, **cfg.vad_kwargs())
            segs = vad.feed(audio) + vad.flush()
            if not segs:
                skipped.append(str(path))
                continue
            audio = segs[0]
        if int(sr) != want:
            skipped.append(str(path))
            continue
        seq = extract_sequence(np.asarray(audio, dtype=np.float32).reshape(-1),
                               int(sr), predictor.feature_spec)
        pred, extras = predictor.predict_detailed(seq)
        probs.append(np.array([pred.scores[c] for c in predictor.classes_],
                              dtype=np.float32))
        seeds.append(extras['seed_probs'].astype(np.float32))
        embs.append(extras['embedding'].astype(np.float32))
        labels.append(label)
        cmds.append(pred.command)
        correct.append(None if label is None else bool(pred.command == label))
        if i % 250 == 0:
            print(f'  [{tag}] {i}/{len(items)}  {time.perf_counter() - t0:.0f}s', flush=True)
    return {'prob': np.stack(probs) if probs else np.zeros((0, len(predictor.classes_))),
            'seed_probs': np.stack(seeds) if seeds else np.zeros((0, 0, 0)),
            'embedding': np.stack(embs) if embs else np.zeros((0, 0)),
            'label': labels, 'correct': correct, 'command': cmds, 'skipped': skipped,
            'n': len(probs), 'seconds': round(time.perf_counter() - t0, 1)}


def save_raw(path: Path, groups: dict) -> None:
    arrays = {}
    for gname, g in groups.items():
        arrays[f'{gname}/prob'] = g['prob']
        arrays[f'{gname}/seed_probs'] = g['seed_probs']
        arrays[f'{gname}/embedding'] = g['embedding']
        arrays[f'{gname}/label'] = np.array(
            ['' if x is None else str(x) for x in g['label']], dtype='<U32')
        arrays[f'{gname}/correct'] = np.array(
            [-1 if x is None else int(x) for x in g['correct']], dtype=np.int8)
        arrays[f'{gname}/command'] = np.array(g['command'], dtype='<U32')
        arrays[f'{gname}/skipped'] = np.array(g['skipped'], dtype='<U512')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_raw(path: Path, names) -> dict:
    with np.load(path, allow_pickle=False) as z:
        groups = {}
        for gname in names:
            lab = [None if s == '' else str(s) for s in z[f'{gname}/label']]
            cor = [None if v < 0 else bool(v) for v in z[f'{gname}/correct']]
            groups[gname] = {'prob': z[f'{gname}/prob'],
                             'seed_probs': z[f'{gname}/seed_probs'],
                             'embedding': z[f'{gname}/embedding'],
                             'label': lab, 'correct': cor,
                             'command': [str(s) for s in z[f'{gname}/command']],
                             'skipped': [str(s) for s in z[f'{gname}/skipped']],
                             'n': int(z[f'{gname}/prob'].shape[0])}
        return groups


# ---------------------------------------------------------------------------
# 信号列
# ---------------------------------------------------------------------------

def signal_columns(group: dict, pool, k: int, zref: dict) -> dict:
    """把一个组的原始量转成逐信号的列。zref 是各信号的 z 标定统计量。"""
    cols = {name: [] for name in SIGNAL_ORDER}
    for i in range(group['n']):
        row = signals(group['prob'][i], group['seed_probs'][i],
                      embedding=group['embedding'][i] if group['embedding'].size else None,
                      pool=pool, k=k)
        for name in ACCEPT_SIGNALS:
            cols[name].append(row.get(name))
        zs_ood = {n: zref[n] for n in ('maxprob', 'ood')}
        zs_seed = {n: zref[n] for n in ('maxprob', 'seed_agree')}
        cols['combo_min_z_maxprob_ood'].append(combo_min_z(row, zs_ood))
        cols['combo_min_z_maxprob_seed'].append(combo_min_z(row, zs_seed))
        cols['combo_sum_z_maxprob_ood'].append(combo_sum_z(row, zs_ood))
    return cols


def evaluate_signal(cols_pos, cols_neg_real, cols_neg_unknown, targets):
    """跨域（负=真人）与同域（负=合成非命令）两套配对，各给 AUC / 风险-覆盖 / 工作点。"""
    rc = risk_coverage(cols_neg_real, cols_pos, n_points=60)
    return {
        'auc_cross_domain': auc(cols_pos, cols_neg_real),
        'auc_same_domain': auc(cols_pos, cols_neg_unknown),
        'risk_coverage_cross_domain': rc,
        'aurc_cross_domain': aurc(rc, COVERAGE_LEVELS),
        'workpoints_cross_domain': [solve_workpoint(cols_neg_real, cols_pos, t)
                                    for t in targets],
        'workpoints_same_domain': [solve_workpoint(cols_neg_unknown, cols_pos, t)
                                   for t in targets],
    }


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='S0 真实域测试集 manifest.csv')
    parser.add_argument('--subset', default='real_test')
    parser.add_argument('--corpus-manifest', required=True, help='合成语料 manifest.csv')
    parser.add_argument('--known-subset', default='val', help='覆盖率一侧的合成已知命令')
    parser.add_argument('--pool-subset', default='train', help='kNN 参照池分割（不得与 known 重叠）')
    parser.add_argument('--unknown-dir', required=True, help='合成非命令 WAV 目录（同域负样本）')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--no-vad', action='store_true')
    parser.add_argument('--targets', default='0.10,0.05,0.02')
    parser.add_argument('--config', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--thresholds', default=None)
    parser.add_argument('--limit', type=int, default=0, help='每组只跑前 N 条（调试用）')
    parser.add_argument('--raw-cache', default=str(
        ROOT.parent / 'DialectVoice_Research' / 'data' / 'real_domain_testset'
        / 'selective_raw.npz'))
    parser.add_argument('--reuse-raw', action='store_true',
                        help='跳过推理，直接用 --raw-cache 里的原始量重算信号与工作点')
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'selective_eval.json'))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    predictor = V3Predictor.load(str(args.model or cfg.model_path),
                                 str(args.thresholds or cfg.thresholds_path))
    targets = [float(t) for t in args.targets.split(',') if t.strip()]
    use_vad = not args.no_vad
    cache = Path(args.raw_cache)

    if args.reuse_raw:
        if not cache.exists():
            sys.exit(f'--reuse-raw 需要已有缓存: {cache}')
        groups = load_raw(cache, ('pool', 'syn_known', 'syn_unknown', 'real'))
        print(f'复用原始量缓存: {cache}')
    else:
        known_rows = load_rows(Path(args.corpus_manifest), args.known_subset)
        pool_rows = load_rows(Path(args.corpus_manifest), args.pool_subset)
        real_rows = load_rows(Path(args.manifest), args.subset)
        unknown_paths = sorted(Path(args.unknown_dir).glob('*.wav'))
        if not known_rows or not pool_rows or not real_rows or not unknown_paths:
            sys.exit('某一组数据为空，检查 --corpus-manifest / --manifest / --unknown-dir')
        if args.limit:
            known_rows, pool_rows = known_rows[:args.limit], pool_rows[:args.limit]
            real_rows, unknown_paths = real_rows[:args.limit], unknown_paths[:args.limit]
        print(f'组规模: pool(train)={len(pool_rows)} known(val)={len(known_rows)} '
              f'unknown={len(unknown_paths)} real={len(real_rows)}', flush=True)
        groups = {}
        for name, items in (
                ('pool', [(Path(r['audio_path']), r.get('command')) for r in pool_rows]),
                ('syn_known', [(Path(r['audio_path']), r.get('command')) for r in known_rows]),
                ('syn_unknown', [(p, None) for p in unknown_paths]),
                ('real', [(Path(r['audio_path']), None) for r in real_rows])):
            print(f'推理 {name} ...', flush=True)
            groups[name] = extract_group(predictor, cfg, items, use_vad=use_vad, tag=name)
        save_raw(cache, groups)
        print('原始量已缓存:', cache)

    # kNN 参照池：只用 train，且 L2 归一化后存起来
    pool = build_pool([groups['pool']['embedding'][i] for i in range(groups['pool']['n'])])
    print(f'参照池: {pool.shape}')

    # z 标定统计量只来自合成已知命令（部署可见），不碰真实测试集
    known_raw = {}
    for i in range(groups['syn_known']['n']):
        row = signals(groups['syn_known']['prob'][i], groups['syn_known']['seed_probs'][i],
                      embedding=groups['syn_known']['embedding'][i], pool=pool, k=args.k)
        for name in ACCEPT_SIGNALS:
            known_raw.setdefault(name, []).append(row.get(name))
    zref = {name: zstats(known_raw[name]) for name in ACCEPT_SIGNALS}

    cols = {name: signal_columns(groups[name], pool, args.k, zref)
            for name in ('syn_known', 'syn_unknown', 'real')}

    deployed = float(predictor.threshold)
    real_maxprob = cols['real']['maxprob']
    known_maxprob = cols['syn_known']['maxprob']
    unk_maxprob = cols['syn_unknown']['maxprob']

    per_signal = {}
    for name in SIGNAL_ORDER:
        pos = cols['syn_known'][name]
        entry = evaluate_signal(pos, cols['real'][name], cols['syn_unknown'][name], targets)
        entry['domain_gap'] = domain_detector_gap(entry['auc_cross_domain'],
                                                  entry['auc_same_domain'])
        entry['percentiles'] = {g: percentiles([v for v in cols[g][name] if v is not None])
                                for g in ('syn_known', 'syn_unknown', 'real')}
        entry['n'] = {g: sum(1 for v in cols[g][name] if v is not None)
                      for g in ('syn_known', 'syn_unknown', 'real')}
        per_signal[name] = entry

    # AND 双阈值：deployable（合成上标定）与 oracle（真实负样本上搜，仅上界）
    and_pairs = (('maxprob', 'ood'), ('maxprob', 'seed_agree'))
    and_results = {}
    real_cmds = groups['real']['command']
    for pair in and_pairs:
        key = '+'.join(pair)
        cal = and_workpoint({n: cols['syn_unknown'][n] for n in pair},
                            {n: cols['syn_known'][n] for n in pair},
                            pair, target_risk=targets[0])
        applied = None
        if cal.get('reachable'):
            acc = [and_accept({n: v for n, v in zip(pair, row_vals)}, cal['thresholds'])
                   for row_vals in zip(*(cols['real'][n] for n in pair))]
            k_acc = [and_accept({n: v for n, v in zip(pair, row_vals)}, cal['thresholds'])
                     for row_vals in zip(*(cols['syn_known'][n] for n in pair))]
            # 误接受落到哪个命令决定安全含义：STATUS_ALL 只是播一句状态，
            # ALL_OFF 是真的把全屋关掉。降 FAR 的成果必须按这个口径复核，
            # 否则可能只是把危险误动作换成了无害误动作。
            accepted_cmds = Counter(real_cmds[i] for i in range(len(acc)) if acc[i])
            applied = {
                'real_far': round(sum(acc) / len(acc), 4) if acc else None,
                'real_ci95': list(wilson(sum(acc), len(acc))),
                'real_accepted': int(sum(acc)), 'real_n': len(acc),
                'syn_known_coverage': round(sum(k_acc) / len(k_acc), 4) if k_acc else None,
                'false_accept_by_command': dict(sorted(accepted_cmds.items(),
                                                       key=lambda kv: -kv[1])),
                'dangerous_split': dangerous_split(accepted_cmds, len(acc)),
            }
        oracle = and_workpoint({n: cols['real'][n] for n in pair},
                               {n: cols['syn_known'][n] for n in pair},
                               pair, target_risk=targets[0])
        and_results[key] = {'signals': list(pair), 'target_risk': targets[0],
                            'calibrated_on_synthetic': cal,
                            'applied_to_real': applied, 'oracle_on_real': oracle}

    correct = groups['syn_known']['correct']
    deployed_cmds = Counter(real_cmds[i] for i in range(groups['real']['n'])
                            if real_maxprob[i] >= deployed)
    report = {
        'generated': date.today().isoformat(),
        'model': str(args.model or cfg.model_path),
        'deployed_threshold': deployed,
        'knn_k': args.k,
        'classes': predictor.classes_,
        'groups': {g: {'n': groups[g]['n'], 'skipped': len(groups[g]['skipped'])}
                   for g in ('pool', 'syn_known', 'syn_unknown', 'real')},
        'z_stats_calibrated_on': {'source': 'syn_known(val)', 'stats': zref},
        'baseline_maxprob': {
            'real_far_at_deployed_threshold': round(
                sum(1 for v in real_maxprob if v >= deployed) / len(real_maxprob), 4),
            'real_ci95': list(wilson(sum(1 for v in real_maxprob if v >= deployed),
                                     len(real_maxprob))),
            'syn_known_coverage_at_deployed_threshold': round(
                sum(1 for v in known_maxprob if v >= deployed) / len(known_maxprob), 4),
            'syn_unknown_far_at_deployed_threshold': round(
                sum(1 for v in unk_maxprob if v >= deployed) / len(unk_maxprob), 4),
            'syn_known_accuracy': round(
                sum(1 for c in correct if c) / len(correct), 4) if correct else None,
            'ece_on_syn_known': ece(known_maxprob, correct),
        },
        # 部署单阈值的危险误动作率，和 AND 规则同一口径，报告里要做同轴对比
        'deployed_dangerous': dangerous_split(deployed_cmds, len(real_maxprob)),
        'signal_order': list(SIGNAL_ORDER),
        'signals': per_signal,
        'and_rules': and_results,
        'false_accept_by_command_at_deployed': dict(sorted(deployed_cmds.items(),
                                                           key=lambda kv: -kv[1])),
        'caveats': [
            '正样本全部是合成 TTS，覆盖率一侧只代表合成命令，真人命令覆盖率未知（真人正样本仍为 0）。',
            '跨域 AUC 远高于同域 AUC 的信号是域检测器，它的跨域曲线不能拿去承诺真人 FAR。',
            'oracle_on_real 的阈值直接在真实测试集上搜出来，是能力上界，**不可部署**。',
            'kNN 参照池来自合成 train 嵌入，真人命令语音到该池的距离未测，可能同样偏远而被误拒。',
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding='utf-8')

    brief = {'baseline': report['baseline_maxprob'],
             'signals': {n: {'auc_cross': s['auc_cross_domain'],
                             'auc_same': s['auc_same_domain'],
                             'verdict': (s['domain_gap'] or {}).get('verdict'),
                             'aurc_cross': s['aurc_cross_domain'],
                             'wp@0.10': next((w for w in s['workpoints_cross_domain']
                                              if w['target_risk'] == 0.10), None)}
                         for n, s in per_signal.items()},
             'and_rules': and_results}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print('已写入:', args.out)
    return report


if __name__ == '__main__':
    main()

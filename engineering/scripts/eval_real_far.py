# -*- coding: utf-8 -*-
"""真实域误接受率（FAR）：真人粤语音频灌入部署管线，量「不该动却动了」的比例。

与合成 unknown 集的区别：合成 unknown 是 edge-tts 念的非命令行，声学域与训练集
同源，FAR 天然偏乐观；真人语料才是真正的域外负样本。这是 v5 的核心新指标。

三件事必须一起给，单给一个 FAR 数字会误导：
1. Wilson 95% 置信区间——几十条样本测出的 FAR，区间可能宽到盖住目标值；
2. 误接受落到哪个命令——决定「危险动作」概率，不只是「有没有误触发」；
3. FAR-阈值反解——先定目标 FAR 再反解阈值，并给出该阈值下合成已知命令的
   覆盖率代价；不算代价的降 FAR 都是白说。

示例：
    python scripts/eval_real_far.py \
        --manifest data/real_domain_testset/manifest.csv \
        --coverage-manifest "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" \
        --coverage-subset val \
        --out docs/real_far.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.realtime import DialectEngine
from src.vad import EnergyVad
from runtime.audio_io import load_wav

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_GRID = tuple(round(1.0 - 0.05 * i, 2) for i in range(20))  # 1.00 → 0.05


def wilson(k: int, n: int, z: float = 1.96):
    """二项比例的 Wilson 区间；小样本下比正态近似可靠，不会给出负下界。"""
    if n == 0:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def summarize(confs, threshold):
    """给定置信度列表与阈值，返回 FAR 与区间。"""
    n = len(confs)
    k = int(sum(1 for c in confs if c >= threshold))
    lo, hi = wilson(k, n)
    return {
        'n': n,
        'accepted': k,
        'far': round(k / n, 4) if n else None,
        'ci95': [lo, hi],
    }


def percentiles(confs):
    if not confs:
        return {}
    a = np.asarray(confs, dtype=np.float64)
    out = {f'p{q}': round(float(np.percentile(a, q)), 4) for q in (50, 90, 95, 99)}
    out['max'] = round(float(a.max()), 4)
    out['mean'] = round(float(a.mean()), 4)
    return out


def far_at(a_sorted_asc: np.ndarray, tau: float) -> float:
    """阈值 tau 下的 FAR；a_sorted_asc 升序，用 searchsorted 避免 O(n) 扫描。"""
    if a_sorted_asc.size == 0:
        return 0.0
    idx = int(np.searchsorted(a_sorted_asc, tau, side='left'))
    return (a_sorted_asc.size - idx) / a_sorted_asc.size


THRESHOLD_GRID = 1e4  # config.yaml 里阈值写到小数点后 4 位，反解也只能落在这个网格上


def _grid_candidates(a_sorted: np.ndarray):
    """阈值候选：每个观测值对应「>= 它」和「严格 > 它」两个网格点。

    严格大于是必须的——要把置信度恰好等于 v 的样本全排除掉，阈值就得高于 v，
    只拿观测值当候选会漏掉这一档，导致反解结果达不到目标 FAR。
    """
    vals = set()
    for v in np.unique(a_sorted):
        step = math.ceil(round(float(v) * THRESHOLD_GRID, 6)) / THRESHOLD_GRID
        vals.add(round(step, 4))
        vals.add(round(step + 1.0 / THRESHOLD_GRID, 4))
    return sorted(x for x in vals if x <= 1.0)


def solve_thresholds(confs, targets):
    """反解：达到目标 FAR 所需的最小可部署阈值（判据是 confidence >= threshold）。

    上限锁死 1.0。置信度是概率，>1.0 等于「一律拒识」，不是可部署工作点；
    真到那一步就如实标 unreachable，别报一个假阈值。
    返回的 far_at_threshold 按返回的 threshold 重算，两者自洽。
    """
    if not confs:
        return []
    a = np.sort(np.asarray(confs, dtype=np.float64))
    candidates = _grid_candidates(a)
    results = []
    for target in targets:
        hit = next((t for t in candidates if far_at(a, t) <= target + 1e-12), None)
        item = {
            'target_far': target,
            'threshold': hit,
            'far_at_threshold': None if hit is None else round(far_at(a, hit), 4),
            'reachable': hit is not None,
        }
        if hit is None:
            item['reason'] = ('只有把阈值抬到 1.0 以上（一律拒识）才能达标，'
                              '不是可部署工作点；需要补硬负样本重训，不是调参')
        results.append(item)
    return results


def far_curve(confs, thresholds):
    a = np.sort(np.asarray(confs, dtype=np.float64)) if confs else np.array([])
    return [{'threshold': round(t, 3), 'far': round(far_at(a, t), 4)}
            for t in thresholds]


def coverage_at(a_sorted_asc: np.ndarray, tau: float) -> float:
    return far_at(a_sorted_asc, tau)  # 同一算子：已知命令的「接受率」就是覆盖率


def risk_coverage(real_confs, known_confs):
    """跨域风险-覆盖：覆盖率取合成已知命令的接受率，风险取真实域 FAR。

    这是把两条不同来源的曲线按阈值对齐，不是同一批数据上的选择性分类曲线；
    报告里必须这么写，否则会被误读成标准 AURC。
    """
    if not real_confs or not known_confs:
        return None
    real = np.sort(np.asarray(real_confs, dtype=np.float64))
    known = np.sort(np.asarray(known_confs, dtype=np.float64))
    lo, hi = float(known[0]), float(known[-1])
    taus = np.linspace(lo, hi, 60).tolist() + [hi + 1e-3]
    rows = [{'threshold': round(t, 4),
             'coverage': round(coverage_at(known, t), 4),
             'far': round(far_at(real, t), 4)} for t in taus]
    aurc, used = [], 0
    for level in COVERAGE_GRID:
        ok = [r for r in rows if r['coverage'] <= level + 1e-9]
        if not ok:
            continue
        # 覆盖率随阈值单调下降，所以「满足 coverage<=level 且覆盖率最高」那点
        # 才是真正交付 level 覆盖率的工作点；取覆盖率最低那点会让每档都落到
        # FAR=0，AURC 恒等于 0（踩过的坑）。
        # 覆盖率是阶梯函数，同一覆盖率可能对应多个阈值。风险同样随阈值单调
        # **下降**，所以并列时取阈值最高那点——覆盖率一样、FAR 更低，严格占优。
        # 早先按 curve 顺序取第一个并列点，读到的是平台上最松的阈值，AURC 偏高。
        aurc.append(max(ok, key=lambda r: (r['coverage'], r['threshold']))['far'])
        used += 1
    return {
        'curve': rows,
        'aurc_cross_domain': round(sum(aurc) / len(aurc), 4) if aurc else None,
        'coverage_levels_used': used,
        'note': 'coverage 来自合成已知命令，far 来自真实域负样本，按阈值对齐；'
                '非同一数据集上的标准 AURC。',
    }


def dangerous_split(accepted_cmds, n):
    """把误接受拆成「会真的动设备」和「只是查询/无动作」。

    STATUS_ALL 的 actuator 是 none，误触发只是播一句状态，不会关东西；
    ALL_OFF 误触发是真的把全屋关掉。两者的安全含义差一个量级，
    合在一个 FAR 里报会把真正危险的部分稀释掉。
    """
    from src.commands import COMMANDS
    dangerous = {c: k for c, k in accepted_cmds.items()
                 if COMMANDS.get(c, {}).get('actuator', 'none') != 'none'}
    benign = {c: k for c, k in accepted_cmds.items() if c not in dangerous}
    kd, kb = sum(dangerous.values()), sum(benign.values())
    return {
        'dangerous': {'accepted': kd, 'rate': round(kd / n, 4) if n else None,
                      'ci95': list(wilson(kd, n)), 'by_command': dict(
                          sorted(dangerous.items(), key=lambda kv: -kv[1]))},
        'benign': {'accepted': kb, 'rate': round(kb / n, 4) if n else None,
                   'by_command': dict(sorted(benign.items(), key=lambda kv: -kv[1]))},
    }


def load_rows(manifest: Path, subset: str, window=None):
    rows = []
    with manifest.open(encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            if subset and row.get('split') != subset:
                continue
            if not Path(row['audio_path']).exists():
                continue
            if window:
                ms = float(row.get('duration_ms') or 0.0)
                if not (window[0] <= ms <= window[1]):
                    continue
            rows.append(row)
    return rows


def collect_confidences(engine, paths, cfg, use_vad=True):
    """逐文件跑部署管线，返回每条的置信度与判定，便于按来源分组复用同一次推理。"""
    out = []
    for path in paths:
        audio, sr = load_wav(path)
        if use_vad:
            vad = EnergyVad(fs=sr, **cfg.vad_kwargs())
            segments = vad.feed(audio) + vad.flush()
            if not segments:
                out.append({'path': path, 'skipped': 'vad_no_speech'})
                continue
            audio = segments[0]
        result = engine.process_utterance(audio, sr)
        if result.reason.startswith(('empty_audio', 'sample_rate_mismatch')):
            out.append({'path': path, 'skipped': result.reason})
            continue
        out.append({'path': path, 'skipped': None,
                    'confidence': float(result.confidence),
                    'command': result.command,
                    'accepted': bool(result.accepted),
                    'latency_ms': float(result.latency_ms)})
    return out


def confs_of(records):
    return [r['confidence'] for r in records if not r['skipped']]


def synthetic_unknown_far(path: Path):
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding='utf-8'))
    entry = report.get('variants', {}).get('with_vad', report)
    unk = entry.get('unknown') or {}
    return {'far': unk.get('false_accept_rate'), 'n': unk.get('n'),
            'source': str(path.name)}


def load_provenance(manifest: Path):
    """带上构建时的许可与筛选口径。

    CC BY 4.0 要求署名跟着数据走；FAR 数字一旦进报告，许可与「有没有按响度挑样本」
    必须同时可见，否则读者无法判断这个数字能不能用、是不是被筛低了。
    """
    path = manifest.parent / 'provenance.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    filters = data.get('filters') or {}
    return {
        'built': data.get('built'),
        'vad_kwargs': data.get('vad_kwargs'),
        'text_filter': filters.get('text_filter'),
        'exclude_phrases': filters.get('exclude_phrases'),
        'min_ms': filters.get('min_ms'),
        'min_rms_db': filters.get('min_rms_db'),
        'drop_clipped': filters.get('drop_clipped'),
        'selection_note': data.get('selection_note'),
        'excluded': data.get('excluded'),
        'segments_text_verified': data.get('segments_text_verified'),
        'segments_with_device_hint': data.get('segments_with_device_hint'),
        'sources': {name: {'license': s.get('license'), 'path': s.get('path'),
                           'segments': s.get('segments'),
                           'files_scanned': s.get('files_scanned')}
                    for name, s in (data.get('sources') or {}).items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True,
                        help='build_real_testset.py 产出的 manifest.csv')
    parser.add_argument('--subset', default='real_test')
    parser.add_argument('--window-ms', default=None,
                        help='只评估这个时长窗内的片段，如 400-4000（命令长度域）')
    parser.add_argument('--no-vad', action='store_true',
                        help='不再过一遍 VAD（测试集片段已是 VAD 输出时用得到）')
    parser.add_argument('--targets', default='0.10,0.05,0.02',
                        help='FAR 反解目标，逗号分隔')
    parser.add_argument('--coverage-manifest', default=None,
                        help='合成已知命令 manifest，用于算阈值上调的覆盖率代价')
    parser.add_argument('--coverage-subset', default='val')
    parser.add_argument('--coverage-limit', type=int, default=0)
    parser.add_argument('--config', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--thresholds', default=None)
    parser.add_argument('--synthetic-eval', default=str(ROOT / 'docs' / 'pipeline_eval.json'))
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'real_far.json'))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    engine = DialectEngine.load(str(args.model or cfg.model_path),
                                str(args.thresholds or cfg.thresholds_path))
    threshold = float(engine.threshold)
    window = None
    if args.window_ms:
        lo, hi = args.window_ms.split('-', 1)
        window = (float(lo), float(hi))

    rows = load_rows(Path(args.manifest), args.subset, window)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit(f'manifest 中没有 {args.subset} 子集的可用片段: {args.manifest}')

    paths = [Path(r['audio_path']) for r in rows]
    records = collect_confidences(engine, paths, cfg, use_vad=not args.no_vad)
    confs = confs_of(records)
    skipped = sum(1 for r in records if r['skipped'])
    accepted_cmds = Counter(r['command'] for r in records
                            if not r['skipped'] and r['accepted'])
    latencies = [r['latency_ms'] for r in records if not r['skipped']]

    source_of = {Path(r['audio_path']): r.get('source_corpus', '?') for r in rows}
    source_stats = {}
    for name in sorted(set(source_of.values())):
        sub = [rec for rec in records if source_of.get(rec['path']) == name]
        source_stats[name] = dict(
            summarize(confs_of(sub), threshold),
            skipped=sum(1 for rec in sub if rec['skipped']))

    targets = [float(t) for t in args.targets.split(',') if t.strip()]
    solved = solve_thresholds(confs, targets)

    known_confs = []
    if args.coverage_manifest:
        crows = load_rows(Path(args.coverage_manifest), args.coverage_subset)
        if args.coverage_limit:
            crows = crows[:args.coverage_limit]
        if crows:
            known_confs = confs_of(collect_confidences(
                engine, [Path(r['audio_path']) for r in crows], cfg,
                use_vad=not args.no_vad))

    curve_thresholds = sorted({round(threshold, 3)}
                              | {r['threshold'] for r in solved if r['threshold']}
                              | {round(0.80 + 0.02 * i, 3) for i in range(11)})
    report = {
        'generated': date.today().isoformat(),
        'model': str(args.model or cfg.model_path),
        'threshold': threshold,
        'classes': engine.classes_,
        'testset': {
            'manifest': str(args.manifest),
            'subset': args.subset,
            'window_ms': list(window) if window else None,
            'rows': len(rows),
            'evaluated': len(confs),
            'skipped': skipped,
            'total_duration_s': round(sum(float(r.get('duration_ms') or 0)
                                          for r in rows) / 1000.0, 1),
            'per_source': source_stats,
            'provenance': load_provenance(Path(args.manifest)),
        },
        'headline': dict(summarize(confs, threshold),
                         mean_latency_ms=round(sum(latencies) / len(latencies), 1)
                         if latencies else None),
        'false_accept_by_command': dict(sorted(accepted_cmds.items(),
                                               key=lambda kv: -kv[1])),
        'dangerous_split': dangerous_split(accepted_cmds, len(confs)),
        'confidence_percentiles': percentiles(confs),
        'threshold_solve': solved,
        'far_curve': far_curve(confs, curve_thresholds),
        'risk_coverage': risk_coverage(confs, known_confs),
        'synthetic_unknown_far': synthetic_unknown_far(Path(args.synthetic_eval)),
        'note': '真实域 FAR 只说明「非命令真人语音被误接受的比例」，'
                '不代表真人命令识别率；后者需要真人正样本，尚未采集。',
    }

    head = report['headline']
    dang = report['dangerous_split']['dangerous']
    print(json.dumps({'n': head['n'], 'far': head['far'], 'ci95': head['ci95'],
                      'threshold': threshold,
                      'dangerous_rate': dang['rate'], 'dangerous_ci95': dang['ci95'],
                      'dangerous_by_command': dang['by_command'],
                      'false_accept_by_command': report['false_accept_by_command'],
                      'threshold_solve': solved}, ensure_ascii=False, indent=2))
    if report['synthetic_unknown_far']:
        print('合成 unknown FAR（对比）:', report['synthetic_unknown_far'])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding='utf-8')
    print('已写入:', args.out)
    return report


if __name__ == '__main__':
    main()

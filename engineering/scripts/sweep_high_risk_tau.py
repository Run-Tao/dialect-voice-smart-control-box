# -*- coding: utf-8 -*-
"""用当前部署模型(v32b)的 raw + 新参照池重扫 ALL_OFF 高风险 ood 门槛 tau。

背景（项目审查 2026-11-19）：
    config.yaml 的 high_risk_min_ood.ALL_OFF = -0.014 是旧模型 embedding 下标定的
    （注释写"54/2637→0/2637"）。切到 v32b 后 embedding 空间已变，门限 ood 从
    -0.022306 移到 -0.026796，该二次门槛原样保留，在新空间下未验证。

口径（与 runtime/selective.py + runtime/decision.py 逐字一致）：
    ood = 到下发参照池的 kNN 余弦距离取负（k=5）；
    主门限 = maxprob >= gate.maxprob AND ood >= gate.ood；
    高风险规则 = top1=='ALL_OFF' 且 ood < tau 时→转确认（不直接执行）。

对每组数据，先在"通过主门限 且 top1==ALL_OFF"的样本上扫 tau：
    - 真人非命令(real)：这些是"误动作"，被 tau 拦下越多越安全；
    - 合成已知命令(syn_known)中 top1==ALL_OFF 的：这些是"正确执行"，被拦下是代价。

用法：
    python scripts/sweep_high_risk_tau.py \
      --raw ../DialectVoice_Research/data/real_domain_testset/selective_raw_v32b.npz \
      --gate runtime/models/v3_selective_gate.json \
      --pool runtime/models/v3_knn_pool.npz \
      --out docs/high_risk_tau_v32b.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.selective import knn_distance, l2_normalize


def read_classes(model_path):
    """类别顺序必须从模型 meta 读取，写死会错位（ALL_OFF 实际是索引 4）。"""
    with np.load(str(model_path), allow_pickle=False) as z:
        meta = json.loads(z['__meta__'].item())
    return list(meta['classes'])


def load_pool_embeddings(pool_path):
    with np.load(str(pool_path), allow_pickle=False) as z:
        emb = np.asarray(z['embedding'], dtype=np.float32)
        # 池文件可能是归一化前的裸向量；knn_distance 内部会 l2_normalize query，
        # SelectiveGate 用 pool @ e（pool 未归一化则尺度不同）——这里与 gate 一致：
        # 用与 SelectiveGate 相同的处理。先看 gate 是否对 pool 归一化。
    return emb


def compute(group, pool, k, gate_mp, gate_ood, classes):
    """返回该组里 '通过主门限 且 top1==ALL_OFF' 样本的 ood 列表与全部 all_off 数。"""
    prob = np.asarray(group['prob'], dtype=np.float64)
    emb = np.asarray(group['embedding'], dtype=np.float32)
    cmd = group['command']
    oods = []
    n_top1_alloff = 0
    n_pass_main = 0
    for i in range(len(prob)):
        p = prob[i] / prob[i].sum()
        top1 = classes[int(np.argmax(p))]
        if top1 != 'ALL_OFF':
            continue
        n_top1_alloff += 1
        d = knn_distance(emb[i], pool, k=k)
        ood = -d if d is not None else None
        if ood is None or float(p.max()) < gate_mp or ood < gate_ood:
            continue
        n_pass_main += 1
        oods.append(float(ood))
    return np.asarray(oods), n_top1_alloff, n_pass_main


def sweep(oods_real, oods_syn, taus):
    rows = []
    for tau in taus:
        # ood < tau -> 被拦下转确认
        real_blocked = int(np.sum(oods_real < tau)) if len(oods_real) else 0
        syn_blocked = int(np.sum(oods_syn < tau)) if len(oods_syn) else 0
        rows.append({
            'tau': round(float(tau), 4),
            'real_all_off_wrong_passing_main': len(oods_real),
            'real_blocked': real_blocked,
            'real_remaining_direct': len(oods_real) - real_blocked,
            'syn_all_off_correct_passing_main': len(oods_syn),
            'syn_blocked_cost': syn_blocked,
            'syn_remaining_direct': len(oods_syn) - syn_blocked,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=str(ROOT.parent / 'DialectVoice_Research' / 'data' /
                                         'real_domain_testset' / 'selective_raw_v32b.npz'))
    ap.add_argument('--gate', default=str(ROOT / 'runtime' / 'models' / 'v3_selective_gate.json'))
    ap.add_argument('--pool', default=str(ROOT / 'runtime' / 'models' / 'v3_knn_pool.npz'))
    ap.add_argument('--model', default=str(ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_v32b.npz'))
    ap.add_argument('--out', default=str(ROOT / 'docs' / 'high_risk_tau_v32b.json'))
    ap.add_argument('--tau-min', type=float, default=-0.20)
    ap.add_argument('--tau-max', type=float, default=0.10)
    ap.add_argument('--tau-step', type=float, default=0.005)
    args = ap.parse_args()

    with open(args.gate, encoding='utf-8') as f:
        gate = json.load(f)
    th = gate['thresholds']
    gate_mp, gate_ood = float(th['maxprob']), float(th['ood'])
    k = int(gate.get('ood', {}).get('k', 5))
    pool = load_pool_embeddings(args.pool)
    classes = read_classes(args.model)
    assert len(classes) == 6 and classes[4] == 'ALL_OFF', f'类别顺序异常: {classes}'

    z = np.load(args.raw, allow_pickle=False)
    real = {k2: z['real/' + k2] for k2 in ('prob', 'embedding', 'command')}
    syn = {k2: z['syn_known/' + k2] for k2 in ('prob', 'embedding', 'command')}

    oods_real, n_real_alloff, n_real_pass = compute(real, pool, k, gate_mp, gate_ood, classes)
    oods_syn, n_syn_alloff, n_syn_pass = compute(syn, pool, k, gate_mp, gate_ood, classes)

    taus = np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)
    rows = sweep(oods_real, oods_syn, taus)

    # ood < tau → 拦下转确认；tau 越大拦得越多。
    # 推荐 = 真人误动作全部拦下(remaining=0)的前提下 tau 最小者（拦得最少、合成误拦代价最低）。
    safe = [r for r in rows if r['real_remaining_direct'] == 0]
    rec = min(safe, key=lambda r: r['tau']) if safe else min(rows, key=lambda r: (r['real_remaining_direct'], -r['tau']))

    report = {
        'generated_by': 'scripts/sweep_high_risk_tau.py',
        'model': 'v3_cnn_ensemble_v32b.npz',
        'gate_thresholds': th,
        'k': k,
        'classes': classes,
        'pool_shape': list(pool.shape),
        'old_tau': -0.014,
        'real_top1_alloff': n_real_alloff,
        'real_passing_main': n_real_pass,
        'real_oods_stats': {
            'min': float(oods_real.min()) if len(oods_real) else None,
            'p10': float(np.percentile(oods_real, 10)) if len(oods_real) else None,
            'median': float(np.median(oods_real)) if len(oods_real) else None,
            'max': float(oods_real.max()) if len(oods_real) else None,
        } if len(oods_real) else {},
        'syn_top1_alloff': n_syn_alloff,
        'syn_passing_main': n_syn_pass,
        'sweep': rows,
        'recommendation': rec,
        'note': 'real=真人非命令误动作(越低越安全); syn=合成已知正确执行(被拦是代价)。'
                'tau 越大拦得越多。口径与 runtime/selective.py + decision.py 一致。',
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'真人 top1=ALL_OFF 共 {n_real_alloff}，过主门限 {n_real_pass}；'
          f'合成 top1=ALL_OFF 共 {n_syn_alloff}，过主门限 {n_syn_pass}')
    if len(oods_real):
        print(f'真人 ood 分布: min={oods_real.min():.4f} p10={np.percentile(oods_real,10):.4f} '
              f'median={np.median(oods_real):.4f} max={oods_real.max():.4f}')
    print('推荐 tau:', rec)
    print('报告:', args.out)


if __name__ == '__main__':
    main()

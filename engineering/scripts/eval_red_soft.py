# -*- coding: utf-8 -*-
"""评估红区近门限澄清的收益与误澄清风险。

红区恢复只使用现有缓存，不再跑模型。它检查：

1. 已知命令中，有多少红区样本会进入澄清，二选一里是否包含真命令；
2. 合成未知指令和真人非命令中，有多少会被误当成“值得再问一句”。

注意：误澄清不会执行动作，但会增加打扰，因此必须与已知命令回收率一起
报告。若真人负样本误澄清率过高，应保持 red_soft 关闭。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.selective import SelectiveGate
from src.commands import COMMANDS

CLASSES = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN',
           'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']
DEFAULT_PAIRS = [('CURTAIN_OPEN', 'CURTAIN_CLOSE'),
                 ('ALL_OFF', 'LIGHT_OFF')]


def pair_for(command: str, pairs: Sequence[Sequence[str]]):
    for pair in pairs:
        members = [str(x) for x in pair]
        if command in members and len(members) >= 2:
            return members
    return None


def near_miss(signals, thresholds, tolerances) -> bool:
    fails = []
    for name, threshold in thresholds.items():
        value = signals.get(name)
        if value is None:
            return False
        deficit = float(threshold) - float(value)
        if deficit > 0:
            fails.append((name, deficit))
    if len(fails) != 1:
        return False
    name, deficit = fails[0]
    return bool(deficit <= float(tolerances.get(name, 0.0)) + 1e-12)


def evaluate_group(z, gate: SelectiveGate, group: str, pairs, tolerances,
                   min_prob: float, min_margin: float) -> dict:
    probs = z[f'{group}/prob']
    emb = z[f'{group}/embedding']
    labels = z[f'{group}/label']

    accepted = 0
    triggered = 0
    recoverable = 0
    correct_candidate = 0
    examples = []
    for i in range(len(probs)):
        p = np.asarray(probs[i], dtype=np.float64)
        sig = gate.signals_of(p, emb[i])
        ok = all(float(sig.get(k)) >= float(t)
                 for k, t in gate.thresholds.items())
        accepted += int(ok)
        if ok or not near_miss(sig, gate.thresholds, tolerances):
            continue
        order = np.argsort(-p)
        top1 = CLASSES[int(order[0])]
        top2 = CLASSES[int(order[1])]
        margin = float(p[order[0]] - p[order[1]])
        cand = pair_for(top1, pairs)
        if (cand is None or float(p[order[0]]) < min_prob
                or margin < min_margin):
            continue
        triggered += 1
        label = str(labels[i]) if labels[i] else ''
        hit = bool(label and label in cand)
        correct_candidate += int(hit)
        recoverable += int(hit)
        if len(examples) < 10:
            examples.append({
                'i': i, 'top1': top1, 'top2': top2,
                'label': label, 'candidate_contains_truth': hit,
                'maxprob': round(float(sig['maxprob']), 4),
                'ood': round(float(sig['ood']), 4),
                'margin': round(margin, 4),
            })
    n = len(probs)
    return {
        'n': n,
        'gate_accepted': accepted,
        'gate_acceptance': round(accepted / n, 4) if n else None,
        'red_soft_triggered': triggered,
        'red_soft_trigger_rate': round(triggered / n, 4) if n else None,
        'candidate_contains_truth': correct_candidate,
        'candidate_hit_given_trigger': (
            round(correct_candidate / triggered, 4) if triggered else None),
        'examples': examples,
    }


def evaluate_high_risk_group(z, gate: SelectiveGate, group: str,
                             command: str, min_ood: float) -> dict:
    """评估高风险命令额外 OOD 边界对“直接执行”的影响。"""
    probs = z[f'{group}/prob']
    emb = z[f'{group}/embedding']
    accepted = 0
    dangerous_direct = 0
    high_risk_accepted = 0
    blocked_to_confirm = 0
    for i in range(len(probs)):
        p = np.asarray(probs[i], dtype=np.float64)
        sig = gate.signals_of(p, emb[i])
        if not all(float(sig.get(k)) >= float(t)
                   for k, t in gate.thresholds.items()):
            continue
        accepted += 1
        top1 = CLASSES[int(np.argmax(p))]
        actuator = (COMMANDS.get(top1) or {}).get('actuator', 'none')
        dangerous_direct += int(actuator != 'none')
        if top1 == command:
            high_risk_accepted += 1
            if float(sig['ood']) < min_ood:
                blocked_to_confirm += 1
    effective_dangerous = dangerous_direct - blocked_to_confirm
    return {
        'n': len(probs),
        'gate_accepted': accepted,
        'dangerous_direct': dangerous_direct,
        'dangerous_direct_rate': (
            round(dangerous_direct / len(probs), 4) if len(probs) else None),
        'high_risk_accepted': high_risk_accepted,
        'blocked_to_confirm': blocked_to_confirm,
        'effective_dangerous_direct': effective_dangerous,
        'effective_dangerous_rate': (
            round(effective_dangerous / len(probs), 4) if len(probs) else None),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw-cache',
        default=str(ROOT.parent / 'DialectVoice_Research' / 'data'
                    / 'real_domain_testset' / 'selective_raw.npz'))
    parser.add_argument(
        '--gate', default=str(ROOT / 'runtime' / 'models'
                              / 'v3_selective_gate.json'))
    parser.add_argument(
        '--pool', default=str(ROOT / 'runtime' / 'models'
                              / 'v3_knn_pool.npz'))
    parser.add_argument('--min-prob', type=float, default=0.95)
    parser.add_argument('--min-margin', type=float, default=0.5)
    parser.add_argument('--maxprob-tolerance', type=float, default=0.01)
    parser.add_argument('--ood-tolerance', type=float, default=0.001)
    parser.add_argument('--high-risk-command', default='ALL_OFF')
    parser.add_argument('--high-risk-min-ood', type=float, default=-0.014)
    parser.add_argument('--out', default=str(ROOT / 'docs' / 'red_soft_eval.json'))
    args = parser.parse_args(argv)

    gate = SelectiveGate.load(args.gate, args.pool)
    tolerances = {'maxprob': args.maxprob_tolerance, 'ood': args.ood_tolerance}
    with np.load(args.raw_cache, allow_pickle=False) as z:
        results = {
            group: evaluate_group(z, gate, group, DEFAULT_PAIRS, tolerances,
                                  args.min_prob, args.min_margin)
            for group in ('syn_known', 'syn_unknown', 'real')
        }
        high_risk_results = {
            group: evaluate_high_risk_group(
                z, gate, group, args.high_risk_command, args.high_risk_min_ood)
            for group in ('syn_known', 'syn_unknown', 'real')
        }

    known = results['syn_known']
    unknown = results['syn_unknown']
    real = results['real']
    report = {
        'gate_thresholds': gate.thresholds,
        'pairs': DEFAULT_PAIRS,
        'min_prob': args.min_prob,
        'min_margin': args.min_margin,
        'tolerances': tolerances,
        'groups': results,
        'high_risk': {
            'command': args.high_risk_command,
            'min_ood': args.high_risk_min_ood,
            'groups': high_risk_results,
        },
        'decision': {
            'known_recovery_rate': round(
                known['candidate_contains_truth'] / known['n'], 4),
            'unknown_false_prompt_rate': unknown['red_soft_trigger_rate'],
            'real_noncommand_false_prompt_rate': real['red_soft_trigger_rate'],
            'recommended': '需要同时看已知回收率与真人误澄清率，不能只追求前者',
        },
        'note': '误澄清只播报问句，不执行动作；但会打扰用户。'
                'syn_known 是已知命令验证组，real 是公开真人粤语非命令。',
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'groups'},
                     ensure_ascii=False, indent=2))
    print('已写入:', out)


if __name__ == '__main__':
    main()

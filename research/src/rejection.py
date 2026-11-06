# -*- coding: utf-8 -*-
"""置信度阈值校准与拒识评价。"""
import numpy as np


def threshold_table(confidences, correct, thresholds=None):
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    if len(conf) != len(corr):
        raise ValueError('confidences和correct长度不一致')
    if thresholds is None:
        thresholds = np.unique(np.round(np.linspace(0, 1, 101), 3))
    rows = []
    for t in thresholds:
        accepted = conf >= t
        coverage = float(np.mean(accepted)) if len(conf) else 0.0
        acc = float(np.mean(corr[accepted])) if accepted.any() else 0.0
        rows.append({'threshold': float(t), 'coverage': coverage, 'accepted_accuracy': acc})
    return rows


def calibrate_threshold(confidences, correct, min_coverage=0.8):
    rows = threshold_table(confidences, correct)
    feasible = [r for r in rows if r['coverage'] >= min_coverage]
    if not feasible:
        feasible = rows
    # 先保覆盖，再选接受样本准确率；同分时取更高覆盖
    best = max(feasible, key=lambda r: (r['accepted_accuracy'], r['coverage']))
    return best


def rejection_metrics(known_conf, known_correct, unknown_conf, threshold):
    known_conf = np.asarray(known_conf, dtype=float)
    known_correct = np.asarray(known_correct, dtype=bool)
    unknown_conf = np.asarray(unknown_conf, dtype=float)
    known_acc = known_conf >= threshold
    unknown_acc = unknown_conf >= threshold
    return {
        'threshold': float(threshold),
        'known_coverage': float(np.mean(known_acc)) if len(known_conf) else 0.0,
        'known_accepted_accuracy': float(np.mean(known_correct[known_acc])) if known_acc.any() else 0.0,
        'unknown_false_accept_rate': float(np.mean(unknown_acc)) if len(unknown_conf) else 0.0,
        'unknown_rejection_rate': float(1.0 - np.mean(unknown_acc)) if len(unknown_conf) else 1.0,
    }


def calibrate_threshold_with_negatives(known_conf, known_correct, negative_conf,
                                       target_far=0.1, min_coverage=0.2):
    """在已知验证集和未知校准集上选阈值，优先控制未知误接受率。"""
    known_conf = np.asarray(known_conf, dtype=float)
    known_correct = np.asarray(known_correct, dtype=bool)
    negative_conf = np.asarray(negative_conf, dtype=float)
    rows = threshold_table(known_conf, known_correct)
    for r in rows:
        t = r['threshold']
        r['unknown_false_accept_rate'] = float(np.mean(negative_conf >= t)) if len(negative_conf) else 0.0
    feasible = [
        r for r in rows
        if r['unknown_false_accept_rate'] <= target_far and r['coverage'] >= min_coverage
    ]
    if feasible:
        # 先保覆盖，再看接受准确率；同等覆盖取更低FAR
        best = max(feasible, key=lambda r: (r['coverage'], r['accepted_accuracy'], -r['unknown_false_accept_rate']))
    else:
        best = min(rows, key=lambda r: (r['unknown_false_accept_rate'], -r['coverage']))
    return best

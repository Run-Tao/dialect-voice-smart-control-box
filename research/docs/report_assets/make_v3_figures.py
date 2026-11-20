# -*- coding: utf-8 -*-
"""从 results.json 生成报告/汇报用的关键图（matplotlib，中文标注）。

产物（docs/report_assets/）：
    fig_main_results.png   E1–E5 主结果柱图（full 483 / matched-v2 111）
    fig_confusion.png      E4 最终模型混淆矩阵热图
    fig_stratified.png     条件 / 语域分层准确率

用法::

    D:\\anaconda3\\python.exe docs/report_assets/make_v3_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs' / 'v3_experiments' / 'main_v3_1'
ASSETS = Path(__file__).resolve().parent

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

INTENTS_CN = ['开灯', '关灯', '开窗帘', '关窗帘', '全屋关', '查状态']
LABELS = {'E1_v2_dtw_baseline': 'E1 二期 DTW', 'E2_classical_logreg': 'E2 经典 logreg',
          'E3_cnn_ensemble': 'E3 CNN 集成', 'E4_cnn_tta': 'E4 CNN+TTA', 'E5_fusion': 'E5 融合'}
KEYS = list(LABELS)


def load_results():
    p = OUT / 'results.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def _matched_acc(results, key):
    """E4 的 matched-v2 未存进 results.json，从预测 CSV 复算（与报告表 8 口径一致）。"""
    m = (results[key].get('matched_v2_test') or {}).get('accuracy')
    if m is not None:
        return m
    if key == 'E4_cnn_tta':
        import csv as _csv
        path = OUT / 'predictions_E4_cnn_tta_final.csv'
        if path.exists():
            with open(path, encoding='utf-8', newline='') as f:
                rows = list(_csv.DictReader(f))
            v2 = {'text_morphology', 'rate_slow', 'legacy', 'baseline_v1', ''}
            sel = [r for r in rows if r['condition'] in v2]
            if sel:
                return sum(int(r['correct']) for r in sel) / len(sel)
    return None


def fig_main(results):
    rows = []
    for k in KEYS:
        if k not in results:
            continue
        rows.append((LABELS[k], results[k]['full_test']['accuracy'], _matched_acc(results, k)))
    if not rows:
        return None
    names = [r[0] for r in rows]
    full = [r[1] for r in rows]
    matched = [float('nan') if r[2] is None else r[2] for r in rows]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    b1 = ax.bar(x - w / 2, full, w, label='Test Acc（full 483）', color='#2E5C9A')
    b2 = ax.bar(x + w / 2, matched, w, label='Test Acc（matched-v2 111，与二期可比）', color='#1F7A73')
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('准确率')
    ax.set_title('同一固定测试集上的 E1–E5（数字来源 results.json）', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = ASSETS / 'fig_main_results.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_confusion(results):
    ea = results.get('E10_error_analysis') or {}
    cm = ea.get('confusion_matrix')
    if not cm:
        return None
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=160)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(INTENTS_CN)
    ax.set_yticklabels(INTENTS_CN)
    ax.set_xlabel('预测')
    ax.set_ylabel('真实', labelpad=12)
    ax.set_title('E5 最优模型混淆矩阵（test 483）', fontsize=12)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() * 0.55 else '#333333', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    path = ASSETS / 'fig_confusion.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_stratified(results):
    item = results.get('E5_fusion') or results.get('E4_cnn_tta') or results.get('E3_cnn_ensemble') or {}
    strat = (item.get('full_test') or {}).get('stratified_accuracy') or {}
    cond = strat.get('condition') or {}
    reg = strat.get('register') or {}
    if not cond and not reg:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    for ax, data, title in [(axes[0], cond, '按声学条件'), (axes[1], reg, '按语域')]:
        names = list(data.keys())
        vals = [data[n]['accuracy'] for n in names]
        ns = [data[n]['n'] for n in names]
        bars = ax.bar(range(len(names)), vals, color='#1F7A73')
        for i, (b, n) in enumerate(zip(bars, ns)):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f'{vals[i]:.3f}\n(n={n})', ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('E5 最优模型分层准确率（test 483）', fontsize=12)
    fig.tight_layout()
    path = ASSETS / 'fig_stratified.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_rejection(rej):
    curve = (rej or {}).get('validation_curve') or []
    if not curve:
        return None
    thr = [c['threshold'] for c in curve]
    cov = [c.get('known_coverage') or 0 for c in curve]
    far = [c.get('unknown_false_accept_rate') or 0 for c in curve]
    acc = [c.get('accepted_accuracy') or 0 for c in curve]
    fig, ax = plt.subplots(figsize=(8.8, 4.7), dpi=160)
    ax.plot(thr, cov, color='#2E5C9A', marker='o', ms=3, label='已知命令覆盖率')
    ax.plot(thr, far, color='#C55A11', marker='s', ms=3, label='未知误接受率 FAR')
    ax.plot(thr, acc, color='#1F7A73', marker='^', ms=3, ls='--', label='被接受样本准确率')
    sel = (rej or {}).get('selected_threshold')
    if isinstance(sel, (int, float)):
        ax.axvline(sel, color='#555555', ls=':', lw=1.0)
        ax.text(sel + 0.005, 0.03, f'选中阈值 {sel:.2f}', fontsize=8.5, color='#555555')
    ax.axhline(0.85, color='#999999', lw=0.8, ls='-.')
    ax.text(0.06, 0.855, '覆盖率约束 0.85', fontsize=8, color='#777777')
    ax.axhline(0.08, color='#BBBBBB', lw=0.8, ls='-.')
    ax.text(0.06, 0.085, 'FAR 目标 0.08', fontsize=8, color='#777777')
    ax.set_xlabel('阈值（只在 val 已知命令 + 未知校准集上选择）')
    ax.set_ylabel('比例')
    ax.set_ylim(0, 1.05)
    ax.set_title('拒识：提高阈值能救覆盖率，但 FAR 始终降不到目标（validation_curve）', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='center left', fontsize=9, bbox_to_anchor=(0.42, 0.45))
    fig.tight_layout()
    path = ASSETS / 'fig_rejection_curve.png'
    fig.savefig(path)
    plt.close(fig)
    return path

def fig_train(train_report):
    seeds = (train_report or {}).get('seeds') or []
    if not seeds:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=160)
    for s in seeds:
        hist = s.get('history') or []
        if not hist:
            continue
        ax.plot([h['epoch'] for h in hist], [h['val_accuracy'] for h in hist],
                lw=1.4, label=f"seed={s.get('seed')}（best {s.get('best_val_accuracy'):.4f}）")
    ax.set_xlabel('epoch')
    ax.set_ylabel('验证集准确率')
    ax.set_title('3 种子 CNN 验证集学习曲线（早停看 val acc）', fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    path = ASSETS / 'fig_train_curves.png'
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    results = load_results()
    rejection = None
    rp = OUT / 'rejection.json'
    if rp.exists():
        rejection = json.loads(rp.read_text(encoding='utf-8'))
    train_report = None
    for tp in (OUT / 'train_report.json', OUT.parent / 'train_report.json'):
        if tp.exists():
            train_report = json.loads(tp.read_text(encoding='utf-8'))
            break
    if results is None:
        print('缺少 results.json，跳过主结果/分层/混淆图')
    for fn, arg in ((fig_main, results), (fig_confusion, results),
                    (fig_stratified, results), (fig_rejection, rejection),
                    (fig_train, train_report)):
        p = fn(arg) if arg is not None else None
        print('OK' if p else 'SKIP', p)


if __name__ == '__main__':
    main()






# -*- coding: utf-8 -*-
"""从 v3 实验结果 JSON 生成报告用 Markdown 表格。

用法::

    python scripts/report_v3_tables.py main_v3_1 > outputs/v3_tables.md

只读取结果文件，不重新训练/评估；缺失的段落会自动跳过并标注。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'outputs' / 'v3_experiments'


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def md_table(headers, rows):
    out = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def pct(x):
    return f'{x * 100:.2f}%' if isinstance(x, (int, float)) else '-'


def fmt(x, nd=4):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f'{x:.{nd}f}'
    return '-' if x is None else str(x)


def data_section(run_name):
    audit_v2 = load(BASE / 'outputs' / 'v2_data_audit.json')
    audit_v3 = load(BASE / 'outputs' / 'v3_data_audit.json')
    audit_v31 = load(BASE / 'outputs' / 'v3_1_data_audit.json')
    lines = ['## 1. 数据规模与条件', '']
    rows = []
    for name, audit in [('v2', audit_v2), ('v3', audit_v3), ('v3.1', audit_v31)]:
        if not audit:
            continue
        c = audit.get('counts') or {}
        total = c.get('total_rows') or c.get('total') or '-'
        groups = c.get('num_expression_groups') or c.get('expression_groups') or '-'
        by_split = c.get('by_split') or {}
        rows.append([name, total, by_split.get('train', '-'), by_split.get('val', '-'),
                     by_split.get('test', '-'), groups])
    if rows:
        lines.append(md_table(['数据版本', '总样本', 'train', 'val', 'test', '表达组'], rows))
    if audit_v3:
        cond = (audit_v3.get('counts') or {}).get('by_condition_usable') \
            or (audit_v3.get('counts') or {}).get('by_condition')
        if cond:
            lines += ['', '**v3 条件分布**', '',
                      md_table(['条件', '条数'], sorted(cond.items()))]
    if audit_v31:
        c = audit_v31.get('counts') or {}
        lines += ['', f"**v3.1 增量**：新增 {c.get('v3_1_added_rows', '-')} 行全部 train；"
                      f"correction 组 train {c.get('correction_groups_by_split', {}).get('train', '-')} 个、"
                      f"test {c.get('correction_groups_by_split', {}).get('test', '-')} 个；"
                      f"test 总行数 {c.get('by_split', {}).get('test', '-')}（未改动）。"]
    return '\n'.join(lines)


def main_section(results):
    if not results:
        return '## 2. 模型对比\n\n（缺少 results.json）'
    rows = []
    names = {
        'E1_v2_dtw_baseline': 'E1 v2 基线 MFCC+DTW',
        'E2_classical_logreg': 'E2 经典判别式 logreg',
        'E3_cnn_ensemble': 'E3 CNN 集成（3 seeds）',
        'E4_cnn_tta': 'E4 CNN + TTA（val 选）',
        'E5_fusion': 'E5 CNN+经典 融合（val 选）',
    }
    for key, label in names.items():
        item = results.get(key)
        if not item:
            continue
        full = item.get('full_test') or item.get('test') or {}
        matched = item.get('matched_v2_test') or {}
        val = item.get('val') or {}
        rows.append([label, fmt(full.get('accuracy')), fmt(full.get('macro_f1')),
                     fmt(matched.get('accuracy')),
                     fmt(val.get('accuracy') if val else item.get('val_accuracy')),
                     fmt(full.get('n'))])
    if not rows:
        return '## 2. 模型对比\n\n（无实验结果）'
    out = ['## 2. 模型对比（同一固定 test）', '',
           md_table(['模型', 'Test Acc（full 483）', 'Test Macro-F1',
                     'Test Acc（matched v2 111）', 'Val Acc', 'n'], rows), '']
    if results.get('E9_latency_size'):
        e9 = results['E9_latency_size']
        out += ['**模型体积与延迟**：体积 %s MB，推理 %.2f ms/条（仅 CNN 前向，不含特征提取），种子数 %s。'
                % (e9.get('model_size_mb'), (e9.get('inference_sec_per_sample') or 0) * 1000,
                   e9.get('n_seeds')), '']
    return '\n'.join(out)


def stat_section(results):
    stats = (results or {}).get('E6_statistics')
    if not stats:
        return '## 3. 统计检验\n\n（缺少 E6）'
    rows = []
    for key, item in stats.items():
        gb = item.get('group_bootstrap', {})
        rows.append([key, fmt(item.get('mcnemar_exact_p_two_sided')),
                     f"{gb.get('acc_diff_mean', 0):+.4f}",
                     f"[{gb.get('ci95', ['-', '-'])[0]:+.4f}, {gb.get('ci95', ['-', '-'])[1]:+.4f}]",
                     item.get('interpretation')])
    return '\n'.join(['## 3. 统计检验（配对 McNemar + 按表达组自助法）', '',
                      md_table(['比较', 'McNemar p', 'ΔAcc(B−A)', 'ΔAcc 95%CI', '判定'], rows)])


def final_key(results):
    """最终模型：优先 E5 融合（val 选），否则 E4 TTA，否则 E3。"""
    for k in ('E5_fusion', 'E4_cnn_tta', 'E3_cnn_ensemble'):
        if (results or {}).get(k):
            return k
    return None


def condition_section(results):
    key = final_key(results)
    item = (results or {}).get(key) or {}
    full = item.get('full_test') or {}
    strat = full.get('stratified_accuracy') or {}
    out = [f'## 4. 条件与语域分层（{key}）', '']
    if strat.get('condition'):
        rows = [[k, v['n'], pct(v['accuracy'])] for k, v in strat['condition'].items()]
        out += ['**按声学条件**', '', md_table(['条件', 'n', 'Accuracy'], rows), '']
    if strat.get('register'):
        rows = [[k, v['n'], pct(v['accuracy'])] for k, v in strat['register'].items()]
        out += ['**按语域**', '', md_table(['语域', 'n', 'Accuracy'], rows), '']
    if strat.get('voice'):
        rows = [[k, v['n'], pct(v['accuracy'])] for k, v in strat['voice'].items()]
        out += ['**按音色**', '', md_table(['音色', 'n', 'Accuracy'], rows), '']
    return '\n'.join(out)


def class_section(results):
    key = final_key(results)
    e4 = (results or {}).get(key) or {}
    pc = (e4.get('full_test') or {}).get('per_class')
    if not pc:
        return '## 5. 每类指标\n\n（缺少）'
    rows = [[k, pct(v['precision']), pct(v['recall']), pct(v['f1']), v['support']]
            for k, v in pc.items()]
    return '\n'.join([f'## 5. 每类 Precision/Recall/F1（{key}）', '',
                      md_table(['意图', 'Precision', 'Recall', 'F1', 'Support'], rows)])


def error_section(results):
    ea = (results or {}).get('E10_error_analysis')
    if not ea:
        return '## 6. 错误分析\n\n（缺少）'
    rows = [[f"{e['true']} → {e['pred']}", e['count']] for e in ea.get('top_error_pairs', [])]
    out = [f"## 6. 错误分析（{ea.get('model', 'final')}）", '',
           '```text', 'confusion matrix（行=真实，列=预测；类顺序 LIGHT_ON, LIGHT_OFF, CURTAIN_OPEN, CURTAIN_CLOSE, ALL_OFF, STATUS_ALL）']
    for row in ea.get('confusion_matrix', []):
        out.append('  ' + ' '.join(f'{v:3d}' for v in row))
    out += ['```', '', '**主要错误对**', '']
    out.append(md_table(['真实 → 预测', '次数'], rows) if rows else '（无错误）')
    if ea.get('errors_by_register'):
        out += ['', '**错误按语域**：' + '，'.join(f'{k}={v}' for k, v in ea['errors_by_register'].items())]
    return '\n'.join(out)


def rejection_section(rej):
    if not rej:
        return '## 7. 拒识评估\n\n（缺少 rejection.json）'
    tr = rej['test_results']
    rows = [
        ['已知命令覆盖率', pct(tr.get('known_coverage'))],
        ['已知命令误拒识率', pct(tr.get('known_false_reject_rate'))],
        ['被接受样本准确率', pct(tr.get('accepted_accuracy'))],
        ['无阈值时准确率', pct(tr.get('no_threshold_accuracy'))],
        ['未知命令误接受率（test）', pct(tr.get('unknown_false_accept_rate'))],
        ['未知误接受 95% Wilson CI', f"[{tr.get('unknown_far_wilson95', [0, 0])[0]:.3f}, {tr.get('unknown_far_wilson95', [0, 0])[1]:.3f}]"],
        ['硬负例误接受率', pct(tr.get('hard_negative_far'))],
        ['未知测试样本数', tr.get('n_unknown_test')],
    ]
    out = ['## 7. 拒识评估（阈值仅验证集 + 未知校准集选择）', '',
           f"选中阈值 **{rej.get('selected_threshold'):.2f}**；"
           f"验证集覆盖率 {pct(rej['selected_on_validation']['known_coverage'])}，"
           f"验证集未知误接受率 {pct(rej['selected_on_validation']['unknown_false_accept_rate'])}。", '',
           md_table(['指标', '值'], rows)]
    by_type = tr.get('by_negative_type') or {}
    if by_type:
        out += ['', '**按硬负例类型（未知 test 36 条）**', '',
                md_table(['类型', 'n', '被接受数'],
                         [[k, v['n'], v['accepted']] for k, v in by_type.items()])]
    per_text = rej.get('unknown_test_per_text') or {}
    risky = sorted(per_text.items(), key=lambda kv: -kv[1]['max_prob'])[:8]
    out += ['', '**最高置信度的未知文本（最危险）**', '',
            md_table(['文本', 'n', '被接受', '最高置信度', '主要预测'],
                     [[k, v['n'], v['n_accepted'], v['max_prob'],
                       max(v['pred_distribution'], key=v['pred_distribution'].get)]
                      for k, v in risky])]
    if rej.get('limitations'):
        out += ['', '限制：' + '；'.join(rej['limitations'][:2])]
    return '\n'.join(out)


def train_section(train_report, results):
    if not train_report:
        return '## 8. 训练协议\n\n（缺少 train_report.json）'
    rows = []
    for s in train_report.get('seeds', []):
        rows.append([s['seed'], fmt(s['best_val_accuracy']), s['best_epoch'],
                     s['epochs_run'], s['train_seconds']])
    out = ['## 8. 训练协议与验证集结果', '',
           f"特征：{train_report['features']['kind'] if 'kind' in train_report['features'] else 'logmel+Δ'}，"
           f"n_frames={train_report['features'].get('n_frames')}；"
           f"数据哈希：manifest {train_report['protocol']['data_hashes']['manifest_sha256'][:12]}…，"
           f"split {train_report['protocol']['data_hashes']['split_sha256'][:12]}…。", '',
           md_table(['seed', 'Val Acc', 'best epoch', 'epochs', '训练秒'], rows), '',
           f"集成验证集准确率 **{pct(train_report['ensemble_val']['accuracy'])}**"
           f"（Macro-F1 {train_report['ensemble_val']['macro_f1']:.4f}）；"
           f"温度 T={train_report['temperature']:.3f}；总训练 {train_report['train_seconds']}s。"]
    return '\n'.join(out)


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else 'main_v3_1'
    results = load(OUT / run_name / 'results.json')
    rejection = load(OUT / run_name / 'rejection.json') or load(OUT / 'rejection.json')
    train_report = load(OUT / run_name / 'train_report.json') or load(OUT / 'train_report.json')
    sections = [
        '# 方言智控盒 v3 实验结果表（自动生成）', '',
        f"运行：`{run_name}`；生成时间：{__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        '',
        data_section(run_name), main_section(results), stat_section(results),
        condition_section(results), class_section(results), error_section(results),
        rejection_section(rejection), train_section(train_report, results),
    ]
    print('\n\n'.join(sections))


if __name__ == '__main__':
    main()

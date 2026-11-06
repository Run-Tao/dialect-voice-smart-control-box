# -*- coding: utf-8 -*-
"""把 outputs/v2_experiments/<run>/results.json 汇总为 Markdown 表格片段。

仅做格式化，不做任何数值改写或挑选；所有数字直接来自结果文件。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INTENTS = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']


def fmt(x, nd=4):
    if x is None:
        return '—'
    if isinstance(x, float):
        return f'{x:.{nd}f}'
    return str(x)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'main_v2'
    path = BASE / 'outputs' / 'v2_experiments' / run / 'results.json'
    r = json.loads(path.read_text(encoding='utf-8'))
    ex = r['experiments']
    out = []

    out.append(f"### 运行元信息\n")
    out.append(f"| 项 | 值 |\n|---|---|")
    out.append(f"| run_id | `{r['run_id']}` |")
    out.append(f"| 生成时间 | {r['generated_at']} |")
    out.append(f"| Python / numpy | {r['python']} / {r['numpy']} |")
    out.append(f"| git commit | {r['git_commit']} |")
    out.append(f"| 清单文件 | `{r['manifest']['path']}` |")
    out.append(f"| 清单 sha256 | `{r['manifest']['sha256'][:16]}…` |")
    out.append(f"| 划分文件 | `{r['split']['split_file']}` |")
    out.append(f"| 划分 sha256 | `{r['split']['sha256'][:16]}…` |")
    out.append(f"| 样本数 | train {r['dataset_sizes']['samples']['train']} / "
               f"val {r['dataset_sizes']['samples']['val']} / "
               f"test {r['dataset_sizes']['samples']['test']} |")
    out.append(f"| 表达组数 | train {r['dataset_sizes']['groups']['train']} / "
               f"val {r['dataset_sizes']['groups']['val']} / "
               f"test {r['dataset_sizes']['groups']['test']} |")
    out.append(f"| 特征提取耗时 | {r['feature_extraction_seconds']:.1f}s |")
    out.append(f"| 总耗时 | {r['total_elapsed_sec']:.1f}s |")
    out.append(f"| 运行命令 | `{r['command_used']}` |")
    out.append('')

    if 'E1_baseline' in ex:
        e = ex['E1_baseline']
        out.append('### E1 基线（仅 v1 历史训练数据，固定测试集）\n')
        if 'test_metrics' in e:
            tm = e['test_metrics']
            out.append(f"训练样本 {e['train_size']} / 验证 {e['val_size']} / 测试 {e['test_size']}；"
                       f"每类模板数 {e['templates_per_class']}\n")
            out.append('| 指标 | 验证集 | 测试集 |\n|---|---:|---:|')
            out.append(f"| Accuracy | {fmt(e['val_metrics'].get('accuracy'))} | {fmt(tm['accuracy'])} |")
            out.append(f"| Macro-F1 | {fmt(e['val_metrics'].get('macro_f1'))} | {fmt(tm['macro_f1'])} |")
            out.append('')
            out.append('每类测试指标：\n')
            out.append('| 命令 | Precision | Recall | F1 | Support |\n|---|---:|---:|---:|---:|')
            for c in INTENTS:
                d = tm['per_class'][c]
                out.append(f"| {c} | {fmt(d['precision'])} | {fmt(d['recall'])} | "
                           f"{fmt(d['f1'])} | {d['support']} |")
            out.append('')
            out.append('测试集混淆矩阵（行=真实，列=预测，顺序 '
                       + ', '.join(tm['labels']) + '）：\n')
            out.append('```')
            for row in tm['confusion_matrix']:
                out.append(' '.join(f'{v:3d}' for v in row))
            out.append('```')
            out.append(f"\n推理耗时：{tm['inference_seconds_per_sample']*1000:.1f} ms/样本\n")
        else:
            out.append(f"未执行：{e.get('error')}\n")

    if 'E2_data_scale' in ex:
        e = ex['E2_data_scale']
        out.append('### E2 数据规模曲线（验证/测试集固定）\n')
        out.append(f"固定验证集 {e['fixed_val_size']} 条，固定测试集 {e['fixed_test_size']} 条\n")
        out.append('| 训练比例 | 训练组数 | 训练样本 | 每类模板 | Val Acc | Val F1 | Test Acc | Test F1 | ms/样本 |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for c in e['curve']:
            tpl = c['templates_per_class']
            tpl_s = f"{min(tpl.values())}–{max(tpl.values())}"
            out.append(f"| {c['fraction']:.0%} | {c['train_groups']} | {c['train_size']} | "
                       f"{tpl_s} | {fmt(c['val_accuracy'])} | {fmt(c['val_macro_f1'])} | "
                       f"{fmt(c['test_accuracy'])} | {fmt(c['test_macro_f1'])} | "
                       f"{c['inference_seconds_per_sample']*1000:.1f} |")
        out.append('')

    if 'E3_template_count' in ex:
        e = ex['E3_template_count']
        out.append('### E3 模板数量（medoid 策略；k=all 为全模板）\n')
        out.append('| 每类模板 k | 模板总数 | Val Acc | Val F1 | Test Acc | Test F1 | ms/样本 |')
        out.append('|---:|---:|---:|---:|---:|---:|---:|')
        for c in e['results']:
            out.append(f"| {c['k_per_class']} | {c['templates_total']} | "
                       f"{fmt(c['val_accuracy'])} | {fmt(c['val_macro_f1'])} | "
                       f"{fmt(c['test_accuracy'])} | {fmt(c['test_macro_f1'])} | "
                       f"{c['inference_seconds_per_sample']*1000:.1f} |")
        out.append('')

    if 'E4_template_selection' in ex:
        e = ex['E4_template_selection']
        out.append('### E4 模板选择策略（多种子 + 按表达组自助法）\n')
        out.append('| 策略 | 模板数/类 | Acc 均值 | Acc 标准差 | F1 均值 | F1 标准差 | Acc 95%CI | F1 95%CI |')
        out.append('|---|---:|---:|---:|---:|---:|---|---|')
        for s, v in e['strategies'].items():
            ci = v['group_bootstrap_95ci']
            out.append(f"| {s} | {v['n_per_class']} | {fmt(v['accuracy_mean'])} | "
                       f"{fmt(v['accuracy_std'])} | {fmt(v['macro_f1_mean'])} | "
                       f"{fmt(v['macro_f1_std'])} | "
                       f"[{fmt(ci['accuracy_ci'][0],3)}, {fmt(ci['accuracy_ci'][1],3)}] | "
                       f"[{fmt(ci['macro_f1_ci'][0],3)}, {fmt(ci['macro_f1_ci'][1],3)}] |")
        out.append('')
        best = max(e['strategies'], key=lambda k: e['strategies'][k]['accuracy_mean'])
        out.append(f"多种子均值最高的策略：`{best}`（自助法按表达组重采样，"
                   f"{e['strategies'][best]['group_bootstrap_95ci']['n_groups']} 个测试组）。\n")

    if 'B_unseen_voice' in ex:
        e = ex['B_unseen_voice']
        out.append('### 轨道B 未見音色（留一音色，辅助证据）\n')
        out.append('| 留出音色 | 训练样本 | 测试样本 | Test Acc | Test Macro-F1 | Acc 95%CI | 严格未见表达子集 n | 严格 Test Acc |')
        out.append('|---|---:|---:|---:|---:|---|---:|---:|')
        for v, s in e['voices'].items():
            ci = s['group_bootstrap_95ci']['accuracy_ci']
            strict = s.get('strict_unseen_expression')
            out.append(f"| {v} | {s['train_samples']} | {s['test_samples']} | "
                       f"{fmt(s['test_accuracy'])} | {fmt(s['test_macro_f1'])} | "
                       f"[{fmt(ci[0],3)}, {fmt(ci[1],3)}] | "
                       f"{strict['test_samples'] if strict else 0} | "
                       f"{fmt(strict['test_accuracy']) if strict else '—'} |")
        out.append('')

    if 'N_normalize' in ex:
        e = ex['N_normalize']
        out.append('### 距离归一化对照\n')
        if e.get('template_strategy'):
            out.append(f"模板策略：{e['template_strategy']}\n")
        out.append('| 归一化 | Val Acc | Val F1 | Test Acc | Test F1 |\n|---|---:|---:|---:|---:|')
        for k, v in e.items():
            if not isinstance(v, dict):
                continue
            out.append(f"| {k} | {fmt(v['val_accuracy'])} | {fmt(v['val_macro_f1'])} | "
                       f"{fmt(v['test_accuracy'])} | {fmt(v['test_macro_f1'])} |")
        out.append('')

    if 'F_max_frames' in ex:
        e = ex['F_max_frames']
        out.append('### max_frames 扫描（选择只看验证集）\n')
        out.append('| max_frames | Val Acc | Val F1 | Test Acc | Test F1 | ms/样本 |')
        out.append('|---|---:|---:|---:|---:|---:|')
        for c in e['max_frames_sweep']:
            out.append(f"| {c['max_frames']} | {fmt(c['val_accuracy'])} | {fmt(c['val_macro_f1'])} | "
                       f"{fmt(c['test_accuracy'])} | {fmt(c['test_macro_f1'])} | "
                       f"{c['inference_seconds_per_sample']*1000:.1f} |")
        out.append('')
        out.append(f"验证集选择结果：`max_frames={e['selected_on_validation']}`\n")

    print('\n'.join(out))


if __name__ == '__main__':
    main()

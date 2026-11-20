# -*- coding: utf-8 -*-
"""报告数字抽查：核对 docs/v3_report.md 中的关键数字与 JSON / 文档源头是否一致。

只做「报告中出现该数字」+「源头值经复算等于该数字」的双向检查，
不修改任何文件。用于交付前的自检（task-5 要求抽查 10 个以上数字）。

用法::

    D:\\anaconda3\\python.exe docs/report_assets/check_report_numbers.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
REPORT = BASE / 'docs' / 'v3_report.md'


def load(rel):
    p = BASE / rel
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def fmt(x, nd=4):
    return f'{x:.{nd}f}'


def main():
    text = REPORT.read_text(encoding='utf-8')
    v3 = load('outputs/v3_data_audit.json')
    v31 = load('outputs/v3_1_data_audit.json')
    res = load('outputs/v3_experiments/main_v3_1/results.json') or {}
    rej = load('outputs/v3_experiments/main_v3_1/rejection.json') or {}
    checks = []  # (说明, 报告中应出现的字符串, 源头值(str), 来源)

    def add(label, token, value, source):
        checks.append((label, token, value, source))

    if v3:
        c = v3['counts']
        add('v3 总行数', '2796', str(c['total_rows']), 'v3_data_audit.counts.total_rows')
        add('v3 表达组', '192', str(c['num_expression_groups']), 'v3_data_audit.counts.num_expression_groups')
        add('v3 每类样本', '466', str(c['by_command_all']['LIGHT_ON']), 'v3_data_audit.counts.by_command_all')
        add('v3 train 行', '1873', str(c['by_split']['train']), 'v3_data_audit.counts.by_split.train')
        add('v3 test 行', '483', str(c['by_split']['test']), 'v3_data_audit.counts.by_split.test')
        add('v3 rate_slow_18', '576', str(c['by_condition_usable']['rate_slow_18']),
            'v3_data_audit.counts.by_condition_usable')
        add('v3 rate_fast_18', '522', str(c['by_condition_usable']['rate_fast_18']),
            'v3_data_audit.counts.by_condition_usable')
        add('v3 offline_signal_aug', '2142', str(c['by_generation_method_usable']['offline_signal_aug']),
            'v3_data_audit.counts.by_generation_method_usable')
        add('v3 HiuGaai 条数', '936', str(c['by_voice_usable']['edge-zh-HK-HiuGaaiNeural']),
            'v3_data_audit.counts.by_voice_usable')
        add('v3 未知集条数', '72', str(v3['unknown']['n_rows']), 'v3_data_audit.unknown.n_rows')
        cv = v3['condition_verification']['rate_slow_18']
        add('rate_slow_18 时长比', '1.2195', fmt(cv['duration_ratio']['mean']),
            'v3_data_audit.condition_verification.rate_slow_18.duration_ratio.mean')
        add('pitch_low f0 偏移', '−14.61',
            '−' + fmt(abs(v3['condition_verification']['pitch_low']['f0_delta_hz_estimated']['mean']), 2),
            'v3_data_audit.condition_verification.pitch_low.f0_delta_hz_estimated.mean')
        nm = v3['fallback']['network_measurements']
        add('并发6 失败率', '41.7', fmt(nm[0]['failure_rate'] * 100, 1), 'v3_data_audit.fallback.network_measurements[0]')
        add('并发3 失败率', '22.2', fmt(nm[2]['failure_rate'] * 100, 1), 'v3_data_audit.fallback.network_measurements[2]')
    if v31:
        c = v31['counts']
        add('v3.1 总行数', '2904', str(c['total_rows']), 'v3_1_data_audit.counts.total_rows')
        add('v3.1 表达组', '210', str(c['num_expression_groups']), 'v3_1_data_audit.counts.num_expression_groups')
        add('v3.1 每类样本', '484', str(c['by_command']['LIGHT_ON']), 'v3_1_data_audit.counts.by_command')
        add('v3.1 train 行', '1981', str(c['by_split']['train']), 'v3_1_data_audit.counts.by_split.train')
        add('v3.1 train correction 组', '24', str(c['correction_groups_by_split']['train']),
            'v3_1_data_audit.counts.correction_groups_by_split.train')
        add('v3.1 测试集 correction 组', '6', str(c['correction_groups_by_split']['test']),
            'v3_1_data_audit.counts.correction_groups_by_split.test')
        add('v3.1 Jaccard 最大', '0.4444', fmt(v31['correction_text_similarity']['max_jaccard_overall']),
            'v3_1_data_audit.correction_text_similarity.max_jaccard_overall')
    if res:
        e1 = res.get('E1_v2_dtw_baseline') or {}
        e2 = res.get('E2_classical_logreg') or {}
        if e1:
            add('E1 full Acc', '0.5901', fmt(e1['full_test']['accuracy']), 'results.E1.full_test.accuracy')
            add('E1 matched Acc', '0.5495', fmt(e1['matched_v2_test']['accuracy']), 'results.E1.matched_v2_test.accuracy')
            add('E1 matched F1', '0.5411', fmt(e1['matched_v2_test']['macro_f1']), 'results.E1.matched_v2_test.macro_f1')
            add('E1 question 语域', '0.3000', fmt(e1['full_test']['stratified_accuracy']['register']['question']['accuracy']),
                'results.E1.stratified.register.question')
            add('E1 correction 语域', '0.5408', fmt(e1['full_test']['stratified_accuracy']['register']['correction']['accuracy']),
                'results.E1.stratified.register.correction')
        if e2:
            add('E2 full Acc', '0.6749', fmt(e2['full_test']['accuracy']), 'results.E2.full_test.accuracy')
            add('E2 matched Acc', '0.7027', fmt(e2['matched_v2_test']['accuracy']), 'results.E2.matched_v2_test.accuracy')
            add('E2 correction 语域', '0.3571', fmt(e2['full_test']['stratified_accuracy']['register']['correction']['accuracy']),
                'results.E2.stratified.register.correction')
        e5 = res.get('E5_fusion') or {}
        if e5:
            add('E5（最优）full Acc', '0.9151', fmt(e5['full_test']['accuracy']), 'results.E5.full_test.accuracy')
            add('E5 matched Acc', '0.8919', fmt(e5['matched_v2_test']['accuracy']), 'results.E5.matched_v2_test.accuracy')
            add('E5 correction 语域', '0.6429',
                fmt(e5['full_test']['stratified_accuracy']['register']['correction']['accuracy']),
                'results.E5.stratified.register.correction')
        ea = res.get('E10_error_analysis') or {}
        if ea:
            add('E10 correction 语域', '0.6429', fmt(ea['correction_register_accuracy']),
                'results.E10.correction_register_accuracy')
    abl = load('outputs/v3_experiments/data_ablation/comparison.json')
    if abl:
        a = abl['test_full']['v3_only']['accuracy']
        b = abl['test_full']['v3_1_seed42']['accuracy']
        add('数据消融 v3-only full', '0.8778', fmt(a), 'comparison.test_full.v3_only.accuracy')
        add('数据消融 v3.1 full', '0.9130', fmt(b), 'comparison.test_full.v3_1_seed42.accuracy')
        add('数据消融 Δ(pp)', '3.52', fmt((b - a) * 100, 2), 'comparison.test_full 差值')
    ood = load('outputs/v3_experiments/main_v3_1/ood.json')
    if ood:
        cs = ood['scores']['cos_centroid']
        add('OOD cos_centroid AUC', '0.9517', fmt(cs['auc_known_val_vs_unknown_cal']),
            'ood.scores.cos_centroid.auc_known_val_vs_unknown_cal')
        add('OOD cos_centroid test FAR', '0.3611', fmt(cs['test']['unknown_false_accept_rate']),
            'ood.scores.cos_centroid.test.unknown_false_accept_rate')
    if rej:
        tr = rej.get('test_results') or {}
        add('拒识阈值', fmt(rej['selected_threshold'], 2), fmt(rej['selected_threshold'], 2),
            'rejection.selected_threshold')
        add('rule-of-three 上界', '8.3', fmt((tr.get('far_rule_of_three_upper95') or 0) * 100, 1),
            'rejection.test_results.far_rule_of_three_upper95')

    ok = 0
    print(f'{"检查项":34s} {"报告含":>6s} {"源头":>8s} {"一致":>4s}  来源')
    print('-' * 110)
    for label, token, value, source in checks:
        in_report = token in text
        same = (token.replace('−', '-') == value.replace('−', '-')) if value not in ('', None) else False
        ok += int(in_report and same)
        print(f'{label:34s} {str(in_report):>6s} {value:>8s} {str(same):>4s}  {source}')
    print('-' * 110)
    print(f'通过 {ok}/{len(checks)}')


if __name__ == '__main__':
    main()

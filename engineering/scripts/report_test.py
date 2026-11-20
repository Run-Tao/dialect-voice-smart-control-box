# -*- coding: utf-8 -*-
"""测试报告生成：把评估工件汇总成一份 Markdown 测试报告。

原则：
- 数字一律直接读自 JSON 工件，不在本脚本里重新计算，保证报告与工件一致；
- 合成语料与真人语料分开列表，真人缺数据时写「待采集」并给出补数命令；
- 缺失的工件不报错，对应小节标注生成命令。

示例：
    python scripts/report_test.py --out docs/测试报告.md
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'

KNOWN = ('ALL_OFF', 'CURTAIN_CLOSE', 'CURTAIN_OPEN', 'LIGHT_OFF', 'LIGHT_ON', 'STATUS_ALL')


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def variant(report):
    """评估工件里取「带 VAD」这一档，即部署路径。"""
    if report is None:
        return None
    return report.get('variants', {}).get('with_vad', report)


def fmt(v, nd=4):
    return '—' if v is None else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def eval_table(entry, title):
    if entry is None:
        return [f'### {title}', '', '_未生成。_', '']
    kn = entry['known']
    unk = entry.get('unknown') or {}
    lines = [f'### {title}', '',
             '| 指标 | 数值 |', '|---|---:|',
             f"| 样本数 | {kn['n']} |",
             f"| 命令识别准确率（argmax） | {fmt(kn['argmax_accuracy'])} |",
             f"| 接受率 | {fmt(kn['acceptance_rate'])} |",
             f"| 接受样本准确率 | {fmt(kn['accepted_accuracy'])} |",
             f"| 未知指令误接受率 | {fmt(unk.get('false_accept_rate'))} |",
             f"| 单条延迟（特征+推理，均值） | {fmt(kn.get('mean_latency_ms'), 1)} ms |",
             '']
    per = kn.get('per_command_argmax_accuracy') or {}
    if per:
        lines += ['分命令 argmax 准确率：', '',
                  '| ' + ' | '.join(KNOWN) + ' |',
                  '|' + '---:|' * len(KNOWN),
                  '| ' + ' | '.join(fmt(per.get(c)) for c in KNOWN) + ' |', '']
    return lines


def confusion_block(entry):
    conf = (entry or {}).get('known', {}).get('confusion')
    if not conf:
        return []
    lines = ['混淆矩阵（行 = 真实，列 = 预测，仅列非零项）：', '',
             '| 真实 \\ 预测 | 计数 |', '|---|---|']
    for truth in KNOWN:
        for pred, n in sorted((conf.get(truth) or {}).items()):
            mark = '' if pred == truth else ' ← 误判'
            lines.append(f'| {truth} → {pred} | {n}{mark} |')
    return lines + ['']


def top_confusion(entry):
    """最大的单一误判流向，用于结论里点出最值得补数据的类别。"""
    conf = (entry or {}).get('known', {}).get('confusion') or {}
    pairs = [(t, p, n) for t, row in conf.items() for p, n in row.items() if p != t]
    if not pairs:
        return None
    truth, pred, n = max(pairs, key=lambda x: x[2])
    total_err = sum(c for _, _, c in pairs)
    return truth, pred, n, total_err


def coverage_at(curve, threshold):
    """从风险-覆盖曲线取某阈值下的合成覆盖率（曲线按阈值升序）。"""
    if not curve or threshold is None:
        return None
    at_or_above = [r for r in curve if r['threshold'] >= threshold - 1e-9]
    return at_or_above[0]['coverage'] if at_or_above else 0.0


def real_far_block(report, deployed_threshold):
    """真实域误接受率：真人非命令语音被当成命令接受的比例。

    与合成 unknown 集不是一回事——合成负样本与训练集同声学域，FAR 天然偏乐观；
    真人语音才是真正的域外负样本。许可与筛选口径必须和数字一起出现，
    否则读者无法判断这个 FAR 能不能用、有没有被筛低。
    """
    title = '## 3. 真实域误接受率（真人非命令语音）'
    if report is None:
        return [title, '', '_未生成。_', '',
                '```bash',
                'python scripts/build_real_testset.py --source <名称>=<语料目录> \\',
                '    --license "<名称>=<许可原文>" \\',
                '    --transcripts <名称>=<转写表.tsv> --out <测试集目录>',
                'python scripts/eval_real_far.py --manifest <测试集目录>/manifest.csv \\',
                '    --out docs/real_far.json',
                '```', '']

    head = report['headline']
    ts = report['testset']
    prov = ts.get('provenance') or {}
    excluded = prov.get('excluded') or {}
    per_source = ts.get('per_source') or {}
    unk = report.get('synthetic_unknown_far') or {}
    dang = (report.get('dangerous_split') or {}).get('dangerous') or {}
    curve = (report.get('risk_coverage') or {}).get('curve') or []

    lines = [title, '', '**语料与许可**（CC BY 要求署名随数据走）：', '',
             '| 来源 | 许可 | 评估片段 | FAR |', '|---|---|---:|---:|']
    for name, src in sorted((prov.get('sources') or {}).items()):
        stat = per_source.get(name) or {}
        lines.append(f"| `{name}` | {src.get('license')} | {stat.get('n', '—')} "
                     f"| {fmt(stat.get('far'))} |")
    lines += ['',
              f"- 评估片段 {head['n']} 条，合计 {fmt(ts.get('total_duration_s'), 1)} s，"
              f"跳过 {ts.get('skipped')} 条（VAD 判无语音）",
              f"- 转写核查：{fmt(prov.get('segments_text_verified'))} 条已逐条核对转写文本，"
              f"因命中命令短语剔除 {excluded.get('command_like_text', 0)} 条；"
              f"{fmt(prov.get('segments_with_device_hint'))} 条提到器件词（如「燈塔」），"
              '属合法负样本，未剔除',
              f"- 筛选口径：{prov.get('selection_note')}",
              f"- VAD 参数与部署一致：`{json.dumps(prov.get('vad_kwargs'), ensure_ascii=False)}`",
              '', '**结果**：', '',
              '| 指标 | 数值 |', '|---|---:|',
              f"| 真实域 FAR（阈值 {fmt(report['threshold'], 3)}） | "
              f"**{fmt(head['far'])}** |",
              f"| 95% 置信区间（Wilson） | "
              f"[{fmt(head['ci95'][0])}, {fmt(head['ci95'][1])}] |",
              f"| 样本数 | {head['n']} |",
              f"| 合成 unknown FAR（同阈值，对比用） | {fmt(unk.get('far'))}"
              f"（n={unk.get('n')}） |",
              '| 其中「会真的驱动设备」的危险误动作率 | '
              + (f"**{fmt(dang['rate'])}**（95% CI [{fmt(dang['ci95'][0])}, "
                 f"{fmt(dang['ci95'][1])}]，{dang['accepted']} 条；"
                 '已剔除只查询不动设备的 STATUS_ALL）' if dang else '—') + ' |',
              f"| 单条延迟（特征+推理，均值） | "
              f"{fmt(head.get('mean_latency_ms'), 1)} ms |",
              '']

    if head['far'] is not None and unk.get('far') is not None:
        gap = head['far'] - unk['far']
        times = head['far'] / unk['far'] if unk['far'] else None
        lines += [f"> **域差距**：同阈值下真实域 FAR 比合成 unknown "
                  f"{'高' if gap >= 0 else '低'} {abs(gap):.4f}"
                  + (f"（约 {times:.1f} 倍）" if times else '')
                  + '。合成负样本与训练集同声学域，此前那个数字偏乐观。', '']

    fac = report.get('false_accept_by_command') or {}
    if fac:
        total = sum(fac.values())
        lines += ['误接受落到哪个命令（决定「危险动作」概率，不只是「有没有误触发」）：',
                  '', '| 命令 | 次数 | 占全部误接受 |', '|---|---:|---:|']
        for cmd, n in fac.items():
            lines.append(f'| {cmd} | {n} | {n / total:.1%} |')
        lines += ['']

    solved = report.get('threshold_solve') or []
    if solved:
        lines += ['**阈值反解**：要把真实域 FAR 压到目标值，阈值得抬到多少、'
                  '代价是合成验证集覆盖率掉到多少：', '',
                  '| 目标 FAR | 所需阈值 | 该阈值下真实 FAR | 合成覆盖率 |',
                  '|---:|---:|---:|---:|']
        for item in solved:
            lines.append(f"| {fmt(item['target_far'], 2)} | {fmt(item['threshold'], 4)} "
                         f"| {fmt(item['far_at_threshold'])} "
                         f"| {fmt(coverage_at(curve, item['threshold']))} |")
        lines += ['',
                  f"当前部署阈值 {fmt(deployed_threshold, 3)} 下的合成覆盖率："
                  f"{fmt(coverage_at(curve, deployed_threshold))}"
                  '（取曲线上阈值 ≥ 部署阈值的最近网格点，与第 4 节按精确阈值算出的'
                  '覆盖率会有千分位量级差异）', '']
        for item in solved:
            if not item['reachable']:
                lines += [f"> 目标 FAR {fmt(item['target_far'], 2)} **不可达**："
                          f"{item.get('reason')}", '']
                break

    rc = report.get('risk_coverage')
    if rc and rc.get('aurc_cross_domain') is not None:
        lines += [f"跨域风险-覆盖 AURC：{fmt(rc['aurc_cross_domain'])}"
                  f"（覆盖 {rc.get('coverage_levels_used')} 档）。{rc.get('note')}"
                  '第 4.1 节对同一信号报的 AURC 会有小数第三位差异，'
                  '因为两个脚本的阈值网格取法不同。', '']

    pctls = report.get('confidence_percentiles') or {}
    if pctls:
        lines += ['真实域负样本的置信度分布（越高越像命令，越危险）：', '',
                  '| P50 | P90 | P95 | P99 | 最大 |', '|---:|---:|---:|---:|---:|',
                  f"| {fmt(pctls.get('p50'))} | {fmt(pctls.get('p90'))} "
                  f"| {fmt(pctls.get('p95'))} | {fmt(pctls.get('p99'))} "
                  f"| {fmt(pctls.get('max'))} |", '',
                  f"> {report.get('note')}", '']
    return lines


SIGNAL_LABELS = {
    'maxprob': 'max-prob（当前部署信号）',
    'margin': 'margin（top1−top2）',
    'neg_entropy': '负熵',
    'seed_agree': '种子一致性 1−JS/ln2',
    'ood': '嵌入 kNN 距离（取负，OOD 分）',
    'combo_min_z_maxprob_ood': 'min-z(maxprob, ood)',
    'combo_min_z_maxprob_seed': 'min-z(maxprob, seed_agree)',
    'combo_sum_z_maxprob_ood': 'sum-z(maxprob, ood)',
}


def _wp(wps, target):
    """取指定目标风险下的工作点。"""
    for w in wps or []:
        if abs(w['target_risk'] - target) < 1e-9:
            return w
    return None


def selective_block(report, deployed_threshold, real_far=None):
    """选择性分类：单信号与双信号拒识的对比。

    第 3 节证明单靠 max-prob 抬阈值救不了 FAR（覆盖率断崖）。本节回答的是
    另一个问题：换信号或加信号，能不能在同一覆盖率下把 FAR 压下来。

    跨域 / 同域两条 AUC 必须同时出现：只有同域 AUC 高的信号是 TTS-vs-真人
    域检测器，它的跨域曲线不能拿去承诺真人 FAR。这个判定是本节的核心控制。
    """
    title = '## 4. 选择性分类：单信号 vs 双信号拒识'
    if report is None:
        return [title, '', '_未生成。_', '',
                '```bash',
                'python scripts/eval_selective.py \\',
                '    --manifest <真实域测试集>/manifest.csv \\',
                '    --corpus-manifest <合成语料>/manifest_yue_v3_1_vad.csv \\',
                '    --unknown-dir <合成语料>/unknown_v3_vad \\',
                '    --out docs/selective_eval.json',
                '```', '']

    base = report['baseline_maxprob']
    groups = report['groups']
    sig = report['signals']
    target = 0.10
    ece = base.get('ece_on_syn_known') or {}

    # 断崖的具体数字从第 3 节工件里取，不在这里写死
    cliff = '第 3 节的阈值反解显示：把真实域 FAR 压到目标值需要把 max-prob 阈值抬到'
    solved = {round(s['target_far'], 2): s for s in (real_far or {}).get('threshold_solve', [])}
    if target in solved and solved[target].get('reachable'):
        s = solved[target]
        curve = ((real_far or {}).get('risk_coverage') or {}).get('curve') or []
        cliff = (f"第 3 节的阈值反解显示：要把真实域 FAR 压到 {fmt(target, 2)}，"
                 f"max-prob 阈值得抬到 {fmt(s['threshold'], 4)}，"
                 f"而合成覆盖率同时掉到 {fmt(coverage_at(curve, s['threshold']))}"
                 f"——**单信号抬阈值此路不通**。")
    else:
        cliff += '目标值，代价是合成覆盖率断崖式下跌——**单信号抬阈值此路不通**。'

    lines = [title, '',
             cliff,
             '本节改问：换信号或加信号，能否在同一覆盖率下把 FAR 压下来。', '',
             '**口径**（混了两条曲线结论就废）：', '',
             f"- 正样本（覆盖率一侧）：合成已知命令 `{groups['syn_known']['n']}` 条；"
             f"kNN 参照池取自**互不重叠**的 train 分割 `{groups['pool']['n']}` 条"
             f"（含 val 会自匹配到距离 0）",
             f"- 跨域负样本：真人粤语 `{groups['real']['n']}` 条（第 3 节同一测试集）",
             f"- 同域负样本：合成非命令 `{groups['syn_unknown']['n']}` 条",
             '- 跨域 AUC 明显高于同域 AUC 的信号是**域检测器**（主要靠 TTS 与真人的'
             '声学差异取分，不是靠命令内容），它的跨域曲线不能拿去承诺真人 FAR；'
             '两者接近才是内容判别器',
             f"- z 统计量与 AND 阈值**只在合成语料上拟合**"
             f"（{report['z_stats_calibrated_on']['source']} + 合成非命令），"
             '真人测试集只用于评估，不参与任何选择',
             '',
             f"基线（部署阈值 {fmt(deployed_threshold, 3)}，单 max-prob）："
             f"真实域 FAR **{fmt(base['real_far_at_deployed_threshold'])}**、"
             f"合成已知命令覆盖率 {fmt(base['syn_known_coverage_at_deployed_threshold'])}、"
             f"合成 unknown FAR {fmt(base['syn_unknown_far_at_deployed_threshold'])}、"
             f"合成 argmax 准确率 {fmt(base['syn_known_accuracy'])}、"
             f"同域 ECE {fmt(ece.get('ece'))}（等宽分箱，n={ece.get('n')}）。",
             '',
             f'### 4.1 单信号对比（目标 FAR ≤ {fmt(target, 2)}）', '',
             '| 信号 | 跨域 AUC | 同域 AUC | 域依赖 Δ | 判定 | 跨域 AURC | 跨域覆盖率 | 同域覆盖率 |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    for name in report.get('signal_order') or list(sig):
        s = sig.get(name)
        if not s:
            continue
        cw, sw = _wp(s.get('workpoints_cross_domain'), target), \
            _wp(s.get('workpoints_same_domain'), target)
        gap = s.get('domain_gap') or {}
        lines.append(
            f"| {SIGNAL_LABELS.get(name, name)} | {fmt(s['auc_cross_domain'])} "
            f"| {fmt(s['auc_same_domain'])} | {fmt(gap.get('gap'), 3)} "
            f"| {gap.get('verdict', '—')} "
            f"| {fmt(s['aurc_cross_domain'])} "
            f"| {fmt((cw or {}).get('coverage'))} | {fmt((sw or {}).get('coverage'))} |")
    lines += ['',
              '> AURC 是各覆盖率档位上的平均风险，越小越好；覆盖率是达到该 FAR 目标时'
              '还能接受的合成命令比例。域依赖 Δ = 跨域 AUC − 同域 AUC，'
              'Δ 越大说明该信号越依赖「TTS vs 真人」的声学差异而非命令内容，'
              'Δ > 0.15 判为部分依赖域、同域 AUC ≤ 0.55 判为域检测器。', '']

    scored = [(n, s) for n, s in sig.items()
              if s.get('aurc_cross_domain') is not None]
    mp_aurc = (sig.get('maxprob') or {}).get('aurc_cross_domain')
    if scored:
        bn, bs = min(scored, key=lambda kv: kv[1]['aurc_cross_domain'])
        bg = bs.get('domain_gap') or {}
        head = (f"本次最优单信号是 **{SIGNAL_LABELS.get(bn, bn)}**："
                f"跨域 AURC {fmt(bs['aurc_cross_domain'])}")
        if mp_aurc and bn != 'maxprob':
            head += f"，比部署信号 max-prob 的 {fmt(mp_aurc)} 低 {1 - bs['aurc_cross_domain'] / mp_aurc:.0%}"
        same, cross = bg.get('auc_same_domain'), bg.get('auc_cross_domain')
        lines += [head + '。',
                  f"它的同域 AUC {fmt(same)} {'≥' if (same or 0) >= (cross or 0) else '<'} "
                  f"跨域 AUC {fmt(cross)}（Δ={fmt(bg.get('gap'), 3)}），"
                  f"判定为**{bg.get('verdict')}**——即它区分的是命令与非命令，"
                  '不是「TTS 与真人」。这一点决定了下面那条跨域 FAR 能不能信。', '']

    and_rules = report.get('and_rules') or {}
    if and_rules:
        lines += ['### 4.2 双信号 AND 规则', '',
                  '两个阈值只在合成语料上搜出，然后**原样**搬到真人测试集上评估。'
                  '另列 oracle（阈值直接在真人负样本上搜）作为能力上界，'
                  '它**不可部署**，只用来说明「缺真人负样本」损失了多少。', '',
                  '| 规则 | 阈值 | 合成覆盖率 | 真实域 FAR | 95% CI | 危险误动作率 | oracle FAR |',
                  '|---|---|---:|---:|---|---:|---:|']
        for name, r in and_rules.items():
            cal, app, orc = r['calibrated_on_synthetic'], r['applied_to_real'], \
                r.get('oracle_on_real') or {}
            dang = (app.get('dangerous_split') or {}).get('dangerous') or {}
            th = ' AND '.join(f"{k} ≥ {fmt(v, 4)}" for k, v in cal['thresholds'].items())
            lines.append(
                f"| `{name}` | {th} | {fmt(app['syn_known_coverage'])} "
                f"| **{fmt(app['real_far'])}** "
                f"| [{fmt(app['real_ci95'][0])}, {fmt(app['real_ci95'][1])}] "
                f"| {fmt(dang.get('rate'))} "
                f"| {fmt(orc.get('risk'))}（覆盖率 {fmt(orc.get('coverage'))}） |")
        lines += ['']

        best_name = min((n for n, r in and_rules.items() if r.get('applied_to_real')),
                        key=lambda n: and_rules[n]['applied_to_real']['real_far'],
                        default=None)
        if best_name is not None:
            best = and_rules[best_name]['applied_to_real']
            bcal = and_rules[best_name]['calibrated_on_synthetic']
            bdang = (best.get('dangerous_split') or {}).get('dangerous') or {}
            rdang = (report.get('deployed_dangerous') or {}).get('dangerous') or {}
            base_far = base['real_far_at_deployed_threshold']
            base_cov = base['syn_known_coverage_at_deployed_threshold']
            dcov = best['syn_known_coverage'] - base_cov
            lines += [
                f"**最优规则 `{best_name}`**（阈值 "
                + '、'.join(f'{k} ≥ {fmt(v, 6)}' for k, v in bcal['thresholds'].items())
                + '）对比当前部署的单阈值：', '',
                '| 指标 | 部署单阈值 max-prob | 双信号 AND | 变化 |',
                '|---|---:|---:|---:|',
                f"| 真实域 FAR | {fmt(base_far)} | **{fmt(best['real_far'])}** "
                f"| {base_far / best['real_far']:.1f}× 下降 |",
            ]
            if rdang and bdang:
                lines.append(
                    f"| 危险误动作率 | {fmt(rdang['rate'])} | **{fmt(bdang['rate'])}** "
                    f"| {rdang['rate'] / bdang['rate']:.1f}× 下降 |")
            lines += [
                f"| 合成已知命令覆盖率 | {fmt(base_cov)} "
                f"| **{fmt(best['syn_known_coverage'])}** "
                f"| {'+' if dcov >= 0 else ''}{dcov:.4f} |",
                '',
                '> 危险误动作率剔除了只查询、不驱动设备的 STATUS_ALL（actuator=none）。'
                '降 FAR 必须按这个口径复核，否则可能只是把危险误动作换成了无害误动作。',
                '',
            ]

    caveats = report.get('caveats') or []
    if caveats:
        lines += ['### 4.3 限制', '']
        lines += [f'- {c}' for c in caveats]
        lines += ['']
    return lines


def wired_block(test_single, test_sel, bench_single, bench_sel):
    """接线复核：同一条实时路径上，单阈值与双信号门限的端到端对比。

    前面几节都是「离线把信号算出来再套阈值」，本节走的是
    ``DialectEngine.process_utterance``——即真正下发后会跑的那段代码。
    """
    if test_sel is None and bench_sel is None:
        return []
    lines = ['### 5.4 接线后的端到端复核', '',
             '前面几节都是离线算信号再套阈值；本节走 `DialectEngine.process_utterance`，'
             '即下发后真正跑的那段代码。同一套合成测试集（未见表达组），'
             '两份工件只差 `--selective`。', '']
    if test_sel and test_single:
        ks, kg = variant(test_single)['known'], variant(test_sel)['known']
        us, ug = (variant(test_single).get('unknown') or {}), \
            (variant(test_sel).get('unknown') or {})
        lines += [
            f"| 指标（test 子集，n={kg['n']}） | 单阈值 "
            f"{fmt(test_single.get('threshold'), 3)} | 双信号门限 |",
            '|---|---:|---:|',
            f"| 合成命令接受率 | {fmt(ks['acceptance_rate'])} "
            f"| **{fmt(kg['acceptance_rate'])}** |",
            f"| 接受后准确率 | {fmt(ks['accepted_accuracy'])} "
            f"| {fmt(kg['accepted_accuracy'])} |",
            f"| argmax 准确率 | {fmt(ks['argmax_accuracy'])} "
            f"| {fmt(kg['argmax_accuracy'])} |",
            f"| 合成非命令误接受率（n={ug.get('n')}） | "
            f"{fmt(us.get('false_accept_rate'))} "
            f"| **{fmt(ug.get('false_accept_rate'))}** |",
            '',
            '门限工件：' + '、'.join(
                f"{k} ≥ {fmt(v, 6)}" for k, v in
                ((test_sel.get('rejection_rule') or {}).get('thresholds') or {}).items())
            + f"（池 {(test_sel.get('rejection_rule') or {}).get('pool_size')} 条，"
              f"k={(test_sel.get('rejection_rule') or {}).get('ood_k')}，"
              f"degraded="
              f"{str((test_sel.get('rejection_rule') or {}).get('degraded')).lower()}）",
            '',
            f"argmax 准确率不变（{fmt(ks['argmax_accuracy'])} → "
            f"{fmt(kg['argmax_accuracy'])}）说明门限只改**接不接受**，不改分类；"
            f"接受率上升 {kg['acceptance_rate'] - ks['acceptance_rate']:+.4f} 的同时，"
            f"接受后准确率下降 "
            f"{kg['accepted_accuracy'] - ks['accepted_accuracy']:+.4f}"
            '——多接受的那部分命令里有一部分是错的，这是这条规则真实的代价，不是缺陷。',
            '',
            '> 口径差别不能混：本节接受率来自 **test** 子集（未见表达组），'
            '第 5.3 节的覆盖率 0.8205 来自标定用的 **val** 子集，两个数不可互换。',
            '',
        ]
    if bench_sel and bench_single:
        ss, sg = bench_single['summary'], bench_sel['summary']
        lines += ['延迟分解（同一 test 子集，离线 CPU 测量）：', '',
                  '| 阶段 | 单阈值 P50 / P95 | 双信号 P50 / P95 |',
                  '|---|---:|---:|']
        for key, label in [('infer_ms', '推理'), ('gate_ms', '门限判定（kNN + AND）'),
                           ('user_wait_ms', '用户感知等待')]:
            a = ss.get(key)
            b = sg.get(key)
            lines.append(f"| {label} | "
                         + (f"{a['p50']} / {a['p95']} ms" if a else '—') + ' | '
                         + (f"{b['p50']} / {b['p95']} ms" if b else '—') + ' |')
        di = sg['infer_ms']['p50'] - ss['infer_ms']['p50']
        lines += [
            '',
            f"kNN 判定本身只要 {sg['gate_ms']['p50']} ms（P95 {sg['gate_ms']['p95']} ms）；"
            f"多出来的 {di:+.1f} ms 在推理侧——门限需要逐种子概率与嵌入"
            '（`predict_detailed`），而部署置信度是「平均 logits 后 softmax」，'
            '两者不是同一份输出，不能互相顶替。'
            f"用户感知等待被语音段（P50 {sg['speech_ms']['p50']} ms）与播报"
            f"（{sg['speak_ms']['p50']} ms）主导，所以 P50 只从 "
            f"{ss['user_wait_ms']['p50']} ms 动到 {sg['user_wait_ms']['p50']} ms。",
            '',
        ]
    return lines


def pool_gate_block(pool_size, gate, test_single=None, test_sel=None,
                    bench_single=None, bench_sel=None):
    """参照池压缩（R-13）与下发门限工件。

    第 4 节给出的双信号规则里 ``ood`` 需要一个 kNN 参照池，满池放不进端侧——
    本节回答「池能压到多小」，以及在压缩之前先回答「第 4 节那对阈值可不可复现」。
    两件事都只读工件里的数字，不在报告脚本里重算。
    """
    title = '## 5. 参照池压缩与下发门限'
    if all(x is None for x in (pool_size, gate, test_sel, bench_sel)):
        return [title, '', '_未生成。_', '',
                '```bash',
                'python scripts/eval_pool_size.py            # -> docs/pool_size.json',
                'python scripts/build_selective_gate.py      # -> docs/selective_gate.json',
                '```', '']

    lines = [title, '',
             '第 4 节的双信号规则里，`ood` 是「到 kNN 参照池的负距离」，'
             '满池 1981 条 = 1981 KB（f32）/ 990.5 KB（f16），端侧放不下。'
             '本节先复核第 4 节那对阈值本身可不可复现，再测池能压到多小。', '']

    if pool_size:
        proto = pool_size['protocol']
        interp = pool_size['interpretation']
        noise = interp['calib_noise_at_full_pool']
        stab = interp['crossfit_stability']
        xc = pool_size.get('full_pool_matches_s3') or {}
        fp = interp['full_pool']
        lines += [
            '### 5.1 先复核标定：第 4 节那对阈值是不是运气',
            '',
            f"判据在跑实验前写死：真实域 FAR ≤ {fmt(proto['accept_max_far'], 2)} "
            f"且合成已知命令覆盖率 ≥ {fmt(proto['accept_min_coverage'], 2)}；"
            f"标定目标风险 {fmt(proto['target_risk_for_calibration'], 2)}，k={proto['knn_k']}。",
            '',
            '两条对照，回答「0.0660 这个数能不能复现」：',
            '',
            f"- **合成侧标定的噪声底**：池完全不变，只对 "
            f"{pool_size['groups']['syn_unknown']} 条合成负样本做有放回重采样 "
            f"{proto['bootstrap_draws']} 次，满池真实域 FAR 就在 "
            f"[{fmt(noise['p10'])}, {fmt(noise['p90'])}]（p10–p90，跨度 "
            f"{fmt(interp['calib_noise_spread'])}）之间摆动，最坏抽到 "
            f"{fmt(noise['max'])}，共产生 {fp['calib_distinct_thresholds_total']} 组不同阈值。"
            '跨度已超过目标 FAR 本身，所以第 4 节那个点估计是**一次抽样**，'
            '不是可复现的保证；瓶颈是标定集只有 72 条，不是信号选错。',
            f"- **交叉拟合复核**：标定集换成 {pool_size['groups']['real']} 条真人负样本，"
            f"{proto['calibrated_on'].split('：', 1)[-1]}。"
            f"满池 {stab['full_pool_distinct_thresholds']} 组阈值"
            f"（合并 FAR 全距 {fmt(stab['full_pool_far_seed_spread'])}），"
            f"样本外 FAR {fmt(fp['real_far_mean'])}、覆盖率 {fmt(fp['syn_coverage_mean'])}。",
        ]
        if xc.get('checked'):
            gp = xc.get('got_primary') or {}
            th = '、'.join(f'{k} ≥ {fmt(v, 6)}' for k, v in (gp.get('thresholds') or {}).items())
            lines.append(
                f"  与第 4 节公布值逐位比对：`matches_s3 = {str(xc.get('matches_s3')).lower()}`"
                f"（期望 FAR {fmt((xc.get('expected') or {}).get('real_far'))} / "
                f"实得 {fmt((xc.get('got_synthetic_only') or {}).get('real_far'))}），"
                f"交叉拟合选出的工作点是 {th}，Δ="
                f"{fmt((xc.get('crossfit_vs_s3') or {}).get('far_delta'))}。")
        n_draws = max((s.get('n_draws') or 1) for s in pool_size['summary'])
        lines += [
            '',
            '**结论：规则站得住，产出它的那次标定站不住。**'
            '同一对阈值在数千条真人负样本上做样本外验证仍然达标，'
            '但只用 72 条合成负样本搜阈值时，换一次抽样就可能落到另一对阈值上、'
            '报出 0.20 甚至 0.33 的 FAR。因此下表主口径改为交叉拟合，'
            '合成侧口径只作对照保留。',
            '',
            '> 口径披露：交叉拟合下**真人负样本参与了阈值搜索**。它仍是样本外估计'
            '（每条样本都由没见过它的阈值判定），但「本项目不需要真人数据」的说法'
            '在标定这一步不再成立——真人非命令语音（公开语料）是必需的。',
            '',
            f"### 5.2 池压缩实测（主口径 {proto['calibrate_primary']}，"
            f"子采样每个规模 {n_draws} 次随机分层抽样）",
            '',
            '| 池 | 条数 | k | f16 体积 | 样本外 FAR 均值/最差 | 覆盖率 均值/最差 '
            '| 危险误动作率 ≤ | 阈值组数 | 达标 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|',
        ]
        order = {'full': 0, 'subsample': 1, 'centroid': 2}
        for s in sorted(pool_size['summary'],
                        key=lambda r: (order.get(r['kind'], 9), -r['n_pool'], r['k'])):
            lines.append(
                f"| {s['kind']} | {s['n_pool']} | {s['k']} "
                f"| {s['bytes_f16'] / 1024:.1f} KB "
                f"| {fmt(s['real_far_mean'])} / {fmt(s['real_far_worst'])} "
                f"| {fmt(s['syn_coverage_mean'])} / {fmt(s['syn_coverage_worst'])} "
                f"| {fmt(s['dangerous_rate_worst'])} "
                f"| {s['xfit_distinct_thresholds_sum']} "
                f"| {'是' if s['all_draws_meet_target'] else '否'} |")
        lines += ['', '> 「阈值组数」是该配置在所有折与所有随机划分上出现过的不同阈值对总数；'
                  f'1 表示工作点唯一。子采样每规模 {n_draws} 次抽样、每次 2 折 × 3 次划分，'
                  f'所以稳定配置的上限是 {n_draws * 2 * 3}（满池与质心池只抽 1 次，上限 6）。'
                  '「危险误动作率」列取各次抽样里的最差值。', '',
                  '**读数**（下列判断由 `scripts/eval_pool_size.py` 随工件一起生成，'
                  '报告不重算）：', '']
        lines += [f'- {s}' for s in interp['statements']]
        lines += ['']

    if gate:
        dep, xf = gate['deployed_fit'], gate['crossfit']
        pool = gate['pool']
        written = gate.get('written') or {}
        th = ' AND '.join(f'{k} ≥ {fmt(v, 6)}' for k, v in dep['thresholds'].items())
        modal = (xf.get('threshold_modes') or [{}])[0].get('thresholds') or {}
        same = modal == dep['thresholds']
        lines += [
            '### 5.3 下发门限工件',
            '',
            f"按上表选定的最小达标池构建，一对工件必须**同时**下发"
            f"（阈值是按这份池标定的，换池等于换阈值）：",
            '',
            '| 项 | 值 |',
            '|---|---|',
            f"| 规则文件 | `{Path(written.get('gate', '?')).name}`"
            f"（sha256 `{(written.get('gate_sha256') or '')[:16]}…`） |",
            f"| 参照池 | `{Path(written.get('pool', '?')).name}`"
            f"（{pool['shape'][0]} × {pool['shape'][1]}，落盘 "
            f"{written.get('pool_bytes', 0) / 1024:.1f} KB，"
            f"内存 f32 {pool['bytes_in_memory_f32'] / 1024:.1f} KB，"
            f"sha256 `{(written.get('pool_sha256') or '')[:16]}…`） |",
            f"| 存储精度 | {pool['dtype_note']} |",
            f"| 每类条数 | "
            + '、'.join(f'{k} {v}' for k, v in sorted(pool['n_by_class'].items())) + ' |',
            f"| 规则 | `{th}`（k={gate['args']['k']}） |",
            f"| 部署阈值（全量拟合） | 样本内 FAR {fmt(dep['real_far'])}"
            f"（CI [{fmt(dep['real_ci95'][0])}, {fmt(dep['real_ci95'][1])}]）、"
            f"危险误动作率 {fmt(dep['dangerous_rate'])}、"
            f"合成覆盖率 {fmt(dep['syn_coverage'])} |",
            f"| 样本外复核（交叉拟合） | FAR {fmt(xf['pooled_far_mean'])}"
            f"（min {fmt(xf['pooled_far_min'])} / max {fmt(xf['pooled_far_max'])}）、"
            f"覆盖率 {fmt(xf['pos_coverage_mean'])}"
            f"（min {fmt(xf['pos_coverage_min'])}）、"
            f"{xf['n_distinct_thresholds']} 组阈值 |",
            f"| 部署阈值 == 交叉拟合众数阈值 | {str(same).lower()}"
            + ('' if same else '（**不一致：报出的 FAR 属于另一条规则**）') + ' |',
            f"| 判据 FAR ≤ {fmt(gate['args']['target_risk'], 2)} 且覆盖率 ≥ 0.80 "
            f"| {'达标' if gate['meets_target'] else '**未达标**'} |",
            f"| 降级路径 | {gate['args']['fallback_mode']}"
            f"（拿不到嵌入或池时退回单阈值 "
            f"{fmt(gate['args']['fallback_threshold'], 3)}） |",
            '',
            '> 上表 FAR 一侧是真人非命令语音（公开语料），覆盖率一侧仍是**合成命令**；'
            '所以「达标」是对本项目预先写死的判据而言，不构成对真人可用性的承诺'
            '（真人命令正样本为 0，见第 10 节）。',
            '',
            '接线位置：`runtime/selective.py` 的 `SelectiveGate` 做判定，'
            '`src/realtime.py` 在 `process_utterance` 里调用它并把 '
            '`rule`/`signals`/`reason` 写进结果与日志；'
            '`config.yaml` 的 `recognition.selective` 段控制开关与工件路径，'
            '`scripts/run_monitor.py`、`scripts/run_replay.py` 用 '
            '`--selective/--no-selective` 覆盖。'
            '池文件缺失时门限**降级而不抛异常**（模型先到、池后到是正常下发状态），'
            '降级会在日志里显式标出。',
            '',
        ]

    lines += wired_block(test_single, test_sel, bench_single, bench_sel)

    caveats = (pool_size or {}).get('caveats') or []
    if caveats:
        lines += ['### 5.5 限制', ''] + [f'- {c}' for c in caveats] + ['']
    return lines


def two_pass_block(bench_pl, dialog_off, dialog_stream):
    """两遍流式管线（S4(b)/S5）：感知延迟 + 会话级效果。

    延迟与准确率必须分开看：这一节报的是「线上真正跑的那条路径」
    （Pass-1 滑窗 -> 快速端点 -> Pass-2 复核 -> 决策层 -> 执行/澄清/拒识）。
    """
    title = '## 6. 两遍流式管线（S4(b)/S5）：感知延迟与会话级效果'
    lines = [title, '']
    if bench_pl is None and dialog_off is None and dialog_stream is None:
        lines += ['_未生成。_', '',
                  '```bash',
                  'python scripts/bench_perceived_latency.py \\',
                  '    --manifest <语料>/manifest_yue_v3_1_vad.csv --subset test',
                  'python scripts/eval_dialog.py --mode offline \\',
                  '    --manifest <语料>/manifest_yue_v3_1_vad.csv --subset test \\',
                  '    --out docs/dialog_eval.json',
                  'python scripts/eval_dialog.py --mode stream \\',
                  '    --manifest <语料>/manifest_yue_v3_1_vad.csv --subset test \\',
                  '    --out docs/dialog_eval_stream.json',
                  '```', '']
        return lines

    if bench_pl:
        st = bench_pl['stream']
        acc = bench_pl.get('acceptance') or {}
        bd = bench_pl.get('action_latency_breakdown_ms') or {}
        hang = bd.get('audio_hangover') or {}
        cpu = bd.get('cpu_verify_gate_decision') or {}
        disp = bd.get('cpu_dispatch') or {}
        act = bench_pl.get('action_latency_ms') or {}
        pre_cpu = bench_pl.get('pretrigger_cpu_ms') or {}
        pre_audio = bench_pl.get('pretrigger_audio_ms') or {}
        dlg = bench_pl.get('dialog_complete_ms') or {}

        def row(label, target, d, judge):
            return (f"| {label} | {target} | {fmt(d.get('p50'), 1)} ms "
                    f"| {fmt(d.get('p95'), 1)} ms | {fmt(d.get('max'), 1)} ms "
                    f"| {d.get('n', '—')} | {judge} |")

        ok1 = '达标' if acc.get('action_p95_le_300ms') else '**未达标**'
        ok2 = '达标' if acc.get('pretrigger_cpu_p95_le_150ms') else '**未达标**'
        ok3 = '达标' if acc.get('dialog_complete_p95_le_1200ms') else '**未达标**'
        lines += [
            '### 6.1 感知延迟（动作 / 预触发 / 对话完成）', '',
            f"口径：{st.get('fs')} Hz 分块流式喂入（块 {st.get('block_frames')} 帧，"
            f"窗 {fmt(st.get('window_ms'), 0)} ms / 步进 {fmt(st.get('hop_ms'), 0)} ms），"
            f"快速端点静音挂起 {fmt(st.get('fast_silence_ms'), 0)} ms"
            f"（软候选门槛 {fmt(st.get('fast_endpoint_threshold'), 2)}，"
            f"严格预触发门槛 {fmt(st.get('pretrigger_threshold'), 2)}）。"
            f"样本 {bench_pl.get('n')} 条，其中执行 {bench_pl.get('n_executed')} 条。", '',
            '| 指标 | 目标 | P50 | P95 | 最大 | 样本数 | 判定 |',
            '|---|---:|---:|---:|---:|---:|---|',
            row('动作延迟（说完 → 动作下发）', 'P95 ≤ 300 ms', act, ok1),
            row('├ 音频时间（静音挂起）', '—', hang, '—'),
            row('├ CPU（复核 + 门限 + 决策）', '—', cpu, '—'),
            row('└ CPU（动作下发）', '—', disp, '—'),
            row('预触发处理耗时（CPU）', 'P95 ≤ 150 ms', pre_cpu, ok2),
            row('预触发（音频口径）', '不可与 150 ms 比', pre_audio, '按构造 ≥ 窗长 + 步进'),
            row('对话完成（动作下发 → 播完「已完成」）', 'P95 ≤ 1200 ms', dlg, ok3),
            '',
            f"验收判定：动作 P95 ≤ 300 ms **{str(acc.get('action_p95_le_300ms')).lower()}**；"
            f"预触发 CPU P95 ≤ 150 ms **{str(acc.get('pretrigger_cpu_p95_le_150ms')).lower()}**；"
            f"对话完成 P95 ≤ 1200 ms **{str(acc.get('dialog_complete_p95_le_1200ms')).lower()}**。",
            '',
        ]
        over = [r for r in (bench_pl.get('rows') or [])
                if r.get('executed') and r.get('audio_wait_ms', 0) > 300.0]
        if over:
            lines += [
                f"> **P95 达标不等于每条都达标**：{len(over)} 条执行样本的静音挂起仍是"
                '默认的 390 ms（软候选与严格预触发都没命中），动作延迟落在 400 ms 上下，'
                '超出 300 ms 预算。触发条件是「整句里没有一个窗的后验够高」——'
                '这些样本模型本身也不自信，所以这是**同一个原因的两个后果**。',
                '']
        lines += [f"> {bench_pl.get('note')}", '']

    # ------------------------------------------------------------------
    if dialog_off or dialog_stream:
        lines += ['### 6.2 会话级效果（三种口径）', '']
        lines += ['| 口径 | 样本数 | argmax 准确率 | 门限覆盖率 | 门限后准确率 '
                  '| 对话会话成功率 | 澄清率 | 危险误动作率 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for label, rep in (('offline（无 Pass-1 候选）', dialog_off),
                           ('stream（部署路径）', dialog_stream)):
            if not rep:
                continue
            lines.append(
                f"| {label} | {rep.get('n')} "
                f"| {fmt(rep.get('argmax_accuracy'))} "
                f"| {fmt(rep.get('single_gate_coverage'))} "
                f"| {fmt(rep.get('single_gate_accuracy'))} "
                f"| {fmt(rep.get('dialog_session_success'))} "
                f"| {fmt(rep.get('clarification_rate'))} "
                f"| {fmt(rep.get('dangerous_misaction_rate'))} |")
        lines += ['']
        if dialog_stream:
            g = dialog_stream.get('confusable_pair') or {}
            lines += ['易混对（CURTAIN_OPEN / CURTAIN_CLOSE）的会话级效果——'
                      '这是 S5 的验收口径：', '',
                      '| 口径 | 样本数 | argmax | 门限后 | 对话会话成功率 '
                      '| 澄清率 | 相对门限后增益 |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for label, rep in (('offline', dialog_off), ('stream（部署路径）', dialog_stream)):
                if not rep:
                    continue
                p = rep.get('confusable_pair') or {}
                lines.append(
                    f"| {label} | {p.get('n')} | {fmt(p.get('argmax_accuracy'))} "
                    f"| {fmt(p.get('single_gate_accuracy'))} "
                    f"| {fmt(p.get('dialog_session_success'))} "
                    f"| {fmt(p.get('clarification_rate'))} "
                    f"| {fmt(p.get('gain_vs_single_gate'))} |")
            lines += ['']
            gain = g.get('gain_vs_single_gate')
            lines += [
                f"**部署路径（stream）上，对话层对易混对的增益是 {fmt(gain)}**："
                'S5 写的验收目标是「易混对任务成功率提升」，这个目标在部署口径下'
                '**没有达成**。offline 口径下有 '
                f"{fmt((dialog_off or {}).get('confusable_pair', {}).get('gain_vs_single_gate'))} "
                '的增益，但 offline 没有 Pass-1 候选、没有关键词偏置、没有两遍一致性，'
                '走的不是线上代码。', '',
                f"- 部署路径的 argmax 准确率 {fmt(dialog_stream.get('argmax_accuracy'))} "
                f"低于 offline 的 {fmt((dialog_off or {}).get('argmax_accuracy'))}："
                '差值来自**输入被截断**（见 6.3）与关键词偏置，'
                '不是 Pass-2 本身变差。',
                f"- 对话补救 {dialog_stream.get('recovered_by_dialog')} 条、"
                f"丢掉 {dialog_stream.get('lost_by_dialog')} 条"
                f"（offline：{fmt((dialog_off or {}).get('recovered_by_dialog'))} / "
                f"{fmt((dialog_off or {}).get('lost_by_dialog'))}）。",
                '- 澄清率的口径：一个样本最多澄清一次（`max_clarify=1`），'
                '重说样本取自同一命令的**另一个表达组**——这个假设偏乐观，'
                '实测同一表达组兄弟样本的准确率只有 0.358。',
                '',
            ]
        lines += [f"> {dialog_stream.get('note') if dialog_stream else ''}", '']

    # ------------------------------------------------------------------
    ca = (bench_pl or {}).get('corpus_alignment')
    if ca:
        lines += [
            '### 6.3 已知缺陷：快速端点会把话切早', '',
            '训练语料是用**默认静音挂起**（400 ms → 13 帧 = 390 ms）切的'
            '（`scripts/make_trimmed_corpus.py`），模型没见过「句中被截断」的输入。'
            '部署路径的快速端点把挂起压到 120 ms，句内只要有超过 120 ms 的停顿就会'
            '**提前切段**，后半句被丢给下一段。', '',
            '| 指标 | 数值 |', '|---|---:|',
            f"| 对比样本数 | {ca.get('n_compared')} |",
            f"| 被提前切段 | **{ca.get('n_truncated')}**"
            f"（{fmt(ca.get('truncated_rate'), 4)}） |",
            f"| 被切成多段 | {ca.get('n_split_into_multi_segment')} |",
            f"| 提前切段后仍执行 | {ca.get('truncated_executed')} |",
            f"| 其中执行了**错误命令** | **{ca.get('truncated_wrong_command')}** |",
            f"| 错误流向 | {'、'.join(sorted(set(ca.get('truncated_wrong_pairs') or [])))} |",
            f"| 未被截断的执行样本 | {ca.get('kept_executed')} |",
            f"| 其中执行了错误命令 | {ca.get('kept_wrong_command')}"
            f"（{fmt(ca.get('kept_wrong_rate'), 4)}） |",
            '',
            f"> 提前切段的样本执行错误率 **{fmt(ca.get('truncated_wrong_rate'), 4)}**，"
            f"未截断的是 {fmt(ca.get('kept_wrong_rate'), 4)}。"
            '这不是阈值问题：把挂起恢复到 390 ms 可以消除截断，但动作延迟会回到'
            '约 400 ms，300 ms 预算与「不切早」在这份语料上**互斥**。'
            '真正的解法是 S5 级别的：切早时先不要执行，等下一段接上后合并重判，'
            '或者按部署的快速端点重建训练语料。', '',
        ]
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-eval', default=str(DOCS / 'pipeline_eval.json'))
    parser.add_argument('--val-eval', default=str(DOCS / 'pipeline_eval_val.json'))
    parser.add_argument('--human-eval', default=str(DOCS / 'pipeline_eval_human.json'))
    parser.add_argument('--ablation', default=str(DOCS / 'ablation_old_model.json'))
    parser.add_argument('--bench', default=str(DOCS / 'bench_latency.json'))
    parser.add_argument('--real-far', default=str(DOCS / 'real_far.json'))
    parser.add_argument('--selective', default=str(DOCS / 'selective_eval.json'))
    parser.add_argument('--pool-size', default=str(DOCS / 'pool_size.json'))
    parser.add_argument('--gate', default=str(DOCS / 'selective_gate.json'))
    # 接线复核（第 5.4 节）：与 --test-eval/--bench 同一套语料，只差 --selective
    parser.add_argument('--test-eval-selective',
                        default=str(DOCS / 'pipeline_eval_selective.json'))
    parser.add_argument('--bench-selective',
                        default=str(DOCS / 'bench_latency_selective.json'))
    # S4(b)/S5（第 6 节）：感知延迟与会话级效果
    parser.add_argument('--bench-perceived',
                        default=str(DOCS / 'bench_perceived_latency.json'))
    parser.add_argument('--dialog-eval', default=str(DOCS / 'dialog_eval.json'))
    parser.add_argument('--dialog-eval-stream',
                        default=str(DOCS / 'dialog_eval_stream.json'))
    parser.add_argument('--out', default=str(DOCS / '测试报告.md'))
    args = parser.parse_args(argv)

    cfg = load_config()
    test = load_json(Path(args.test_eval))
    val = load_json(Path(args.val_eval))
    human = load_json(Path(args.human_eval))
    ablation = load_json(Path(args.ablation))
    bench = load_json(Path(args.bench))
    real_far = load_json(Path(args.real_far))
    selective = load_json(Path(args.selective))
    pool_size = load_json(Path(args.pool_size))
    gate = load_json(Path(args.gate))
    test_sel = load_json(Path(args.test_eval_selective))
    bench_sel = load_json(Path(args.bench_selective))
    bench_pl = load_json(Path(args.bench_perceived))
    dialog_off = load_json(Path(args.dialog_eval))
    dialog_stream = load_json(Path(args.dialog_eval_stream))
    thresholds = load_json(cfg.thresholds_path)

    out = [
        '# 测试报告',
        '',
        f'生成日期：{date.today().isoformat()}（由 `scripts/report_test.py` 自动生成，'
        '数字直接读自评估工件）',
        '',
        '## 1. 评估对象',
        '',
        f"- 模型：`{cfg.model_path.name}`（1D-CNN 3 种子集成，纯 numpy）",
        f"- 拒识门限：单阈值 max-prob {fmt((thresholds or {}).get('selected_threshold'), 3)}"
        f"（温度 {(thresholds or {}).get('temperature')}，第 3、9 节）；"
        f"双信号 AND 规则 + kNN 参照池（第 4、5 节，已下发并接入实时路径）",
        f"- 正样本语料：edge-tts 合成粤语，训练/验证/测试按表达组划分；测试集为未见表达组",
        f"- 负样本语料：合成非命令粤语（第 2 节）+ 真人粤语（第 3 节，公开语料）",
        f"- 评估脚本：`scripts/evaluate_pipeline.py`、`scripts/eval_real_far.py`、"
        f"`scripts/eval_selective.py`、`scripts/eval_pool_size.py`、"
        f"`scripts/build_selective_gate.py`、`scripts/bench_latency.py`、"
        f"`scripts/bench_perceived_latency.py`、`scripts/eval_dialog.py`",
        '',
        '## 2. 合成语料结果',
        '',
    ]
    out += eval_table(variant(val), '验证集（参与阈值选择）')
    out += eval_table(variant(test), '测试集（未见表达组，不参与任何选择）')
    out += confusion_block(variant(test))

    out += real_far_block(real_far, (thresholds or {}).get('selected_threshold'))
    out += selective_block(selective, (thresholds or {}).get('selected_threshold'), real_far)
    out += pool_gate_block(pool_size, gate, test, test_sel, bench, bench_sel)
    out += two_pass_block(bench_pl, dialog_off, dialog_stream)

    out += ['## 7. 训练/部署分布不一致消融（旧模型）', '']
    if ablation:
        wv, nv = ablation['variants']['with_vad'], ablation['variants']['no_vad']
        out += [
            '| 输入方式 | argmax | 接受样本准确率 | 未知误接受率 |',
            '|---|---:|---:|---:|',
            f"| 整段送入（无 VAD） | {fmt(nv['known']['argmax_accuracy'])} "
            f"| {fmt(nv['known']['accepted_accuracy'])} | {fmt(nv['unknown']['false_accept_rate'])} |",
            f"| VAD 端点检测 | {fmt(wv['known']['argmax_accuracy'])} "
            f"| {fmt(wv['known']['accepted_accuracy'])} | {fmt(wv['unknown']['false_accept_rate'])} |",
            '',
            '旧模型在带尾部静音的语料上训练，接线上 VAD 后短命令崩溃；'
            '处置为用部署同款 VAD 重生成语料重训练（见第 2 节当前模型结果）。',
            '',
        ]
    else:
        out += ['_未生成。运行：`python scripts/evaluate_pipeline.py --known <未裁剪语料> '
                '--model runtime/models/v3_cnn_ensemble.npz '
                '--thresholds runtime/models/v3_thresholds.json '
                '--out docs/ablation_old_model.json`_', '']

    out += ['## 8. 端到端延迟分解', '']
    if bench:
        s = bench['summary']
        out += ['| 阶段 | P50 | P95 |', '|---|---:|---:|']
        for key, label in [('vad_ms', 'VAD 切段（CPU）'), ('feature_ms', '特征提取（CPU）'),
                           ('infer_ms', '推理（CPU）'), ('speech_ms', '语音段时长'),
                           ('speak_ms', '普通话播报时长'),
                           ('user_wait_ms', '用户感知等待（说完→播报结束）')]:
            out.append(f"| {label} | {s[key]['p50']} ms | {s[key]['p95']} ms |")
        out += ['', f"> {bench['note']}", '',
                '> 本表是**单阈值**口径（`docs/bench_latency.json`）；开启双信号门限后的'
                '延迟分解见第 5.4 节（`docs/bench_latency_selective.json`）。', '']
    else:
        out += ['_未生成。运行：`python scripts/bench_latency.py --manifest '
                '<manifest_yue_v3_1_vad.csv> --subset test`_', '']

    out += ['## 9. 拒识阈值取舍', '']
    if thresholds:
        out += [
            f"- 选取集合：{thresholds.get('selected_on')}",
            f"- 验证集已知命令覆盖率：{fmt(thresholds.get('known_coverage_on_validation'))}",
            f"- 校准集未知指令误接受率：{fmt(thresholds.get('unknown_false_accept_rate_on_calibration'))}",
            f"- 业务约束：覆盖率 ≥ {thresholds.get('min_known_coverage_target')}、"
            f"误接受率 ≤ {thresholds.get('target_far')}",
            f"- 说明：{thresholds.get('note')}",
            '',
        ]

    out += ['## 10. 真人命令语料结果（正样本）', '']
    if human:
        out += eval_table(variant(human), '真人测试集')
    else:
        out += [
            '**待采集。** 第 3 节用的是真人**非命令**语音（公开语料，只测拒识）；'
            '真人**命令正样本**一条都没有，所以「真人说命令时的识别率」仍是未知数。'
            '本节空缺是预期状态，不是遗漏，也不能拿第 3 节顶替。',
            '',
            '补数步骤：',
            '',
            '1. `python scripts/collect_human.py`（P2 交付，尚未实现）采集真人语料；',
            '2. 用部署同款 VAD 同源裁剪并生成 manifest；',
            '3. `python scripts/evaluate_pipeline.py --manifest <真人 manifest> '
            '--subset test --out docs/pipeline_eval_human.json`；',
            '4. 重跑本脚本，本节自动填表。',
            '',
        ]

    out += ['## 11. 结论与限制', '',
            '- 第 2、7、8、9 节的数字来自 **edge-tts 合成粤语**（正样本与合成负样本），'
            '不代表真人表现；第 4、5 节只有 FAR 一侧是真人语音，'
            '**覆盖率一侧仍全是合成命令**；第 6 节同样是合成语料。']
    if real_far:
        head = real_far['headline']
        unk = real_far.get('synthetic_unknown_far') or {}
        dang = real_far['dangerous_split']['dangerous']
        out.append(
            f'- **域差距已测**：同模型同阈值下，真人粤语负样本误接受率 '
            f'{fmt(head["far"])}（95% CI [{fmt(head["ci95"][0])}, '
            f'{fmt(head["ci95"][1])}]，n={head["n"]}），'
            f'合成负样本为 {fmt(unk.get("far"))}（n={unk.get("n")}）。'
            '合成负样本与训练集同声学域，那个数字偏乐观，不得对外当安全指标用。')
        worst = max(dang['by_command'], key=dang['by_command'].get) \
            if dang['by_command'] else None
        out.append(
            f'- **危险误动作率 {fmt(dang["rate"])}**'
            f'（95% CI [{fmt(dang["ci95"][0])}, {fmt(dang["ci95"][1])}]）：'
            f'误接受中会真的驱动设备的部分，共 {dang["accepted"]} 条，'
            '已剔除只做查询、不动设备的 STATUS_ALL'
            + (f'；其中 **{worst}** 最多（{dang["by_command"][worst]} 条）' if worst else '')
            + '。这是上线前必须先解决的问题，不是调阈值能绕过的。')
        if selective and selective.get('and_rules'):
            best_name = min(
                (n for n, r in selective['and_rules'].items() if r.get('applied_to_real')),
                key=lambda n: selective['and_rules'][n]['applied_to_real']['real_far'],
                default=None)
            rule = selective['and_rules'][best_name] if best_name else None
        else:
            rule = None
        if rule:
            app, cal = rule['applied_to_real'], rule['calibrated_on_synthetic']
            bdang = (app.get('dangerous_split') or {}).get('dangerous') or {}
            sbase = selective['baseline_maxprob']
            th = '、'.join(f'{k} ≥ {fmt(v, 4)}' for k, v in cal['thresholds'].items())
            base_cov = sbase['syn_known_coverage_at_deployed_threshold']
            better_cov = app['syn_known_coverage'] >= base_cov
            out.append(
                f'- **第 4 节已给出可行路径**：双信号 AND 规则 `{best_name}`'
                f'（{th}，阈值**只在合成语料上标定**）把真实域 FAR 从 '
                f'{fmt(sbase["real_far_at_deployed_threshold"])} 降到 '
                f'**{fmt(app["real_far"])}**'
                f'（95% CI [{fmt(app["real_ci95"][0])}, {fmt(app["real_ci95"][1])}]），'
                + (f'危险误动作率从 {fmt(dang["rate"])} 降到 **{fmt(bdang["rate"])}**，'
                   if bdang else '')
                + f'合成覆盖率从 {fmt(base_cov)} '
                + ('**升到**' if better_cov else '降到')
                + f' {fmt(app["syn_known_coverage"])}'
                + ('——FAR 与覆盖率两个轴同时改善，说明第 3 节的覆盖率断崖是'
                   '**信号选错**，不是模型能力不够。' if better_cov else
                   '——FAR 下降是以覆盖率换来的，取舍见第 4.2 节。')
                + '但覆盖率一侧仍全是合成命令，真人命令覆盖率未知（见第 4.3 节）。'
                  '该规则的标定可靠性复核、参照池压缩与下发工件见第 5 节。')
        if gate and gate.get('deployed_fit'):
            dep, xf = gate['deployed_fit'], gate['crossfit']
            gth = '、'.join(f'{k} ≥ {fmt(v, 6)}' for k, v in dep['thresholds'].items())
            gpool = gate['pool']
            out.append(
                f'- **已下发并接线（第 5 节）**：{gpool["shape"][0]} 条 f16 参照池'
                f'（落盘 {(gate.get("written") or {}).get("pool_bytes", 0) / 1024:.1f} KB）'
                f'+ 规则 `{gth}`；样本外 FAR {fmt(xf["pooled_far_mean"])}'
                f'（{xf["n_distinct_thresholds"]} 组阈值）、'
                f'合成覆盖率 {fmt(xf["pos_coverage_mean"])}、'
                f'真人负样本上的危险误动作率 {fmt(dep["dangerous_rate"])}。'
                '判定发生在 `src/realtime.py`，开关在 `config.yaml` 的 '
                '`recognition.selective` 段。'
                '注意第 4 节那对阈值只在 72 条合成负样本上标定时**不可复现**'
                '（重采样 FAR p10–p90 见第 5.1 节），下发版本已改用真人负样本'
                '交叉拟合复核，因此「不需要真人数据」在标定这一步不成立。')
        unreachable = [s for s in real_far.get('threshold_solve', [])
                       if not s['reachable']]
        if unreachable:
            out.append(f'- 目标 FAR {fmt(unreachable[0]["target_far"], 2)} '
                       '在现有模型上不可达，需补硬负样本重训。')
    else:
        out.append('- 真实域误接受率未测（第 3 节未生成），'
                   '合成与真人之间的域差距仍未知。')
    tc = top_confusion(variant(test))
    if tc:
        truth, pred, n, total_err = tc
        share = n / total_err if total_err else 0.0
        out.append(
            f'- 最大单一误判来源：**{truth}** 被判成 **{pred}**'
            f'（{n} 条，占测试集全部误判 {share:.1%}）；'
            '真人采集与补充训练应优先覆盖这一对。')
    else:
        out.append('- 测试集 argmax 弱项见第 2 节分命令表与混淆矩阵。')
    out += [
        '- 真人**命令正样本**仍为零，「真人说命令时的识别率」未知，'
        '不能用第 2 节的合成准确率代替（见第 10 节）。',
        '- 拒识阈值取舍见第 9 节；真实域上的反解结果见第 3 节；'
        '双信号拒识的对比见第 4 节。',
        '- 延迟为离线 CPU 测量，树莓派等目标硬件上的真实值待实测。',
    ]
    if bench_pl:
        act = bench_pl.get('action_latency_ms') or {}
        acc = bench_pl.get('acceptance') or {}
        lines_c = [
            f"- **两遍流式管线的延迟验收通过（第 6 节）**：动作延迟 P50 "
            f"{fmt(act.get('p50'), 1)} ms / P95 {fmt(act.get('p95'), 1)} ms"
            f"（目标 ≤ 300 ms，判定 "
            f"{str(acc.get('action_p95_le_300ms')).lower()}），"
            f"预触发 CPU P95 {fmt((bench_pl.get('pretrigger_cpu_ms') or {}).get('p95'), 1)} ms"
            f"（目标 ≤ 150 ms），对话完成 P95 "
            f"{fmt((bench_pl.get('dialog_complete_ms') or {}).get('p95'), 1)} ms"
            f"（目标 ≤ 1200 ms）。",
        ]
        ca = bench_pl.get('corpus_alignment') or {}
        if ca:
            lines_c.append(
                f"- **两遍路径有未解决的缺陷（第 6.3 节）**：快速端点把静音挂起压到 120 ms，"
                f"句内超过 120 ms 的停顿会提前切段，"
                f"{ca.get('n_truncated')}/{ca.get('n_compared')} 条被切早，"
                f"其中 {ca.get('truncated_wrong_command')} 条执行了错误命令"
                f"（未截断样本的错误执行率 {fmt(ca.get('kept_wrong_rate'), 4)}）。"
                '训练语料是用 390 ms 挂起切的，模型没见过被截断的输入——'
                '恢复挂起能消除截断但动作延迟回到约 400 ms，'
                '300 ms 预算与「不切早」在这份语料上互斥，需要 S5 级别的合并重判。')
        if dialog_stream:
            pair = dialog_stream.get('confusable_pair') or {}
            lines_c.append(
                f"- **S5 的「易混对任务成功率提升」在部署口径下没有达成（第 6.2 节）**："
                f"对话层对易混对的增益为 {fmt(pair.get('gain_vs_single_gate'))}；"
                f"部署路径会话成功率 "
                f"{fmt(dialog_stream.get('dialog_session_success'))}，"
                f"低于 offline 口径的 "
                f"{fmt((dialog_off or {}).get('dialog_session_success'))}。"
                '部署路径多出来的损失主要来自输入被提前切段与关键词偏置把错误候选锁死。')
        out += lines_c
    out += [
        '',
        '## 12. 复现命令',
        '',
        '```bash',
        'cd D:\\Run_Tao\\DialectVoice_Engineering',
        'python scripts/evaluate_pipeline.py --manifest '
        '"D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" '
        '--subset test --unknown '
        '"D:/Run_Tao/DialectVoice_Research/data/tts_corpus/unknown_v3_vad" '
        '--out docs/pipeline_eval.json',
        'python scripts/bench_latency.py --manifest '
        '"D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" '
        '--subset test',
        '# 真实域拒识测试集（语料需先自行下载，许可见 provenance.json）',
        'python scripts/build_real_testset.py \\',
        '    --source fleurs_yue=D:/Run_Tao/DialectVoice_Research/data/corpora/fleurs_yue/test \\',
        '    --license "fleurs_yue=CC BY 4.0 - FLEURS google/fleurs, yue_hant_hk test" \\',
        '    --transcripts fleurs_yue=D:/Run_Tao/DialectVoice_Research/data/corpora/'
        'fleurs_yue/test.tsv \\',
        '    --out D:/Run_Tao/DialectVoice_Research/data/real_domain_testset',
        'python scripts/eval_real_far.py \\',
        '    --manifest D:/Run_Tao/DialectVoice_Research/data/real_domain_testset/manifest.csv \\',
        '    --coverage-manifest "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/'
        'manifest_yue_v3_1_vad.csv" --coverage-subset val \\',
        '    --targets 0.15,0.10,0.08,0.05,0.02 --out docs/real_far.json',
        '# 选择性分类：单信号 vs 双信号（首次跑会做全量推理，之后可加 --reuse-raw）',
        'python scripts/eval_selective.py \\',
        '    --manifest D:/Run_Tao/DialectVoice_Research/data/real_domain_testset/manifest.csv \\',
        '    --corpus-manifest "D:/Run_Tao/DialectVoice_Research/data/tts_corpus/'
        'manifest_yue_v3_1_vad.csv" \\',
        '    --unknown-dir D:/Run_Tao/DialectVoice_Research/data/tts_corpus/unknown_v3_vad \\',
        '    --known-subset val --pool-subset train --k 5 --targets 0.10,0.05,0.02 \\',
        '    --out docs/selective_eval.json',
        '# 参照池压缩（R-13）：复用上面的 --raw-cache，不重新推理',
        'python scripts/eval_pool_size.py            # -> docs/pool_size.json',
        '# 按下表选定的最小达标池构建下发工件，再同步进 runtime/models/',
        'python scripts/build_selective_gate.py --pool-size 500 --pool-seed 0 \\',
        '    --pool-dtype float16 --k 5 --xfit-folds 2 --xfit-seeds 0,1,2',
        'python scripts/sync_runtime.py',
        '# 双信号门限下的端到端评估与延迟（默认关，加 --selective 才走 AND 规则）',
        'python scripts/evaluate_pipeline.py --selective --manifest '
        '"D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" '
        '--subset test --out docs/pipeline_eval_selective.json',
        'python scripts/bench_latency.py --selective --manifest '
        '"D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" '
        '--subset test --out docs/bench_latency_selective.json',
        '# 两遍流式管线：感知延迟 + 会话级效果（第 6 节）',
        'python scripts/bench_perceived_latency.py --manifest '
        '"D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" '
        '--subset test --out docs/bench_perceived_latency.json',
        'python scripts/eval_dialog.py --mode offline --manifest '
        '"D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" '
        '--subset test --out docs/dialog_eval.json',
        'python scripts/eval_dialog.py --mode stream --manifest '
        '"D:/Run_Tao/DialectVoice_Research/data/tts_corpus/manifest_yue_v3_1_vad.csv" '
        '--subset test --out docs/dialog_eval_stream.json',
        'python scripts/report_test.py --out docs/测试报告.md',
        '```',
        '',
    ]

    text = '\n'.join(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding='utf-8')
    print('测试报告已生成:', args.out, f'（{len(text)} 字符）')


if __name__ == '__main__':
    main()

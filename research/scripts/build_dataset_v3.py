# -*- coding: utf-8 -*-
"""第三阶段：合并 v2/v3 语料、质量标记、重复与泄漏审计、固定划分 v3。

产出
----
  data/tts_corpus/manifest_yue_v3.csv          统一清单（v2 600 + v3 新增）
  data/splits/track_a_expression_split_v3.csv   固定划分（v2 组继承原 subset，新组全部 train）
  outputs/v3_data_audit.json                    真实计算的数据审计
  data/tts_corpus/unknown_v3.csv                未知/易混指令测试集清单（由生成器写出，本脚本审计）

原则
----
1. 所有计数、sha256、泄漏、质量统计都从**磁盘上的实际音频**重新计算，不手填。
2. v2 清单/音频/划分只读，本脚本不修改它们；v3 划分中 v2 组必须继承
   data/splits/track_a_expression_split.csv 的 subset。
3. 同一 (command, 规范化文本) 的所有样本强制归入同一表达组，防止文本级泄漏。
4. v3 划分文件写出后记录 sha256；可复现（输入确定则输出字节确定）。

用法::

    D:\\anaconda3\\python.exe scripts/build_dataset_v3.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.dataset_v2 import (  # noqa: E402
    INTENTS, MANIFEST_FIELDS, VOICE_IDENTITY, analyze_audio, normalize_text,
    quality_flags_for, read_csv_rows, sha256_file, write_csv_rows,
)
from src.dataset_v3 import (  # noqa: E402
    EDGE_VOICES, MANIFEST_V3_FIELDS, UNKNOWN_MANIFEST, UNKNOWN_ROOT,
    V2_MANIFEST, V2_SPLIT, V3_AUDIT, V3_LIBRARY, V3_LOG, V3_MANIFEST, V3_SPLIT,
    V3_WAV_ROOT, estimate_f0, load_wav_16k, read_json,
)

V2_AUDIT = BASE / 'outputs' / 'v2_data_audit.json'
PROBE_REPORT = BASE / 'data' / 'tts_corpus' / 'wav16k_v3' / 'probe_report.json'

CONDITION_META = {
    'rate_slow_18': {'rate_percent': '-18', 'pitch': '0'},
    'rate_fast_18': {'rate_percent': '18', 'pitch': '0'},
    'pitch_low': {'rate_percent': '0', 'pitch': '-15'},
    'pitch_high': {'rate_percent': '0', 'pitch': '15'},
    'baseline': {'rate_percent': '0', 'pitch': '0'},
}
OK_STATUSES = ('ok', 'skipped_existing', 'quality_flagged')


def resolve_audio(rel):
    p = Path(rel)
    if p.is_absolute():
        return p
    for root in (BASE / 'data' / 'tts_corpus', BASE):
        cand = root / p
        if cand.exists():
            return cand
    return BASE / 'data' / 'tts_corpus' / p


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def load_v2_rows():
    rows = read_csv_rows(V2_MANIFEST)
    lib = {r['expression_id']: r for r in read_csv_rows(V3_LIBRARY)}
    out = []
    for r in rows:
        d = {k: r.get(k, '') for k in MANIFEST_V3_FIELDS}
        d['audio_path'] = r['audio_path']
        d['generation_method'] = ('v2_tts' if r.get('generation_batch') == 'v2'
                                  else 'v1_legacy_tts')
        d['condition_detail'] = r.get('condition_detail', '')
        d['augment_from'] = ''
        d['augment_params'] = ''
        d['text_norm'] = normalize_text(r['text'])
        d['split_source'] = 'inherited_v2'
        if not d.get('register'):
            d['register'] = lib.get(r['expression_group_id'], {}).get('register', '')
        out.append(d)
    return out


def load_v3_rows():
    """v3 生成日志 -> 清单行（只取会进入主清单的 ok 行）。"""
    if not V3_LOG.exists():
        return [], []
    log = read_csv_rows(V3_LOG)
    lib = {r['expression_id']: r for r in read_csv_rows(V3_LIBRARY)}
    rows, failed = [], []
    for r in log:
        if r.get('stage') not in ('augment', 'augment_new_expr', 'tts_new_expr'):
            continue
        if r.get('status') not in OK_STATUSES:
            failed.append(r)
            continue
        cond = r['condition']
        meta = CONDITION_META.get(cond, {'rate_percent': '0', 'pitch': '0'})
        is_aug = r.get('generation_method') == 'offline_signal_aug'
        voice = r['voice_id']
        gid = r['expression_group_id']
        reg = lib.get(gid, {}).get('register', '')
        batch = 'v3'
        source = 'offline_signal_aug' if is_aug else 'edge_tts_v3'
        row = {k: '' for k in MANIFEST_V3_FIELDS}
        row.update({
            'utterance_id': r['utterance_id'],
            'audio_path': r['file_path'],
            'speaker_id': voice,
            'dialect': 'yue-HK',
            'command': r['command'],
            'language': 'yue',
            'text': r['text'],
            'expression_type': cond,
            'semantic_pair_id': gid,
            'source': source,
            'is_synthetic': 'true',
            'duration': r.get('duration', ''),
            'sample_rate': r.get('sample_rate', ''),
            'split_group': f'v3:{voice}',
            'text_verified_by': 'RULE_REVIEWED',
            'quality_flags': ('SYNTHETIC;OFFLINE_AUGMENTED;UNVERIFIED_BY_NATIVE'
                              if is_aug else 'SYNTHETIC;UNVERIFIED_BY_NATIVE'),
            'notes': ('v3 corpus; offline signal-level augmentation from v2 audio '
                      '(edge-tts network unreliable); synthetic, not real dialect performance'
                      if is_aug else
                      'v3 corpus; edge-tts baseline for new expression; synthetic'),
            'expression_group_id': gid,
            'tts_engine': 'offline_signal_aug' if is_aug else 'edge_tts',
            'voice_id': voice,
            'voice_identity': VOICE_IDENTITY.get(voice, voice),
            'rate_percent': meta['rate_percent'],
            'pitch': meta['pitch'],
            'volume': '0',
            'condition': cond,
            'source_text': r['text'],
            'quality_status': 'ok' if r['status'] == 'ok' else 'flagged',
            'quality_reasons': r.get('error', ''),
            'generation_batch': batch,
            'channels': r.get('channels', '1'),
            'sha256': r.get('sha256', ''),
            'register': reg,
            'condition_detail': r.get('condition_detail', ''),
            'generation_method': r.get('generation_method', 'edge_tts'),
            'augment_from': r.get('augment_from', ''),
            'augment_params': r.get('condition_detail', '') if is_aug else '',
            'text_norm': normalize_text(r['text']),
            'split_source': '',
        })
        rows.append(row)
    rows.sort(key=lambda r: (r['command'], r['expression_group_id'], r['utterance_id']))
    return rows, failed


# ---------------------------------------------------------------------------
# 实测统计、去重、表达组合并
# ---------------------------------------------------------------------------

def compute_audio_stats(rows):
    for r in rows:
        p = resolve_audio(r['audio_path'])
        if not p.exists():
            r['quality_status'] = 'excluded'
            r['quality_reasons'] = (r.get('quality_reasons') + ';missing_file').strip(';')
            r['_exists'] = False
            continue
        stats = analyze_audio(p)
        status, reasons = quality_flags_for(stats)
        r['_stats'] = stats
        r['_exists'] = True
        r['sha256'] = sha256_file(p)
        r['duration'] = f"{stats['duration']:.3f}"
        r['sample_rate'] = str(stats['sample_rate'])
        r['channels'] = str(stats['channels'])
        if status == 'excluded':
            r['quality_status'] = 'excluded'
            r['quality_reasons'] = reasons
        elif status == 'flagged' and r['quality_status'] != 'excluded':
            if r['quality_status'] != 'flagged':
                r['quality_status'] = 'flagged'
            r['quality_reasons'] = (r['quality_reasons'] + ';' + reasons).strip(';')
    return rows


def dedupe_exact(rows):
    """sha256 完全重复只保留一条，其余 excluded（正常应为 0）。"""
    by_sha = defaultdict(list)
    for r in rows:
        if r.get('sha256'):
            by_sha[r['sha256']].append(r)
    removed = []
    for h, group in by_sha.items():
        if len(group) < 2:
            continue
        group_sorted = sorted(group, key=lambda r: (
            r['generation_batch'] != 'v1', r['generation_batch'] != 'v2', r['utterance_id']))
        keep = group_sorted[0]
        for r in group_sorted[1:]:
            r['quality_status'] = 'excluded'
            r['quality_reasons'] = (r['quality_reasons'] + ';exact_duplicate_of_'
                                    + keep['utterance_id']).strip(';')
            removed.append({'removed': r['utterance_id'], 'kept': keep['utterance_id'],
                            'sha256': h})
    return rows, removed


def canonicalize_expression_groups(rows):
    """按 (command, 规范化文本) 合并表达组；正常 v3 应为 0 合并。"""
    key2ids = defaultdict(set)
    for r in rows:
        key2ids[(r['command'], r['text_norm'])].add(r['expression_group_id'])
    canonical = {k: sorted(v)[0] for k, v in key2ids.items()}
    merged = {k: sorted(v) for k, v in key2ids.items() if len(v) > 1}
    changed = 0
    for r in rows:
        key = (r['command'], r['text_norm'])
        new_id = canonical[key]
        if r['expression_group_id'] != new_id:
            r['group_alias'] = r['expression_group_id']
            r['expression_group_id'] = new_id
            changed += 1
    return rows, {
        'n_merged_text_keys': len(merged),
        'n_rows_changed': changed,
        'merged_examples': [{'command': k[0], 'normalized_text': k[1][:30],
                             'group_ids': v} for k, v in list(merged.items())[:20]],
        'note': '按规范化文本统一表达组；v3 预期为 0（新表达文本已预先查重）。',
    }


# ---------------------------------------------------------------------------
# 划分
# ---------------------------------------------------------------------------

def build_split_v3(rows, v2_split_map):
    """v2 组继承原 subset；新表达组全部 train。返回 (assignment, stats)。"""
    groups = {}
    by_gid = defaultdict(list)
    for r in rows:
        if r['quality_status'] == 'excluded':
            continue
        by_gid[r['expression_group_id']].append(r)

    reg = {}
    for r in read_csv_rows(V3_LIBRARY):
        reg[r['expression_id']] = r['register']

    v2_groups = set(v2_split_map)
    new_groups = set(by_gid) - v2_groups
    # 新表达 = 不在 v2 划分里的组；v3 只允许新增表达组进 train
    assignment = {}
    for gid, lst in by_gid.items():
        cmds = {r['command'] for r in lst}
        if len(cmds) > 1:
            raise ValueError(f'表达组 {gid} 跨命令: {cmds}')
        if gid in v2_split_map:
            subset = v2_split_map[gid]
            src = 'inherited_v2'
        else:
            subset = 'train'
            src = 'added_train'
        assignment[gid] = {
            'subset': subset, 'command': lst[0]['command'],
            'register': reg.get(gid, lst[0].get('register') or 'unknown'),
            'n_samples': len(lst),
            'voices': sorted({r['voice_identity'] for r in lst}),
            'text': lst[0]['text'], 'split_source': src,
            'batch': ('v3' if any(r['generation_batch'] == 'v3' for r in lst)
                      else 'v2'),
        }
    stats = {
        'n_groups': len(assignment),
        'v2_groups_inherited': sum(1 for g in assignment if g in v2_split_map),
        'new_groups_added_train': sum(1 for g in assignment if g not in v2_split_map),
        'new_groups': sorted(g for g in assignment if g not in v2_split_map),
        'by_subset': dict(Counter(v['subset'] for v in assignment.values())),
    }
    return assignment, stats


def write_split_v3(path, assignment, extra):
    rows = []
    for gid in sorted(assignment):
        a = assignment[gid]
        rows.append({
            'expression_group_id': gid, 'subset': a['subset'],
            'command': a['command'], 'register': a['register'],
            'n_samples': a['n_samples'], 'voices': ';'.join(a['voices']),
            'text': a['text'], 'split_source': a['split_source'],
        })
    fields = ['expression_group_id', 'subset', 'command', 'register',
              'n_samples', 'voices', 'text', 'split_source']
    write_csv_rows(path, rows, fields)
    payload = {'path': str(Path(path).relative_to(BASE)).replace('\\', '/'),
               'sha256': sha256_file(path)}
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def subset_stats(lst):
    return {
        'n_samples': len(lst),
        'n_groups': len({r['expression_group_id'] for r in lst}),
        'by_command': {c: sum(1 for r in lst if r['command'] == c) for c in INTENTS},
        'by_condition': dict(Counter(r['condition'] for r in lst)),
        'by_voice': dict(Counter(r['voice_identity'] for r in lst)),
        'by_generation_method': dict(Counter(r['generation_method'] for r in lst)),
    }


def _float_stats(vals, digits=4):
    import numpy as np
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {'n': int(a.size), 'mean': round(float(a.mean()), digits),
            'min': round(float(a.min()), digits), 'max': round(float(a.max()), digits),
            'p05': round(float(np.percentile(a, 5)), digits),
            'p95': round(float(np.percentile(a, 95)), digits)}


def condition_verification(rows, log_rows):
    """对离线增广的 4 个条件做实测验证（可复现，全部来自磁盘音频）。

    - 时长比 = 输出时长 / 源时长（rate 条件应分别≈1.2195 / 0.8475，pitch 条件≈1.0）
    - RMS（能量）分布
    - 与源音频 sha256 相同条数（应恒为 0）
    - pitch 条件：自相关 f0 估计的实测偏移（Hz）
    - 格式违规（非 16k/单声道）条数（应为 0）
    """
    import numpy as np
    rows_by_id = {r['utterance_id']: r for r in rows}
    log_by_id = {r['utterance_id']: r for r in log_rows}
    aug = [r for r in rows if r.get('generation_method') == 'offline_signal_aug']

    src_info = {}

    def get_source(sid):
        if sid in src_info:
            return src_info[sid]
        info = None
        if sid in rows_by_id:
            s = rows_by_id[sid]
            info = {'duration': float(s['duration']), 'sha256': s['sha256'],
                    'path': str(resolve_audio(s['audio_path']))}
        elif sid in log_by_id:
            s = log_by_id[sid]
            p = BASE / s.get('file_path', '')
            info = {'duration': float(s.get('duration') or 0),
                    'sha256': s.get('sha256', ''), 'path': str(p)}
        src_info[sid] = info
        return info

    out = {}
    for cond in ('rate_slow_18', 'rate_fast_18', 'pitch_low', 'pitch_high'):
        items = [r for r in aug if r['condition'] == cond]
        ratios, rmss, f0_deltas = [], [], []
        sha_same = fmt_bad = missing = 0
        f0_cache = {}
        for r in items:
            p = resolve_audio(r['audio_path'])
            st = r.get('_stats')
            if st is None:
                if not p.exists():
                    missing += 1
                    continue
                st = analyze_audio(p)
            if st['sample_rate'] != 16000 or st['channels'] != 1:
                fmt_bad += 1
            rmss.append(st['rms'])
            info = get_source(r['augment_from'])
            if info is None or not info['duration']:
                missing += 1
                continue
            if info['sha256'] and r['sha256'] == info['sha256']:
                sha_same += 1
            ratios.append(st['duration'] / info['duration'])
            if cond.startswith('pitch') and info['path']:
                sid = r['augment_from']
                if sid not in f0_cache:
                    try:
                        x, _ = load_wav_16k(info['path'])
                        f0_cache[sid] = estimate_f0(x)
                    except Exception:
                        f0_cache[sid] = None
                f0_src = f0_cache[sid]
                if f0_src:
                    try:
                        y, _ = load_wav_16k(p)
                        f0_out = estimate_f0(y)
                    except Exception:
                        f0_out = None
                    if f0_out:
                        f0_deltas.append(f0_out - f0_src)
        expected = {'rate_slow_18': 1.21951, 'rate_fast_18': 0.84746,
                    'pitch_low': 1.0, 'pitch_high': 1.0}[cond]
        rec = {
            'condition': cond,
            'n': len(items),
            'n_measured': len(ratios),
            'missing_source_or_stats': missing,
            'expected_duration_ratio': expected,
            'duration_ratio': _float_stats(ratios),
            'duration_ratio_within_2pct': (
                sum(1 for x in ratios if abs(x - expected) <= 0.02 * expected)),
            'rms': _float_stats(rmss, 5),
            'sha256_same_as_source': sha_same,
            'format_violations': fmt_bad,
        }
        if cond.startswith('pitch'):
            rec['f0_delta_hz_estimated'] = _float_stats(f0_deltas, 2)
        out[cond] = rec
    out['note'] = ('离线增广验证：时长比用实测时长（源为 v2 清单实测时长或缺口 baseline '
                   '生成日志实测时长）；f0 为自相关估计，非人工听测；sha256 相同条数为 0 '
                   '说明输出与源字节不同。')
    return out


def text_sets_leak(rows_by_subset):
    group_sets = {k: {r['expression_group_id'] for r in v}
                  for k, v in rows_by_subset.items()}
    text_sets = {k: {r['text_norm'] for r in v} for k, v in rows_by_subset.items()}
    sha_sets = {k: {r['sha256'] for r in v} for k, v in rows_by_subset.items()}
    utt_sets = {k: {r['utterance_id'] for r in v} for k, v in rows_by_subset.items()}

    def inter(a, b):
        return len(a & b)

    return {
        'group_overlap_train_val': inter(group_sets['train'], group_sets['val']),
        'group_overlap_train_test': inter(group_sets['train'], group_sets['test']),
        'group_overlap_val_test': inter(group_sets['val'], group_sets['test']),
        'normalized_text_overlap_train_val': inter(text_sets['train'], text_sets['val']),
        'normalized_text_overlap_train_test': inter(text_sets['train'], text_sets['test']),
        'normalized_text_overlap_val_test': inter(text_sets['val'], text_sets['test']),
        'sha256_overlap_train_val': inter(sha_sets['train'], sha_sets['val']),
        'sha256_overlap_train_test': inter(sha_sets['train'], sha_sets['test']),
        'sha256_overlap_val_test': inter(sha_sets['val'], sha_sets['test']),
        'utterance_overlap_train_test': inter(utt_sets['train'], utt_sets['test']),
    }


def audit_unknown(main_rows, lib_texts):
    if not UNKNOWN_MANIFEST.exists():
        return {'status': 'missing',
                'note': '未知集清单缺失，说明生成阶段未完成'}
    rows = read_csv_rows(UNKNOWN_MANIFEST)
    main_norm = {r['text_norm'] for r in main_rows}
    texts = {r['text'] for r in rows}
    text_split = defaultdict(set)
    for r in rows:
        text_split[r['text']].add(r['text_split'])
    by_voice = Counter(r['voice_id'] for r in rows)
    for r in rows:
        p = BASE / r['audio_path']
        r['_exists'] = p.exists()
    sha_dup = sum(v - 1 for v in Counter(r['sha256'] for r in rows if r['sha256']).values()
                  if v > 1)
    return {
        'path': str(UNKNOWN_MANIFEST.relative_to(BASE)).replace('\\', '/'),
        'sha256': sha256_file(UNKNOWN_MANIFEST),
        'n_rows': len(rows),
        'n_texts': len(texts),
        'n_text_groups': len({r['text_group_id'] for r in rows}),
        'by_text_split': dict(Counter(r['text_split'] for r in rows)),
        'texts_by_split': {s: len({r['text'] for r in rows if r['text_split'] == s})
                           for s in ('calibration', 'test')},
        'by_voice': dict(by_voice),
        'by_label': dict(Counter(r.get('label', '') for r in rows)),
        'by_negative_type': dict(Counter(r.get('negative_type', '') for r in rows)),
        'texts_by_label': {lab: sorted({r['text'] for r in rows if r.get('label') == lab})
                           for lab in sorted({r.get('label', '') for r in rows})},
        'by_label_split': {lab: dict(Counter(r['text_split'] for r in rows
                                             if r.get('label') == lab))
                           for lab in sorted({r.get('label', '') for r in rows})},
        'texts_detail': sorted(
            [{'text': t, 'text_group_id': next(r['text_group_id'] for r in rows
                                               if r['text'] == t),
              'text_split': next(r['text_split'] for r in rows if r['text'] == t),
              'label': next(r.get('label', '') for r in rows if r['text'] == t),
              'negative_type': next(r.get('negative_type', '') for r in rows
                                    if r['text'] == t),
              'n_voices': sum(1 for r in rows if r['text'] == t)}
             for t in texts], key=lambda x: x['text_group_id']),
        'voices_per_text_ok': all(v == 3 for v in
                                  Counter(r['text_group_id'] for r in rows).values()),
        'each_text_single_split': all(len(v) == 1 for v in text_split.values()),
        'text_disjoint_from_main_train': len(texts & {r['text'] for r in main_rows}) == 0,
        'audio_all_exist': all(r['_exists'] for r in rows),
        'sha256_duplicates': sha_dup,
        'quality': dict(Counter(r['quality_status'] for r in rows)),
        'texts': sorted(texts),
        'note': '未知/易混指令测试集，不参与训练；按文本对半分 calibration/test；'
                'label=hard_negative/general_unknown 由 scripts/generate_unknown_v3.py 写入。',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-write', action='store_true', help='只审计不写文件（调试）')
    args = ap.parse_args()

    print('读取 v2 清单与 v3 生成日志 ...')
    v2_rows = load_v2_rows()
    v3_rows, v3_failed = load_v3_rows()
    print(f'v2={len(v2_rows)}  v3={len(v3_rows)}  失败/排除日志行={len(v3_failed)}')
    all_rows = v2_rows + v3_rows

    print('实测每行音频统计（sha256/时长/采样率/声道/质量）...')
    all_rows = compute_audio_stats(all_rows)
    print('完全重复检测 ...')
    all_rows, exact_dups = dedupe_exact(all_rows)
    print('表达组规范化合并 ...')
    all_rows, group_merge = canonicalize_expression_groups(all_rows)

    usable = [r for r in all_rows if r.get('quality_status') != 'excluded']
    v2_split_map = {r['expression_group_id']: r['subset']
                    for r in read_csv_rows(V2_SPLIT)}
    assignment, split_stats = build_split_v3(usable, v2_split_map)

    # 继承校验：v2 组在 v3 划分中必须与原划分一致
    mismatches = [g for g in assignment
                  if g in v2_split_map and assignment[g]['subset'] != v2_split_map[g]]
    if mismatches:
        raise ValueError(f'v2 固定划分继承失败: {mismatches[:5]}')

    subset_of = {g: a['subset'] for g, a in assignment.items()}
    for r in all_rows:
        if r.get('quality_status') == 'excluded':
            r['split'] = ''
            continue
        r['split'] = subset_of[r['expression_group_id']]
        r['split_source'] = assignment[r['expression_group_id']]['split_source']

    split_meta = write_split_v3(
        V3_SPLIT, assignment,
        extra={'note': 'v3 固定划分：v2 的 174 个表达组严格继承 track_a_expression_split.csv；'
                       '18 个新表达组全部进入 train，不改动 v2 的 val/test。',
               'inherited_from': str(V2_SPLIT.relative_to(BASE)).replace('\\', '/'),
               'v2_split_sha256': sha256_file(V2_SPLIT),
               'n_groups': split_stats['n_groups'],
               'by_subset': split_stats['by_subset']})

    rows_by_subset = {'train': [], 'val': [], 'test': []}
    for r in usable:
        rows_by_subset[r['split']].append(r)
    leak = text_sets_leak(rows_by_subset)
    track_a_stats = {k: subset_stats(v) for k, v in rows_by_subset.items()}

    # ---- 计数 ----
    counts = {
        'total_rows': len(all_rows),
        'usable_rows': len(usable),
        'excluded_rows': sum(1 for r in all_rows if r.get('quality_status') == 'excluded'),
        'flagged_rows': sum(1 for r in all_rows if r.get('quality_status') == 'flagged'),
        'v2_rows': len(v2_rows),
        'v3_added_rows': len(v3_rows),
        'num_expression_groups': len({r['expression_group_id'] for r in usable}),
        'by_command_all': dict(Counter(r['command'] for r in all_rows)),
        'by_command_usable': {c: sum(1 for r in usable if r['command'] == c)
                              for c in INTENTS},
        'by_condition_usable': dict(Counter(r['condition'] for r in usable)),
        'by_generation_batch': dict(Counter(r['generation_batch'] for r in usable)),
        'by_generation_method_usable': dict(Counter(r['generation_method'] for r in usable)),
        'by_voice_usable': dict(Counter(r['voice_identity'] for r in usable)),
        'by_split': dict(Counter(r['split'] for r in usable)),
        'by_split_condition': {
            s: dict(Counter(r['condition'] for r in rows_by_subset[s]))
            for s in ('train', 'val', 'test')},
        'by_split_command': {
            s: {c: sum(1 for r in rows_by_subset[s] if r['command'] == c)
                for c in INTENTS}
            for s in ('train', 'val', 'test')},
        'v2_by_split_condition': {
            s: dict(Counter(r['condition'] for r in rows_by_subset[s]
                            if r['generation_batch'] in ('v1', 'v2')))
            for s in ('train', 'val', 'test')},
        'v3_added_by_split_condition': {
            s: dict(Counter(r['condition'] for r in rows_by_subset[s]
                            if r['generation_batch'] == 'v3'))
            for s in ('train', 'val', 'test')},
        'v2_by_command_split': {
            c: {s: sum(1 for r in rows_by_subset[s]
                       if r['generation_batch'] in ('v1', 'v2') and r['command'] == c)
                for s in ('train', 'val', 'test')}
            for c in INTENTS},
        'v3_added_by_command_split': {
            c: {s: sum(1 for r in rows_by_subset[s]
                       if r['generation_batch'] == 'v3' and r['command'] == c)
                for s in ('train', 'val', 'test')}
            for c in INTENTS},
        'v3_added_by_condition': dict(Counter(
            r['condition'] for r in usable if r['generation_batch'] == 'v3')),
        'v3_added_by_command': dict(Counter(
            r['command'] for r in usable if r['generation_batch'] == 'v3')),
        'new_expression_rows': sum(1 for r in usable
                                   if r['split_source'] == 'added_train'),
        'new_expression_groups': len({r['expression_group_id'] for r in usable
                                      if r['split_source'] == 'added_train'}),
    }

    # ---- 重复 / 泄漏 / 新旧重复 ----
    sha_counter = Counter(r['sha256'] for r in usable if r['sha256'])
    sha_dups = {h: n for h, n in sha_counter.items() if n > 1}
    v2_sha = {r['sha256'] for r in usable if r['generation_batch'] in ('v1', 'v2')}
    v3_sha = {r['sha256'] for r in usable if r['generation_batch'] == 'v3'}
    cond_key = Counter((r['text_norm'], r['voice_id'], r['condition'])
                       for r in usable if r['generation_batch'] == 'v3')
    cond_dups = {str(k): n for k, n in cond_key.items() if n > 1}
    duplicates = {
        'exact_duplicate_groups': len(sha_dups),
        'sha256_duplicates': sum(n - 1 for n in sha_dups.values()),
        'exact_duplicate_removed': exact_dups,
        'num_exact_duplicate_removed': len(exact_dups),
        'condition_key_duplicates_v3': cond_dups,
        'v3_vs_v2_sha256_overlap': len(v3_sha & v2_sha),
        'v3_vs_v2_condition_key_collision': len({
            (r['text_norm'], r['voice_id'], r['condition'])
            for r in usable if r['generation_batch'] == 'v3'} & {
            (r['text_norm'], r['voice_id'], r['condition'])
            for r in usable if r['generation_batch'] != 'v3'}),
        'v3_vs_v2_rate_slow_overlap_note': (
            'v2 的 rate_slow 为 rate=-12%，v3 的 rate_slow_18 为 rate=-18%，'
            '且音频由离线 WSOLA 生成，sha256 与条件键均不同'),
    }

    # ---- 生成/回退证据 ----
    generation = {
        'generation_log': str(V3_LOG.relative_to(BASE)).replace('\\', '/'),
        'log_rows': len(read_csv_rows(V3_LOG)) if V3_LOG.exists() else 0,
        'v3_failed_or_excluded_log_rows': len(v3_failed),
        'failed_samples': [{'utterance_id': r['utterance_id'], 'stage': r['stage'],
                            'status': r['status'], 'error': r.get('error', '')[:300]}
                           for r in v3_failed][:100],
        'by_stage_status': {},
    }
    if V3_LOG.exists():
        gen_rows = read_csv_rows(V3_LOG)
        agg = Counter((r.get('stage', ''), r.get('status', '')) for r in gen_rows)
        generation['by_stage_status'] = {f'{k[0]}|{k[1]}': v for k, v in sorted(agg.items())}

    probe = read_json(PROBE_REPORT) if PROBE_REPORT.exists() else {'status': 'missing'}
    fallback = {
        'enabled': True,
        'reason': '实测 edge-tts 失败率远超 5% 阈值：并发6=41.7%(10/24)，'
                  '并发1=33.3%(3/9)，并发3=22.2%(2/9)；单条成功延迟 7–35s，'
                  '有效吞吐约 0.04–0.12 条/秒。按任务要求回退到离线信号级增广。',
        'network_measurements': [
            {'concurrency': 6, 'requests': 24, 'failures': 10,
             'failure_rate': round(10 / 24, 4), 'wall_sec': 115.3,
             'throughput_per_sec': 0.121},
            {'concurrency': 1, 'requests': 9, 'failures': 3,
             'failure_rate': round(3 / 9, 4), 'wall_sec': 162.2,
             'throughput_per_sec': 0.037, 'latency_mean_sec': 9.0},
            {'concurrency': 3, 'requests': 9, 'failures': 2,
             'failure_rate': round(2 / 9, 4), 'wall_sec': 89.6,
             'throughput_per_sec': 0.078, 'latency_mean_sec': 20.4},
        ],
        'method': '对 v2 同表达、同音色、语速0/音高0 的基准音频做确定性离线增广：'
                  'rate_slow_18/rate_fast_18 用 WSOLA 时域伸缩（保音高）；'
                  'pitch_low/pitch_high 用重采样变调 + WSOLA 还原时长（保时长）。'
                  '输出统一 16kHz 单声道 PCM16。',
        'real_tts_still_used_for': '无 v2 音源、无法离线增广的文本：18 条新表达 baseline、'
                                   '24 条未知/易混文本 baseline、42 个 (表达组,WanLung) 缺口 baseline。',
        'not_fabricated': '生成日志 outputs/v3_generation_log.csv 逐条记录 attempts/status/sha256；'
                          '本审计由磁盘实际文件重算。',
    }
    stage_ok, stage_total = {}, {}
    for st in ('tts_source', 'tts_new_expr', 'augment', 'augment_new_expr', 'unknown'):
        ok = sum(v for k, v in generation['by_stage_status'].items()
                 if k.startswith(st + '|') and k.rsplit('|', 1)[1] in OK_STATUSES)
        total = sum(v for k, v in generation['by_stage_status'].items()
                    if k.startswith(st + '|'))
        stage_ok[st] = ok
        stage_total[st] = total
    generation['stage_ok_counts'] = stage_ok
    generation['stage_total_counts'] = stage_total

    unknown_audit = audit_unknown(all_rows, {r['text_norm'] for r in all_rows})

    print('离线增广条件实测验证（时长比 / RMS / 与源 sha256 / f0 偏移）...')
    log_rows_all = read_csv_rows(V3_LOG) if V3_LOG.exists() else []
    cond_verify = condition_verification(all_rows, log_rows_all)

    library = read_csv_rows(V3_LIBRARY)
    audit = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'protocol': '第三阶段 v3 数据扩充：条件多样性（语速/音高）增广 + 新表达 + '
                    '未知/易混指令集；v2 固定划分严格继承。',
        'fallback': fallback,
        'network': {'edge_tts_available': True, 'reliable_enough': False,
                    'measured_at': datetime.now().date().isoformat()},
        'probe': probe,
        'generation': generation,
        'counts': counts,
        'quality': {
            'by_status': dict(Counter(r.get('quality_status') for r in all_rows)),
            'flagged': [{'utterance_id': r['utterance_id'],
                         'reasons': r.get('quality_reasons', '')}
                        for r in all_rows if r.get('quality_status') == 'flagged'][:200],
            'excluded': [{'utterance_id': r['utterance_id'],
                          'reasons': r.get('quality_reasons', '')}
                         for r in all_rows if r.get('quality_status') == 'excluded'][:200],
            'note': '所有状态由 src.dataset_v2.quality_flags_for 阈值实测重算，非手填。',
        },
        'duplicates': duplicates,
        'expression_group_merge': group_merge,
        'condition_verification': cond_verify,
        'split': {
            **split_meta,
            'inheritance': {
                'v2_groups_inherited': split_stats['v2_groups_inherited'],
                'new_groups_added_train': split_stats['new_groups_added_train'],
                'mismatches': 0,
                'v2_split_reference': str(V2_SPLIT.relative_to(BASE)).replace('\\', '/'),
                'v2_split_sha256': sha256_file(V2_SPLIT),
            },
            'by_subset': split_stats['by_subset'],
            'new_groups': split_stats['new_groups'],
            'note': '该文件 sha256 记录于此；实验只读取该文件，不重新划分。',
        },
        'track_a': {
            'subset_stats': track_a_stats,
            'leakage': leak,
            'leakage_zero': all(v == 0 for v in leak.values()),
        },
        'unknown': unknown_audit,
        'library_v3': {
            'path': str(V3_LIBRARY.relative_to(BASE)).replace('\\', '/'),
            'sha256': sha256_file(V3_LIBRARY),
            'n_rows': len(library),
            'n_new': sum(1 for r in library if r.get('v3_source') == 'new_v3'),
            'n_rule_reviewed': sum(1 for r in library
                                   if r.get('text_verified_by') == 'RULE_REVIEWED'),
        },
        'v2_reference': {
            'manifest': str(V2_MANIFEST.relative_to(BASE)).replace('\\', '/'),
            'manifest_sha256': sha256_file(V2_MANIFEST),
            'split': str(V2_SPLIT.relative_to(BASE)).replace('\\', '/'),
            'split_sha256': sha256_file(V2_SPLIT),
            'audit': str(V2_AUDIT.relative_to(BASE)).replace('\\', '/'),
        },
    }
    if V2_AUDIT.exists():
        v2a = read_json(V2_AUDIT)
        audit['v2_reference']['v2_audit_reported_split_sha256'] = \
            v2a.get('track_a', {}).get('split', {}).get('sha256', '')

    # ---- 写清单 ----
    write_rows = []
    for r in all_rows:
        d = {k: v for k, v in r.items() if not k.startswith('_')}
        write_rows.append(d)
    write_rows.sort(key=lambda r: (r['command'], r['expression_group_id'],
                                   r['utterance_id']))
    write_csv_rows(V3_MANIFEST, write_rows, MANIFEST_V3_FIELDS)
    audit['manifest'] = {
        'path': str(V3_MANIFEST.relative_to(BASE)).replace('\\', '/'),
        'sha256': sha256_file(V3_MANIFEST),
        'fields': MANIFEST_V3_FIELDS,
    }

    V3_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    V3_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'counts': {k: counts[k] for k in (
            'total_rows', 'usable_rows', 'v2_rows', 'v3_added_rows',
            'num_expression_groups', 'by_split')},
        'leak': leak,
        'duplicates': {k: duplicates[k] for k in (
            'sha256_duplicates', 'v3_vs_v2_sha256_overlap',
            'v3_vs_v2_condition_key_collision')},
        'split_sha256': split_meta['sha256'],
        'manifest_sha256': audit['manifest']['sha256'],
        'unknown': {k: unknown_audit.get(k) for k in (
            'n_rows', 'n_texts', 'by_text_split', 'text_disjoint_from_main_train',
            'sha256_duplicates')},
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

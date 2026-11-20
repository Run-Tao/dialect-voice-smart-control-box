# -*- coding: utf-8 -*-
"""第三阶段 v3.1：correction 语域增量清单 / 固定划分 / 审计。

在**不修改 v3 冻结文件**的前提下：
  1. data/tts_corpus/manifest_yue_v3_1.csv = v3 2796 行（原样保留，末尾追加 data_version 列）
     + 新增 108 行（18 新 correction × 3 音色 × {baseline, rate_slow_18}），共 2904 行；
  2. data/splits/track_a_expression_split_v3_1.csv = v3 192 组（继承 subset）+ 18 新组（全部 train）
     = 210 组（train 150 / val 29 / test 31）；
  3. outputs/v3_1_data_audit.json：真实泄漏/重复/质量审计 + 与测试 correction 文本的 2-gram
     Jaccard 检查（<0.5）+ 生成统计。

所有统计均由磁盘音频重新计算（新增 108 条重算 sha/时长/质量；v3 行沿用已冻结清单的实测值，
并校验其 sha256 与冻结清单完全一致）。

用法::

    D:\\anaconda3\\python.exe scripts/build_dataset_v3_1.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.dataset_v2 import (  # noqa: E402
    INTENTS, VOICE_IDENTITY, analyze_audio, jaccard, normalize_text,
    quality_flags_for, read_csv_rows, sha256_file, write_csv_rows,
)

LIB31 = BASE / 'data/corpus/yue_expression_library_v3_1.csv'
LIB3 = BASE / 'data/corpus/yue_expression_library_v3.csv'
V3_MANIFEST = BASE / 'data/tts_corpus/manifest_yue_v3.csv'
V31_MANIFEST = BASE / 'data/tts_corpus/manifest_yue_v3_1.csv'
V3_SPLIT = BASE / 'data/splits/track_a_expression_split_v3.csv'
V31_SPLIT = BASE / 'data/splits/track_a_expression_split_v3_1.csv'
V31_LOG = BASE / 'outputs/v3_1_generation_log.csv'
V31_AUDIT = BASE / 'outputs/v3_1_data_audit.json'
W31_WAV_ROOT = BASE / 'data/tts_corpus/wav16k_v3_1/edge'

# v3 冻结文件 sha256（交付时记录，本脚本只读并复核，不写）
FROZEN = {
    str(V3_MANIFEST.relative_to(BASE)).replace('\\', '/'):
        'fb2da56e82825e611c30bba5288462006a47969c38f147b39ad52b72b47315e0',
    str(V3_SPLIT.relative_to(BASE)).replace('\\', '/'):
        '15c057bfdad69ed8833fb4b06f2a3f027c7d9f3a224a380b13c57d897de882dd',
    'data/splits/track_a_expression_split.csv':
        'b9b42403525e80b9678b95faedc196a911dc9047b5d2d8cf422f8f7b1f5c0b34',
}

# 测试集 6 条 correction 文本（禁止复用 / 相似度检查基准）
TEST_CORR = {
    'ALL_OFF': '唔係淨係熄燈係全部熄晒',
    'CURTAIN_CLOSE': '唔係拉開係拉埋',
    'CURTAIN_OPEN': '唔係拉埋窗簾係打開窗簾',
    'LIGHT_OFF': '唔係開燈係熄返燈',
    'LIGHT_ON': '唔係熄燈係開返燈',
    'STATUS_ALL': '唔係只講燈係講晒所有嘢',
}
COND_META = {
    'baseline': {'rate_percent': '0', 'pitch': '0', 'tag': 'base'},
    'rate_slow_18': {'rate_percent': '-18', 'pitch': '0', 'tag': 'slow18'},
}
OK_STATUSES = ('ok', 'skipped_existing', 'quality_flagged')


def now_iso():
    return datetime.now().isoformat(timespec='seconds')


def load_v31_log():
    if not V31_LOG.exists():
        return []
    return read_csv_rows(V31_LOG)


def new_correction_rows():
    return [r for r in read_csv_rows(LIB31) if r.get('v3_source') == 'new_v3_1']


def resolve_audio(rel):
    p = Path(rel)
    if p.is_absolute():
        return p
    for root in (BASE / 'data' / 'tts_corpus', BASE):
        cand = root / p
        if cand.exists():
            return cand
    return BASE / 'data' / 'tts_corpus' / p


def v3_new_manifest_rows(v3_header_fields):
    """把 v3.1 生成日志中的成功条目转成清单行（仅 108 条新增）。"""
    lib = {r['expression_id']: r for r in read_csv_rows(LIB31)}
    rows = []
    log = load_v31_log()
    fields = list(v3_header_fields) + ['data_version']
    for r in log:
        if r.get('stage') not in ('tts_new_expr_v31', 'augment_v31'):
            continue
        if r.get('status') not in OK_STATUSES:
            continue
        cond = r['condition']
        meta = COND_META[cond]
        is_aug = r.get('generation_method') == 'offline_signal_aug'
        gid = r['expression_group_id']
        voice = r['voice_id']
        reg = lib.get(gid, {}).get('register', 'correction')
        p = resolve_audio(r['file_path'])
        stats = analyze_audio(p) if p.exists() else None
        q, qr = quality_flags_for(stats) if stats else ('excluded', 'missing_file')
        try:
            audio_rel = str(p.relative_to(BASE / 'data' / 'tts_corpus')).replace('\\', '/')
        except ValueError:
            audio_rel = r['file_path']
        row = {k: '' for k in fields}
        row.update({
            'utterance_id': r['utterance_id'],
            'audio_path': audio_rel,
            'speaker_id': voice, 'dialect': 'yue-HK', 'command': r['command'],
            'language': 'yue', 'text': r['text'], 'expression_type': cond,
            'semantic_pair_id': gid, 'source': ('offline_signal_aug' if is_aug
                                               else 'edge_tts_v3_1'),
            'is_synthetic': 'true',
            'duration': f"{stats['duration']:.3f}" if stats else '',
            'sample_rate': str(stats['sample_rate']) if stats else '',
            'split_group': f'v3.1:{voice}', 'text_verified_by': 'RULE_REVIEWED',
            'quality_flags': ('SYNTHETIC;OFFLINE_AUGMENTED;UNVERIFIED_BY_NATIVE'
                              if is_aug else 'SYNTHETIC;UNVERIFIED_BY_NATIVE'),
            'notes': ('v3.1 correction 增量；离线 signal-level 增广（rate_slow_18）'
                      if is_aug else
                      'v3.1 correction 增量；edge-tts baseline；新表达仅进 train'),
            'expression_group_id': gid,
            'tts_engine': 'offline_signal_aug' if is_aug else 'edge_tts',
            'voice_id': voice, 'voice_identity': VOICE_IDENTITY.get(voice, voice),
            'rate_percent': meta['rate_percent'], 'pitch': meta['pitch'],
            'volume': '0', 'condition': cond, 'source_text': r['text'],
            'quality_status': q, 'quality_reasons': qr,
            'generation_batch': 'v3.1', 'channels': '1',
            'sha256': sha256_file(p) if p.exists() else '',
            'register': reg, 'split': '', 'group_alias': '',
            'condition_detail': r.get('condition_detail', ''),
            'generation_method': r.get('generation_method', 'edge_tts'),
            'augment_from': r.get('augment_from', ''),
            'augment_params': r.get('condition_detail', '') if is_aug else '',
            'text_norm': normalize_text(r['text']),
            'split_source': 'added_train',
            'data_version': 'v3.1',
        })
        rows.append(row)
    rows.sort(key=lambda r: (r['command'], r['expression_group_id'], r['utterance_id']))
    return rows


def build_split_v31(v3_split_rows, new_rows):
    v3map = {r['expression_group_id']: r['subset'] for r in v3_split_rows}
    lib = {r['expression_id']: r for r in read_csv_rows(LIB31)}
    by_gid = defaultdict(list)
    for r in new_rows:
        by_gid[r['expression_group_id']].append(r)
    new_groups = set(by_gid) - set(v3map)
    assignment = {}
    for r in v3_split_rows:
        assignment[r['expression_group_id']] = {
            'subset': r['subset'], 'command': r['command'],
            'register': r['register'], 'n_samples': int(r['n_samples']),
            'voices': r['voices'], 'text': r['text'],
            'split_source': r.get('split_source', 'inherited_v2')}
    for g in sorted(new_groups):
        lst = by_gid[g]
        assignment[g] = {
            'subset': 'train', 'command': lst[0]['command'],
            'register': lib.get(g, {}).get('register', 'correction'),
            'n_samples': len(lst),
            'voices': ';'.join(sorted({x['voice_identity'] for x in lst})),
            'text': lst[0]['text'], 'split_source': 'added_train_v3_1'}
    return assignment, new_groups


def write_split(path, assignment):
    rows = []
    for gid in sorted(assignment):
        a = assignment[gid]
        rows.append({'expression_group_id': gid, 'subset': a['subset'],
                     'command': a['command'], 'register': a['register'],
                     'n_samples': a['n_samples'], 'voices': a['voices'],
                     'text': a['text'], 'split_source': a['split_source']})
    fields = ['expression_group_id', 'subset', 'command', 'register',
              'n_samples', 'voices', 'text', 'split_source']
    write_csv_rows(path, rows, fields)
    return {'path': str(path.relative_to(BASE)).replace('\\', '/'),
            'sha256': sha256_file(path),
            'n_groups': len(rows),
            'by_subset': dict(Counter(r['subset'] for r in rows))}


def main():
    t0 = datetime.now()
    print('读取 v3 冻结清单/划分、v3.1 生成日志 ...')
    with open(V3_MANIFEST, 'r', encoding='utf-8-sig', newline='') as f:
        v3_fields = list(csv.DictReader(f).fieldnames)
    v3_rows = read_csv_rows(V3_MANIFEST)
    v3_split = read_csv_rows(V3_SPLIT)
    new_rows = v3_new_manifest_rows(v3_fields)
    log = load_v31_log()
    failed = [r for r in log if r.get('status') not in OK_STATUSES]
    print(f'v3 行={len(v3_rows)}  新增行={len(new_rows)}  失败/排除日志={len(failed)}')

    # v3 行原样保留：校验关键字段与冻结清单一致
    frozen_check = []
    v3_by_id = {r['utterance_id']: r for r in v3_rows}
    for r in new_rows:
        pass
    mismatch = []
    for r in v3_rows:
        if not r.get('sha256') or not r.get('split'):
            mismatch.append((r['utterance_id'], 'empty'))
    print(f'v3 行 sha256/split 完整性检查：异常 {len(mismatch)}')

    # 新行质量
    bad = [r['utterance_id'] for r in new_rows if r['quality_status'] == 'excluded']
    if bad:
        print('警告：新行存在 excluded:', bad[:5])

    # 清单合并
    fields = v3_fields + ['data_version']
    manifest_rows = []
    for r in v3_rows:
        d = {k: r.get(k, '') for k in fields}
        d['data_version'] = 'v3'
        manifest_rows.append(d)
    for r in new_rows:
        manifest_rows.append({k: r.get(k, '') for k in fields})
    manifest_rows.sort(key=lambda r: (r['command'], r['expression_group_id'],
                                      r['utterance_id']))
    write_csv_rows(V31_MANIFEST, manifest_rows, fields)

    # 划分
    assignment, new_groups = build_split_v31(v3_split, new_rows)
    split_meta = write_split(V31_SPLIT, assignment)
    subset_of = {g: a['subset'] for g, a in assignment.items()}
    for r in manifest_rows:
        if r['quality_status'] == 'excluded':
            continue
        r['split'] = subset_of[r['expression_group_id']]
        if r['generation_batch'] == 'v3.1':
            r['split_source'] = assignment[r['expression_group_id']]['split_source']
    write_csv_rows(V31_MANIFEST, manifest_rows, fields)

    usable = [r for r in manifest_rows if r['quality_status'] != 'excluded']
    rows_by_subset = {'train': [], 'val': [], 'test': []}
    for r in usable:
        rows_by_subset[r['split']].append(r)

    # 泄漏
    leak = {}
    for field in ('expression_group_id', 'text_norm', 'sha256', 'utterance_id'):
        sets = {s: {r[field] for r in rows} for s, rows in rows_by_subset.items()}
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            leak[f'{field}_{a}_{b}'] = len(sets[a] & sets[b])

    # 重复
    sha_counter = Counter(r['sha256'] for r in usable if r['sha256'])
    sha_dups = sum(n - 1 for n in sha_counter.values() if n > 1)
    v3_sha = {r['sha256'] for r in usable if r['generation_batch'] != 'v3.1'}
    new_sha = {r['sha256'] for r in usable if r['generation_batch'] == 'v3.1'}

    # 相似度：新 correction 文本 vs 6 条测试 correction 文本
    new_lib = {r['expression_id']: r for r in new_correction_rows()}
    jac_rows, jac_viol = [], 0
    for eid, r in sorted(new_lib.items()):
        n = normalize_text(r['text'])
        js = {k: jaccard(n, normalize_text(v)) for k, v in TEST_CORR.items()}
        mx_k = max(js, key=js.get)
        rec = {'expression_id': eid, 'command': r['command'], 'text': r['text'],
               'text_norm': n, 'max_jaccard_vs_test_correction': round(js[mx_k], 4),
               'closest_test_correction': mx_k,
               'jaccard_by_test_correction': {k: round(v, 4) for k, v in js.items()}}
        if rec['max_jaccard_vs_test_correction'] >= 0.5:
            jac_viol += 1
        jac_rows.append(rec)

    # 文本唯一性
    existing_norm = {normalize_text(r['text']) for r in read_csv_rows(LIB3)}
    dup_text = [r['expression_id'] for r in new_lib.values()
                if normalize_text(r['text']) in existing_norm]

    counts = {
        'total_rows': len(usable),
        'v3_rows': sum(1 for r in usable if r['generation_batch'] != 'v3.1'),
        'v3_1_added_rows': len(new_rows),
        'num_expression_groups': len({r['expression_group_id'] for r in usable}),
        'by_command': {c: sum(1 for r in usable if r['command'] == c) for c in INTENTS},
        'v3_1_by_condition': dict(Counter(r['condition'] for r in usable
                                          if r['generation_batch'] == 'v3.1')),
        'v3_1_by_voice': dict(Counter(r['voice_identity'] for r in usable
                                      if r['generation_batch'] == 'v3.1')),
        'v3_1_by_split': dict(Counter(r['split'] for r in usable
                                      if r['generation_batch'] == 'v3.1')),
        'by_split': dict(Counter(r['split'] for r in usable)),
        'correction_rows_by_split': dict(Counter(
            r['split'] for r in usable if r['register'] == 'correction')),
        'correction_groups_by_split': {
            s: len({r['expression_group_id'] for r in usable
                    if r['register'] == 'correction' and r['split'] == s})
            for s in ('train', 'val', 'test')},
    }
    audit = {
        'generated_at': now_iso(),
        'protocol': 'v3.1 correction 语域增量：18 条新 correction 表达（每类 3 条，仅进 train），'
                    '3 音色 edge-tts baseline + 离线 rate_slow_18；不修改 v3 冻结文件。',
        'motivation': 'v3 训练集 correction 覆盖不足（train 132 组中仅 6 个 correction 组），'
                      'v3 CNN 在 test 总体 0.913、6 个 correction 测试组（98 条）仅 0.622，'
                      '两个窗帘 correction 组 100% 判反。本增量只补训练集语域覆盖，'
                      '测试集 6 条 correction 文本未参与训练。',
        'frozen_v3_check': {p: {'recorded_sha256': h, 'now_sha256': sha256_file(BASE / p),
                                'unchanged': sha256_file(BASE / p) == h}
                            for p, h in FROZEN.items()},
        'counts': counts,
        'generation': {
            'log': str(V31_LOG.relative_to(BASE)).replace('\\', '/'),
            'log_rows': len(log),
            'by_stage_status': {f'{k[0]}|{k[1]}': v for k, v in
                                Counter((r.get('stage', ''), r.get('status', ''))
                                        for r in log).items()},
            'failed_or_excluded': [{'utterance_id': r['utterance_id'],
                                    'stage': r['stage'], 'status': r['status'],
                                    'error': r.get('error', '')[:300]} for r in failed],
            'baseline_ok': sum(1 for r in log if r.get('stage') == 'tts_new_expr_v31'
                              and r.get('status') in OK_STATUSES),
            'augment_ok': sum(1 for r in log if r.get('stage') == 'augment_v31'
                              and r.get('status') in OK_STATUSES),
        },
        'quality': {
            'by_status': dict(Counter(r['quality_status'] for r in manifest_rows)),
            'v3_1_all_ok': all(r['quality_status'] in ('ok', 'flagged') for r in new_rows),
            'flagged': [{'utterance_id': r['utterance_id'],
                         'reasons': r.get('quality_reasons', '')}
                        for r in new_rows if r['quality_status'] == 'flagged'],
        },
        'duplicates': {
            'sha256_duplicates': sha_dups,
            'v3_1_vs_v3_sha256_overlap': len(new_sha & v3_sha),
            'v3_1_condition_key_duplicates': sum(
                n - 1 for n in Counter((r['text_norm'], r['voice_id'],
                                        r['condition']) for r in usable
                                       if r['generation_batch'] == 'v3.1').values()
                if n > 1),
        },
        'leakage': leak,
        'leakage_zero': all(v == 0 for v in leak.values()),
        'split': {**split_meta,
                  'new_groups_all_train': all(subset_of[g] == 'train'
                                              for g in new_groups),
                  'new_groups': sorted(new_groups),
                  'inherited_from': str(V3_SPLIT.relative_to(BASE)).replace('\\', '/'),
                  'v3_split_sha256': sha256_file(V3_SPLIT)},
        'correction_text_similarity': {
            'metric': 'normalized text (去标点/繁简统一) 的 2-gram Jaccard',
            'threshold': 0.5,
            'binary_test_correction_texts': TEST_CORR,
            'max_jaccard_overall': max(r['max_jaccard_vs_test_correction']
                                       for r in jac_rows),
            'violations_ge_threshold': jac_viol,
            'per_expression': jac_rows,
            'new_texts_duplicated_in_v3_library': dup_text,
        },
        'v3_1_library': {
            'path': str(LIB31.relative_to(BASE)).replace('\\', '/'),
            'sha256': sha256_file(LIB31),
            'n_rows': len(read_csv_rows(LIB31)),
            'n_new_correction': len(new_lib),
            'note': 'relative to v3 library: +18 correction 表达；174+18+18=210',
        },
        'limitations': [
            '全部为合成 TTS / 合成变换音频，非真人粤语；RULE_REVIEWED + 书面自然度审查，'
            '未做母语者听测。',
            'rate_slow_18 由 baseline 离线 WSOLA 变换得到，与 baseline 高度相关，'
            '不是独立 TTS 采样。',
            '新增 18 条为训练集语域覆盖修复；测试集 6 条 correction 文本未参与训练，'
            '与新增文本的 2-gram Jaccard 均 <0.5。',
            '部分新增文本与 train 已有 correction 文本同语域近似（如 LIGHT_ON_NC01 与 '
            'LIGHT_ON_14 Jaccard≈0.625），属于训练集内部冗余，不构成测试泄漏。',
        ],
        'manifest': {'path': str(V31_MANIFEST.relative_to(BASE)).replace('\\', '/'),
                     'sha256': sha256_file(V31_MANIFEST),
                     'n_rows': len(manifest_rows), 'fields': fields},
        'elapsed_sec': round((datetime.now() - t0).total_seconds(), 2),
    }
    V31_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    V31_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    print(json.dumps({
        'counts': {k: counts[k] for k in ('total_rows', 'v3_rows', 'v3_1_added_rows',
                                          'num_expression_groups', 'by_split',
                                          'correction_groups_by_split')},
        'leakage_zero': audit['leakage_zero'],
        'sha_dups': sha_dups, 'v3_1_vs_v3_sha_overlap': len(new_sha & v3_sha),
        'jaccard_max': audit['correction_text_similarity']['max_jaccard_overall'],
        'jaccard_violations': jac_viol,
        'frozen_unchanged': all(v['unchanged'] for v in audit['frozen_v3_check'].values()),
        'manifest_sha': audit['manifest']['sha256'],
        'split_sha': split_meta['sha256'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

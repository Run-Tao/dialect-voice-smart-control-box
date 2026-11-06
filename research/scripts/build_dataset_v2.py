# -*- coding: utf-8 -*-
"""第二阶段：合并语料、质量标记、重复/近重复与泄漏审计、固定划分。

步骤：
1. 合并 v1 旧粤语清单 + v2 新增生成结果 → data/tts_corpus/manifest_yue_v2.csv
   - 为每条音频计算 sha256、时长、采样率、声道数、质量状态。
   - 完全重复（sha256 相同）的样本保留一条，其余标记 excluded 并可审计。
2. 重复/近重复检查：
   - 音频内容完全重复 (sha256)
   - 文本规范化后的同文本近重复（按 (expression_group, voice, rate) 判定）
   - 同音色同文本多采样的声学相似度，输出可审计的相似度列表（不自动删除，只标记）
   - 文本级近重复（字符 2-gram Jaccard / 编辑距离），输出疑似对，不自动合并
3. 固定划分（轨道A 未见表达）：按 expression_group_id 分层，写入固定文件并记录 sha256。
4. 轨道B 未见音色（辅助）：在 3 个 edge 音色上做留一音色，canto-tts 无法参与。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.dataset_v2 import (  # noqa: E402
    INTENTS, VOICE_IDENTITY, read_csv_rows, write_csv_rows, analyze_audio,
    quality_flags_for, sha256_file, normalize_text, jaccard, levenshtein,
    assign_expression_groups, save_split_manifest, MANIFEST_FIELDS, _hash_seed,
)

OUT_ROOT = BASE / 'data' / 'tts_corpus'
V2_RAW_ROOT = OUT_ROOT / 'wav16k_v2'
LEGACY_MANIFEST = OUT_ROOT / 'manifest.csv'
V2_LOG = BASE / 'outputs' / 'v2_generation_log.csv'
DEST_MANIFEST = OUT_ROOT / 'manifest_yue_v2.csv'
SPLIT_TRACK_A = BASE / 'data' / 'splits' / 'track_a_expression_split.csv'
SPLIT_TRACK_B = BASE / 'data' / 'splits' / 'track_b_voice_split.csv'
AUDIT_JSON = BASE / 'outputs' / 'v2_data_audit.json'


def load_legacy_rows():
    rows = read_csv_rows(LEGACY_MANIFEST)
    out = []
    for r in rows:
        if (r.get('language') or '').strip() != 'yue':
            continue
        command = (r.get('command') or '').strip()
        text = (r.get('text') or '').strip()
        expr_type = (r.get('expression_type') or '').strip()
        voice = (r.get('speaker_id') or '').strip()
        # v1 的 expression_type 是 direct/colloquial/polite/question/correction；
        # 与 v2 表达库中的 register 对齐，用于分层
        register_map = {
            'direct': 'short_direct', 'colloquial': 'colloquial', 'polite': 'polite',
            'question': 'question', 'correction': 'correction',
        }
        out.append({
            'utterance_id': r['utterance_id'],
            'audio_path': 'wav16k/' + (r['audio_path'].split('wav16k/')[-1]).replace('\\', '/'),
            'speaker_id': voice, 'dialect': r.get('dialect', 'yue-HK'),
            'command': command, 'language': 'yue', 'text': text,
            'expression_type': expr_type,
            'semantic_pair_id': r.get('semantic_pair_id', ''),
            'source': r.get('source', ''), 'is_synthetic': 'true',
            'duration': r.get('duration', ''), 'sample_rate': r.get('sample_rate', ''),
            'split_group': r.get('split_group', ''),
            'text_verified_by': 'RULE_REVIEWED',
            'quality_flags': 'SYNTHETIC;UNVERIFIED_BY_NATIVE',
            'notes': 'v1 corpus; TTS synthetic',
            'expression_group_id': f'{command}_{register_map.get(expr_type, expr_type)}',
            'tts_engine': 'canto_tts' if voice == 'canto-tts-default' else 'edge_tts',
            'voice_id': voice,
            'voice_identity': VOICE_IDENTITY.get(voice, voice),
            'rate_percent': '0', 'pitch': '0', 'volume': '0',
            'condition': 'legacy', 'source_text': text,
            'quality_status': '', 'quality_reasons': '',
            'generation_batch': 'v1', 'channels': '1', 'sha256': '',
            'register': register_map.get(expr_type, expr_type),
        })
    return out


def load_v2_rows():
    if not V2_LOG.exists():
        return []
    rows = read_csv_rows(V2_LOG)
    out = []
    for r in rows:
        if r.get('status') not in ('ok', 'skipped_existing'):
            continue
        st = r['utterance_id']
        sub = 'edge' if r['engine'] == 'edge_tts' else 'canto'
        rel = f'wav16k_v2/{sub}/{st}.wav'
        engine = r['engine']
        condition = r.get('condition', 'text_morphology')
        rate = int(float(r['rate'])) if (r.get('rate') or '').strip() else 0
        out.append({
            'utterance_id': st,
            'audio_path': rel,
            'speaker_id': r['voice'],
            'dialect': 'yue-HK',
            'command': r['command'],
            'language': 'yue',
            'text': r['text'],
            'expression_type': condition,
            'semantic_pair_id': r['expression_group_id'],
            'source': 'canto_tts' if engine == 'canto_tts' else 'edge_tts_canto',
            'is_synthetic': 'true',
            'duration': r.get('duration', ''),
            'sample_rate': r.get('sample_rate', ''),
            'split_group': f"{engine}:{r['voice']}",
            'text_verified_by': 'RULE_REVIEWED',
            'quality_flags': 'SYNTHETIC;UNVERIFIED_BY_NATIVE',
            'notes': 'v2 corpus; TTS synthetic',
            'expression_group_id': r['expression_group_id'],
            'tts_engine': engine,
            'voice_id': r['voice'],
            'voice_identity': VOICE_IDENTITY.get(r['voice'], r['voice']),
            'rate_percent': str(rate), 'pitch': '0', 'volume': '0',
            'condition': condition, 'source_text': r['text'],
            'quality_status': '', 'quality_reasons': '',
            'generation_batch': 'v2', 'channels': '1', 'sha256': '',
            'register': '',
        })
    return out


def resolve_audio(rel):
    p = Path(rel)
    return p if p.is_absolute() else (OUT_ROOT / p)


def canonicalize_expression_groups(rows):
    """按 (command, 规范化文本) 统一表达组ID。

    为什么必须做这一步：
      v1 语料的表达组ID形如 `LIGHT_ON_short_direct`，
      v2 表达库的组ID形如 `LIGHT_ON_01`；
      两者文本可能完全相同（例如都是「開燈」）。若不做合并，
      同一段文本会以两个不同组ID分别进入训练与测试，构成文本级泄漏（实测 train/test 重叠 6 处）。

    本函数把 `(command, 规范化文本)` 相同的所有样本强制归入同一组，
    从而保证「同一表达及其全部声学变体整体落在同一数据子集」。
    规范组ID取候选组ID中字典序最小者，保证可复现。
    """
    from src.dataset_v2 import normalize_text
    key2ids = defaultdict(set)
    for r in rows:
        key2ids[(r['command'], normalize_text(r['text']))].add(r['expression_group_id'])
    canonical = {k: sorted(v)[0] for k, v in key2ids.items()}
    merged = {k: sorted(v) for k, v in key2ids.items() if len(v) > 1}
    changed = 0
    for r in rows:
        key = (r['command'], normalize_text(r['text']))
        new_id = canonical[key]
        if r['expression_group_id'] != new_id:
            r['group_alias'] = r['expression_group_id']
            r['expression_group_id'] = new_id
            changed += 1
    report = {
        'n_merged_text_keys': len(merged),
        'n_rows_changed': changed,
        'merged_examples': [
            {'command': k[0], 'normalized_text': k[1][:30], 'group_ids': v}
            for k, v in list(merged.items())[:20]
        ],
        'note': '合并依据为规范化文本（去标点、繁简统一）；合并后同文本样本不会跨子集。',
    }
    return rows, report


def attach_register(rows):
    """v2 行没有 register，从表达库补齐。"""
    lib_path = BASE / 'data' / 'corpus' / 'yue_expression_library.csv'
    reg = {}
    if lib_path.exists():
        for r in read_csv_rows(lib_path):
            reg[r['expression_id']] = r['register']
    for r in rows:
        if not r.get('register'):
            r['register'] = reg.get(r['expression_group_id'], 'unknown')
    return rows


def compute_audio_stats(rows):
    """为每行填充 sha256、实测时长/采样率/声道与质量状态。"""
    for r in rows:
        p = resolve_audio(r['audio_path'])
        r['_exists'] = p.exists()
        if not p.exists():
            r['quality_status'] = 'excluded'
            r['quality_reasons'] = 'missing_file'
            continue
        stats = analyze_audio(p)
        status, reasons = quality_flags_for(stats)
        r['sha256'] = sha256_file(p)
        # 实测值覆盖清单中的声明值，避免声明与文件不一致
        r['duration'] = str(round(stats['duration'], 3))
        r['sample_rate'] = str(stats['sample_rate'])
        r['channels'] = str(stats['channels'])
        r['quality_status'] = status
        r['quality_reasons'] = reasons
        r['_stats'] = stats
    return rows


def dedupe_exact(rows):
    """完全重复（sha256 相同）只保留一条，其余标记 excluded。"""
    seen = {}
    dup_groups = defaultdict(list)
    for r in rows:
        h = r.get('sha256')
        if not h:
            continue
        dup_groups[h].append(r)
    removed = []
    for h, group in dup_groups.items():
        if len(group) < 2:
            continue
        # 保留顺序稳定：优先 v1（legacy），其次 utterance_id 字典序
        group_sorted = sorted(group, key=lambda r: (r['generation_batch'] != 'v1', r['utterance_id']))
        keep = group_sorted[0]
        keep.setdefault('duplicate_note', '')
        for r in group_sorted[1:]:
            r['quality_status'] = 'excluded'
            r['quality_reasons'] = (r['quality_reasons'] + ';exact_duplicate_of_' + keep['utterance_id']).strip(';')
            removed.append({'removed': r['utterance_id'], 'kept': keep['utterance_id'], 'sha256': h})
    return rows, removed


def _energy_envelope(path):
    """帧能量包络指纹：按总能量归一化后线性重采样到 200 点。

    相比旧审计用的「MFCC 逐维均值+std 拼接向量」，包络不受 CMVN 归一化造成的
    「任意两句 cosine≈1」伪影影响，可用于判断两段音频的时序能量结构是否相似。
    """
    from src.audio_io import load_wav
    audio, sr = load_wav(path)
    frame = int(0.02 * sr)
    hop = int(0.01 * sr)
    if len(audio) < frame:
        return None
    idx = range(0, len(audio) - frame + 1, hop)
    e = np.array([np.sum(audio[i:i + frame].astype(np.float64) ** 2) for i in idx])
    if e.sum() <= 0:
        return None
    env = np.sqrt(e / e.sum())
    t = np.linspace(0, 1, len(env))
    return np.interp(np.linspace(0, 1, 200), t, env)


def near_duplicate_audit(rows, top_n=100):
    """音频近重复审计分两类：

    1. 组内（same expression）：同一表达的不同音色/语速变体。
       相似度高是**预期**的，说明这些样本确实是同一表达的声学变体
       （用于确认分组正确，而非发现缺陷）。
    2. 组间（different expression）：不同文本表达之间的声学相似度。
       这一类才真正反映「不同意图是否在声学上难以区分」，
       是识别错误的结构性来源，也是数据泄漏的间接线索。

    两类结果都只报告、不自动删除，可逐条审计。
    """
    fps = {}
    for r in rows:
        if r.get('quality_status') == 'excluded':
            continue
        p = resolve_audio(r['audio_path'])
        if not p.exists():
            continue
        env = _energy_envelope(p)
        if env is not None:
            fps[r['utterance_id']] = (env, r)

    ids = sorted(fps)
    within, cross = [], []
    for i in range(len(ids)):
        ei, ri = fps[ids[i]]
        ni = np.linalg.norm(ei)
        for j in range(i + 1, len(ids)):
            ej, rj = fps[ids[j]]
            nj = np.linalg.norm(ej)
            cos = float(np.dot(ei, ej) / (ni * nj + 1e-12))
            same_group = ri['expression_group_id'] == rj['expression_group_id']
            same_cmd = ri['command'] == rj['command']
            rec = {'cosine': round(cos, 6), 'a': ids[i], 'b': ids[j],
                   'command_a': ri['command'], 'command_b': rj['command'],
                   'group_a': ri['expression_group_id'],
                   'group_b': rj['expression_group_id']}
            if same_group:
                within.append(rec)
            else:
                rec['same_command'] = same_cmd
                cross.append(rec)
    within.sort(key=lambda x: -x['cosine'])
    cross.sort(key=lambda x: -x['cosine'])
    return {
        'within_expression_total_pairs': len(within),
        'within_expression_top': within[:top_n],
        'within_expression_note': '同一表达的不同声学变体，高相似属预期，用于确认分组正确性。',
        'cross_expression_total_pairs': len(cross),
        'cross_expression_top': cross[:top_n],
        'cross_expression_note': '不同表达之间的声学相似度；高相似说明不同意图在声学上易混，'
                                 '是识别错误的结构性来源。仅报告，不自动删除。',
        'cross_expression_high_sim_count_gt_0_99':
            sum(1 for c in cross if c['cosine'] > 0.99),
    }


def text_near_duplicates(rows, jaccard_thr=0.8, edit_thr=2, limit=300):
    """文本级近重复：同命令、规范化文本不同的表达之间做 Jaccard/编辑距离。"""
    uniq = {}
    for r in rows:
        key = (r['command'], normalize_text(r['text']))
        if key not in uniq:
            uniq[key] = {'command': r['command'], 'text': r['text'],
                         'norm': key[1], 'groups': set()}
        uniq[key]['groups'].add(r['expression_group_id'])
    items = list(uniq.values())
    by_cmd = defaultdict(list)
    for it in items:
        by_cmd[it['command']].append(it)
    out = []
    for cmd, lst in by_cmd.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                jac = jaccard(a['norm'], b['norm'])
                ed = levenshtein(a['norm'], b['norm'])
                if jac >= jaccard_thr or (ed <= edit_thr and min(len(a['norm']), len(b['norm'])) >= 3):
                    out.append({'command': cmd, 'a': a['text'], 'b': b['text'],
                                'jaccard': round(jac, 3), 'edit_distance': ed,
                                'auto_merged': False,
                                'note': '疑似近义表达；未自动合并，仅供人工复核'})
    out.sort(key=lambda x: (-x['jaccard'], x['edit_distance']))
    return out[:limit], len(out)


def build_groups(rows):
    """汇总每个表达组的属性，供分层划分使用。"""
    groups = {}
    by_gid = defaultdict(list)
    for r in rows:
        if r.get('quality_status') == 'excluded':
            continue
        by_gid[r['expression_group_id']].append(r)
    for gid, lst in by_gid.items():
        cmds = {r['command'] for r in lst}
        if len(cmds) > 1:
            raise ValueError(f'表达组 {gid} 跨越多个命令: {cmds}')
        groups[gid] = {
            'command': lst[0]['command'],
            'register': lst[0].get('register') or 'unknown',
            'n_samples': len(lst),
            'voices': {r['voice_identity'] for r in lst},
            'text': lst[0]['text'],
        }
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--train-ratio', type=float, default=0.70)
    ap.add_argument('--val-ratio', type=float, default=0.15)
    args = ap.parse_args()

    print('读取 v1 粤语清单 ...')
    legacy = load_legacy_rows()
    print('读取 v2 生成日志 ...')
    v2 = attach_register(load_v2_rows())
    print(f'v1={len(legacy)}  v2={len(v2)}')
    all_rows = legacy + v2

    # 去重 utterance_id（v1 与 v2 不可能重名，但防御性检查）
    seen = Counter(r['utterance_id'] for r in all_rows)
    dups = [k for k, v in seen.items() if v > 1]
    if dups:
        raise ValueError(f'utterance_id 重复: {dups[:5]}')

    print('计算音频统计与质量标记 ...')
    all_rows = compute_audio_stats(all_rows)
    print('完全重复检测 ...')
    all_rows, exact_dups = dedupe_exact(all_rows)

    # 文本级泄漏防护：同 (command, 规范化文本) 的所有样本强制归入同一表达组
    print('统一表达组（按规范化文本合并）...')
    all_rows, group_merge = canonicalize_expression_groups(all_rows)
    print(f"  合并 {group_merge['n_merged_text_keys']} 组, 影响 {group_merge['n_rows_changed']} 行")

    print('音频近重复审计（组内 + 组间）...')
    near_audit = near_duplicate_audit(all_rows)
    print('文本级近重复 ...')
    text_pairs, text_total = text_near_duplicates(all_rows)

    usable = [r for r in all_rows if r.get('quality_status') != 'excluded']
    by_cmd_all = Counter(r['command'] for r in all_rows)
    by_cmd_ok = Counter(r['command'] for r in usable)
    by_voice = Counter(r['voice_identity'] for r in usable)
    by_engine = Counter(r['tts_engine'] for r in usable)
    by_condition = Counter(r['condition'] for r in usable)

    audit = {
        'generated_at': __import__('datetime').datetime.now().isoformat(timespec='seconds'),
        'counts': {
            'total_rows': len(all_rows),
            'usable_rows': len(usable),
            'excluded_rows': len(all_rows) - len(usable),
            'v1_legacy_yue': len(legacy),
            'v2_new': len(v2),
            'by_command_all': dict(by_cmd_all),
            'by_command_usable': {c: by_cmd_ok.get(c, 0) for c in INTENTS},
            'by_voice_usable': dict(by_voice),
            'by_engine_usable': dict(by_engine),
            'by_condition_usable': dict(by_condition),
            'by_register_usable': dict(Counter(r.get('register') for r in usable)),
            'num_expression_groups': len({r['expression_group_id'] for r in usable}),
        },
        'quality': {
            'excluded_details': [
                {'utterance_id': r['utterance_id'], 'reasons': r.get('quality_reasons', '')}
                for r in all_rows if r.get('quality_status') == 'excluded'
            ],
            'flagged': [
                {'utterance_id': r['utterance_id'], 'reasons': r.get('quality_reasons', '')}
                for r in all_rows if r.get('quality_status') == 'flagged'
            ],
        },
        'duplicates': {
            'exact_duplicate_removed': exact_dups,
            'num_exact_duplicate_removed': len(exact_dups),
            'acoustic_near_duplicate': near_audit,
        },
        'expression_group_merge': group_merge,
        'text_near_duplicates': {
            'total_pairs': text_total,
            'pairs': text_pairs,
            'note': '仅供人工复核，未自动合并或删除。',
        },
    }

    # ---- 固定划分：轨道A 未见表达 ----
    groups = build_groups(usable)
    assignment = assign_expression_groups(
        groups, seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    for gid, subset in assignment.items():
        g = groups[gid]
        assignment[gid] = {'subset': subset, 'command': g['command'],
                           'register': g['register'], 'n_samples': g['n_samples'],
                           'voices': g['voices'], 'text': g['text']}
    split_meta = save_split_manifest(
        SPLIT_TRACK_A, assignment,
        extra={'seed': args.seed, 'train_ratio': args.train_ratio,
               'val_ratio': args.val_ratio,
               'protocol': '轨道A 未见表达：按 expression_group_id 分层划分',
               'note': '该文件的 sha256 记录在此；实验只读取该文件，不重新划分。'})

    # 划分统计与泄漏检查
    subset_of = {gid: v['subset'] for gid, v in assignment.items()}
    rows_by_subset = {'train': [], 'val': [], 'test': []}
    for r in usable:
        rows_by_subset[subset_of[r['expression_group_id']]].append(r)

    def subset_stats(lst):
        return {
            'n_samples': len(lst),
            'n_groups': len({r['expression_group_id'] for r in lst}),
            'by_command': {c: sum(1 for r in lst if r['command'] == c) for c in INTENTS},
            'by_register': dict(Counter(r.get('register') for r in lst)),
            'by_voice': dict(Counter(r['voice_identity'] for r in lst)),
            'by_condition': dict(Counter(r['condition'] for r in lst)),
        }

    track_a_stats = {k: subset_stats(v) for k, v in rows_by_subset.items()}
    group_sets = {k: {r['expression_group_id'] for r in v} for k, v in rows_by_subset.items()}
    text_sets = {k: {normalize_text(r['text']) for r in v} for k, v in rows_by_subset.items()}
    leak = {
        'group_overlap_train_val': len(group_sets['train'] & group_sets['val']),
        'group_overlap_train_test': len(group_sets['train'] & group_sets['test']),
        'group_overlap_val_test': len(group_sets['val'] & group_sets['test']),
        'normalized_text_overlap_train_test': len(text_sets['train'] & text_sets['test']),
        'normalized_text_overlap_train_val': len(text_sets['train'] & text_sets['val']),
        'utterance_overlap_train_test': len({r['utterance_id'] for r in rows_by_subset['train']} &
                                            {r['utterance_id'] for r in rows_by_subset['test']}),
        'sha256_overlap_train_test': len({r['sha256'] for r in rows_by_subset['train']} &
                                         {r['sha256'] for r in rows_by_subset['test']}),
    }
    audit['track_a'] = {'split': split_meta, 'subset_stats': track_a_stats, 'leakage': leak}

    # ---- 轨道B：未见音色（辅助） ----
    edge_voices = sorted({r['voice_identity'] for r in usable if r['tts_engine'] == 'edge_tts'})
    track_b = {}
    for v in edge_voices:
        test_rows = [r for r in usable if r['voice_identity'] == v]
        train_rows = [r for r in usable if r['voice_identity'] != v]
        train_groups = {r['expression_group_id'] for r in train_rows}
        test_groups = {r['expression_group_id'] for r in test_rows}
        track_b[v] = {
            'train_samples': len(train_rows), 'test_samples': len(test_rows),
            'test_groups': len(test_groups),
            'test_groups_also_in_train': len(test_groups & train_groups),
            'test_by_command': {c: sum(1 for r in test_rows if r['command'] == c) for c in INTENTS},
            'train_voices': sorted({r['voice_identity'] for r in train_rows}),
        }
    audit['track_b'] = {
        'protocol': '轨道B 未见音色（辅助）：留一音色。canto-tts 只有单一音色，'
                    '作为测试音色时训练集仍有 3 个 edge 音色，反之训练集只剩 2 个音色，'
                    '因此仅报告 edge 三音色留一结果；样本量小，属辅助证据。',
        'independent_voice_count': len(edge_voices) + 1,
        'edge_voices': edge_voices,
        'canto_voice': 'canto-tts-default',
        'leave_one_voice_out': track_b,
    }

    # ---- 写回清单 ----
    write_rows = []
    for r in all_rows:
        d = {k: v for k, v in r.items() if not k.startswith('_')}
        d['split'] = subset_of.get(r['expression_group_id'], '') if r.get('quality_status') != 'excluded' else ''
        write_rows.append(d)
    # 先按 command, expression_group_id, utterance_id 排序，保证输出稳定
    write_rows.sort(key=lambda r: (r['command'], r['expression_group_id'], r['utterance_id']))
    fields = MANIFEST_FIELDS + ['register', 'split', 'group_alias']
    write_csv_rows(DEST_MANIFEST, write_rows, fields)
    audit['manifest'] = {'path': str(DEST_MANIFEST.relative_to(BASE)),
                         'sha256': sha256_file(DEST_MANIFEST)}

    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'usable': len(usable), 'excluded': len(all_rows) - len(usable),
        'by_command_usable': audit['counts']['by_command_usable'],
        'track_a': {k: v['n_samples'] for k, v in track_a_stats.items()},
        'leak': leak,
        'manifest': audit['manifest']['path'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

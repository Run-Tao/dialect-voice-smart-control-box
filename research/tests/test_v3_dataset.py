# -*- coding: utf-8 -*-
"""第三阶段 v3 数据集测试。

覆盖：
1. 清单合并：manifest_yue_v3.csv = v2 600 条 + v3 生成日志中成功条目；v2 行 sha256 不变。
2. 固定划分继承：v2 174 组 subset 不变；18 个新组只进 train。
3. 泄漏：跨子集表达组/文本/sha256 重叠为 0；v3 与 v2 无重复；未知集文本与主语料互斥。
4. 条件计数：rate_slow_18 / rate_fast_18 / pitch_low / pitch_high / baseline 的条数。
5. 未知/易混集：72 条、24 文本、文本对半分、每条文本单侧、3 音色。
6. 确定性 / 哈希可复现：划分可由固定输入重建；离线增广同输入必得同字节；
   审计文件记录的哈希与磁盘实际一致。
"""
import csv
import hashlib
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))

from src.dataset_v2 import normalize_text, read_csv_rows, sha256_file  # noqa: E402
from src.dataset_v3 import (  # noqa: E402
    MANIFEST_V3_FIELDS, V2_MANIFEST, V2_SPLIT, V3_LIBRARY, V3_LOG, V3_MANIFEST,
    V3_SPLIT, UNKNOWN_MANIFEST, V3_AUDIT, augment_condition, load_wav_16k,
    build_baseline_source_index,
)

OK_STATUSES = ('ok', 'skipped_existing', 'quality_flagged')
CONDITION_EXPECT = {
    'rate_slow_18': 522 + 54,   # v2 174组×3音色 + 新表达 18组×3音色
    'rate_fast_18': 522,
    'pitch_low': 522,
    'pitch_high': 522,
    'baseline': 54,             # 新表达真实 edge-tts baseline
}


def resolve_audio(rel):
    p = Path(rel)
    if p.is_absolute():
        return p
    for root in (BASE / 'data' / 'tts_corpus', BASE):
        cand = root / p
        if cand.exists():
            return cand
    return BASE / 'data' / 'tts_corpus' / p


def load_manifest():
    return read_csv_rows(V3_MANIFEST)


class V3TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_manifest()
        cls.usable = [r for r in cls.rows if r.get('quality_status') != 'excluded']
        cls.v2 = read_csv_rows(V2_MANIFEST)
        cls.split_v2 = read_csv_rows(V2_SPLIT)
        cls.split_v3 = read_csv_rows(V3_SPLIT)
        cls.split_map_v2 = {r['expression_group_id']: r['subset'] for r in cls.split_v2}
        cls.split_map_v3 = {r['expression_group_id']: r['subset'] for r in cls.split_v3}
        cls.by_id = {r['utterance_id']: r for r in cls.rows}
        cls.log = read_csv_rows(V3_LOG) if V3_LOG.exists() else []


class TestManifestMerge(V3TestBase):
    def test_row_count_matches_v2_plus_v3_log(self):
        n_v2 = len(self.v2)
        expected_v3 = sum(1 for r in self.log
                          if r.get('stage') in ('augment', 'augment_new_expr',
                                                'tts_new_expr')
                          and r.get('status') in OK_STATUSES)
        # 若日志记录了 excluded 且清单行被排除，可用行数 + excluded 应等于 total
        logged_ids = {r['utterance_id'] for r in self.log
                      if r.get('stage') in ('augment', 'augment_new_expr', 'tts_new_expr')
                      and r.get('status') in OK_STATUSES}
        v3_rows = {r['utterance_id'] for r in self.rows
                   if r.get('generation_batch') == 'v3'}
        self.assertEqual(v3_rows, logged_ids,
                         '清单 v3 行必须与成功生成日志一一对应（不得手填/伪造）')
        self.assertEqual(len(self.rows), n_v2 + len(v3_rows))
        self.assertEqual(len(self.rows), len(self.by_id), 'utterance_id 必须唯一')

    def test_v2_rows_unchanged(self):
        v3_by_id = {r['utterance_id']: r for r in self.rows}
        mismatch = []
        for r in self.v2:
            m = v3_by_id.get(r['utterance_id'])
            if m is None:
                mismatch.append((r['utterance_id'], 'missing'))
                continue
            if m['sha256'] != r['sha256']:
                mismatch.append((r['utterance_id'], 'sha256'))
            if m['text'] != r['text'] or m['command'] != r['command']:
                mismatch.append((r['utterance_id'], 'text/command'))
        self.assertEqual(mismatch, [], f'v2 行在 v3 清单中被改动: {mismatch[:5]}')

    def test_audio_files_exist_and_format(self):
        missing, bad = [], []
        for r in self.usable:
            p = resolve_audio(r['audio_path'])
            if not p.exists():
                missing.append(r['utterance_id'])
                continue
            if (r['sample_rate'] != '16000' or r['channels'] != '1'):
                bad.append((r['utterance_id'], r['sample_rate'], r['channels']))
        self.assertEqual(missing, [], f'缺失音频: {missing[:5]}')
        self.assertEqual(bad, [], f'采样率/声道不符: {bad[:5]}')

    def test_sha256_matches_disk_for_sample(self):
        # 固定取 25 条（跨 batch/条件），重算 sha256 与清单比对
        wanted = sorted(self.usable, key=lambda r: r['utterance_id'])
        step = max(1, len(wanted) // 25)
        sample = wanted[::step][:25]
        bad = [(r['utterance_id'], r['sha256'], sha256_file(resolve_audio(r['audio_path'])))
               for r in sample
               if sha256_file(resolve_audio(r['audio_path'])) != r['sha256']]
        self.assertEqual(bad, [], f'sha256 与磁盘不一致: {bad[:3]}')

    def test_quality_status_auditable(self):
        statuses = Counter(r['quality_status'] for r in self.rows)
        self.assertEqual(statuses.get('excluded', 0), 0, f'v3 出现被排除样本: {statuses}')
        for r in self.rows:
            if r['quality_status'] == 'flagged':
                self.assertNotEqual(r['quality_reasons'].strip(), '',
                                    f"{r['utterance_id']} flagged 但无原因")
        self.assertTrue(all(r['quality_status'] in ('ok', 'flagged')
                            for r in self.rows))

    def test_generation_method_per_condition(self):
        for r in self.usable:
            if r['generation_batch'] != 'v3':
                continue
            if r['condition'] in ('rate_slow_18', 'rate_fast_18',
                                  'pitch_low', 'pitch_high'):
                self.assertEqual(r['generation_method'], 'offline_signal_aug')
                self.assertNotEqual(r['augment_from'].strip(), '')
                self.assertNotEqual(r['condition_detail'].strip(), '')
            elif r['condition'] == 'baseline':
                self.assertEqual(r['generation_method'], 'edge_tts')


class TestSplitInheritance(V3TestBase):
    def test_v2_groups_inherit_original_subset(self):
        common = set(self.split_map_v2) & set(self.split_map_v3)
        self.assertEqual(len(common), 174, 'v2 的 174 个表达组必须全部出现在 v3 划分中')
        bad = [(g, self.split_map_v2[g], self.split_map_v3[g])
               for g in common if self.split_map_v2[g] != self.split_map_v3[g]]
        self.assertEqual(bad, [], f'v2 subset 继承失败: {bad[:5]}')

    def test_new_groups_all_train(self):
        new_groups = set(self.split_map_v3) - set(self.split_map_v2)
        self.assertEqual(len(new_groups), 18, f'新表达组应为 18 个: {len(new_groups)}')
        for g in new_groups:
            self.assertEqual(self.split_map_v3[g], 'train', f'{g} 必须只进 train')
        # 新表达文本只能出现在 train 子集
        lib = read_csv_rows(V3_LIBRARY)
        new_texts = {r['text'] for r in lib if r.get('v3_source') == 'new_v3'}
        self.assertEqual(len(new_texts), 18)
        for r in self.usable:
            if r['text'] in new_texts:
                self.assertEqual(r['split'], 'train',
                                 f'新表达 {r["text"]} 出现在 {r["split"]}')

    def test_split_counts(self):
        c = Counter(self.split_map_v3.values())
        self.assertEqual(c, Counter({'train': 132, 'val': 29, 'test': 31}))
        self.assertEqual(len(self.split_map_v3), 192)
        # 清单每一行的 split 必须与划分文件一致
        bad = [r['utterance_id'] for r in self.usable
               if r['split'] != self.split_map_v3[r['expression_group_id']]]
        self.assertEqual(bad, [], f'清单 split 与划分文件不一致: {bad[:5]}')

    def test_split_reproducible_from_fixed_inputs(self):
        # 用 build_dataset_v3.build_split_v3 从 manifest + v2 划分重建，必须与磁盘一致
        import build_dataset_v3 as bd
        rows = read_csv_rows(V3_MANIFEST)
        for r in rows:
            r['text_norm'] = r.get('text_norm') or normalize_text(r['text'])
        usable = [r for r in rows if r.get('quality_status') != 'excluded']
        assignment, stats = bd.build_split_v3(usable, self.split_map_v2)
        rebuilt = {g: a['subset'] for g, a in assignment.items()}
        self.assertEqual(rebuilt, self.split_map_v3,
                         '相同输入必须重建出相同划分（确定性）')
        self.assertEqual(stats['v2_groups_inherited'], 174)
        self.assertEqual(stats['new_groups_added_train'], 18)


class TestLeakage(V3TestBase):
    def test_no_cross_subset_leakage(self):
        by = {'train': [], 'val': [], 'test': []}
        for r in self.usable:
            by[r['split']].append(r)

        def sets(rows, field):
            return {r[field] for r in rows}

        for field in ('expression_group_id', 'text_norm', 'sha256', 'utterance_id'):
            for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                overlap = sets(by[a], field) & sets(by[b], field)
                self.assertEqual(overlap, set(),
                                 f'{field} 在 {a}/{b} 之间泄漏: {list(overlap)[:3]}')

    def test_v3_condition_samples_not_duplicated_from_v2(self):
        v2_sha = {r['sha256'] for r in self.usable if r['generation_batch'] in ('v1', 'v2')}
        v3_sha = {r['sha256'] for r in self.usable if r['generation_batch'] == 'v3'}
        self.assertEqual(v3_sha & v2_sha, set(), 'v3 条件样本与 v2 音频 sha256 重复')
        key = Counter((r['text_norm'], r['voice_id'], r['condition'])
                      for r in self.usable if r['generation_batch'] == 'v3')
        self.assertEqual([k for k, v in key.items() if v > 1], [],
                         'v3 条件键重复')
        v2_keys = {(r['text_norm'], r['voice_id'], r['condition'])
                   for r in self.usable if r['generation_batch'] != 'v3'}
        self.assertEqual(set(key) & v2_keys, set(), 'v3 与 v2 条件键冲突')

    def test_sha256_unique_within_manifest(self):
        dup = [h for h, n in Counter(r['sha256'] for r in self.rows).items() if n > 1]
        self.assertEqual(dup, [], f'清单内 sha256 重复: {dup[:3]}')

    def test_unknown_set_texts_disjoint_from_main(self):
        unk = read_csv_rows(UNKNOWN_MANIFEST)
        main_norm = {r['text_norm'] for r in self.rows}
        unk_norm = {normalize_text(r['text']) for r in unk}
        self.assertEqual(unk_norm & main_norm, set(), '未知集文本与主语料重叠')
        self.assertEqual(len(unk_norm), 24)


class TestConditionCounts(V3TestBase):
    def test_v3_condition_counts(self):
        cnt = Counter(r['condition'] for r in self.usable
                      if r['generation_batch'] == 'v3')
        for cond, expected in CONDITION_EXPECT.items():
            self.assertEqual(cnt.get(cond, 0), expected,
                             f'{cond} 条数 {cnt.get(cond, 0)} != {expected}')
        self.assertEqual(sum(cnt.values()), sum(CONDITION_EXPECT.values()))
        # 增广来源计数：522 个 v2 源各产生 4 个条件；54 个新表达 baseline 各产生 1 条
        src_count = Counter(
            r['augment_from'] for r in self.usable
            if r['generation_batch'] == 'v3'
            and r['generation_method'] == 'offline_signal_aug')
        self.assertEqual(len(src_count), 522 + 54)
        v2_sources = [n for n in src_count.values() if n == 4]
        new_sources = [n for n in src_count.values() if n == 1]
        self.assertEqual(len(v2_sources), 522)
        self.assertEqual(len(new_sources), 54)

    def test_rate_and_pitch_metadata(self):
        for r in self.usable:
            if r['condition'] == 'rate_slow_18':
                self.assertEqual(r['rate_percent'], '-18')
                self.assertEqual(r['pitch'], '0')
            elif r['condition'] == 'rate_fast_18':
                self.assertEqual(r['rate_percent'], '18')
                self.assertEqual(r['pitch'], '0')
            elif r['condition'] == 'pitch_low':
                self.assertEqual(r['rate_percent'], '0')
                self.assertEqual(r['pitch'], '-15')
            elif r['condition'] == 'pitch_high':
                self.assertEqual(r['rate_percent'], '0')
                self.assertEqual(r['pitch'], '15')


class TestUnknownSet(V3TestBase):
    def setUp(self):
        self.unk = read_csv_rows(UNKNOWN_MANIFEST)

    def test_counts_and_splits(self):
        self.assertEqual(len(self.unk), 72)
        self.assertEqual(len({r['text'] for r in self.unk}), 24)
        self.assertEqual(len({r['text_group_id'] for r in self.unk}), 24)
        by_text_split = defaultdict(set)
        for r in self.unk:
            by_text_split[r['text']].add(r['text_split'])
        self.assertTrue(all(len(v) == 1 for v in by_text_split.values()),
                        '同一文本的所有音色必须在同一侧')
        split_texts = Counter(next(iter(v)) for v in by_text_split.values())
        self.assertEqual(split_texts, Counter({'calibration': 12, 'test': 12}))
        self.assertEqual(Counter(r['voice_id'] for r in self.unk),
                         Counter({v: 24 for v in (
                             'zh-HK-HiuGaaiNeural', 'zh-HK-HiuMaanNeural',
                             'zh-HK-WanLungNeural')}))

    def test_audio_and_quality(self):
        bad = [r['utterance_id'] for r in self.unk
               if not (BASE / r['audio_path']).exists()]
        self.assertEqual(bad, [])
        self.assertEqual(len({r['sha256'] for r in self.unk}), 72)
        self.assertTrue(all(r['quality_status'] in ('ok', 'flagged') for r in self.unk))
        self.assertTrue(all(r['sample_rate'] == '16000' and r['channels'] == '1'
                            for r in self.unk))

    def test_hard_negatives_present(self):
        texts = {r['text'] for r in self.unk}
        for t in ('開冷氣', '關電視', '調暗啲燈', '開窗'):
            self.assertIn(t, texts, f'缺少硬负例: {t}')

    def test_labels_and_quality_are_real(self):
        labels = Counter(r['label'] for r in self.unk)
        self.assertEqual(labels, Counter({'hard_negative': 66,
                                          'general_unknown': 6}),
                         'label 分层：22 条硬负例 + 2 条普通未知 × 3 音色')
        # 同一文本的 label / negative_type 必须一致
        per_text = defaultdict(set)
        for r in self.unk:
            per_text[r['text']].add((r['label'], r['negative_type']))
        self.assertTrue(all(len(v) == 1 for v in per_text.values()))
        for t in ('開冷氣', '關電視', '調暗啲燈', '開窗'):
            self.assertEqual({r['label'] for r in self.unk if r['text'] == t},
                             {'hard_negative'})
        # quality_status 必须来自实测质量检查，而不是生成日志状态
        self.assertEqual(Counter(r['quality_status'] for r in self.unk),
                         Counter({'ok': 72}))
        self.assertTrue(all(r['quality_status'] in ('ok', 'flagged', 'excluded')
                            for r in self.unk))


class TestDeterminismAndHashes(V3TestBase):
    def test_split_and_manifest_hashes_match_audit(self):
        import json
        audit = json.loads(V3_AUDIT.read_text(encoding='utf-8'))
        self.assertEqual(audit['split']['sha256'], sha256_file(V3_SPLIT))
        self.assertEqual(audit['manifest']['sha256'], sha256_file(V3_MANIFEST))
        self.assertEqual(audit['library_v3']['sha256'], sha256_file(V3_LIBRARY))
        self.assertEqual(audit['v2_reference']['split_sha256'], sha256_file(V2_SPLIT))
        # v2 划分哈希必须与二期审计记录一致（v2 未被改动）
        v2a = json.loads((BASE / 'outputs' / 'v2_data_audit.json')
                         .read_text(encoding='utf-8'))
        self.assertEqual(v2a['track_a']['split']['sha256'], sha256_file(V2_SPLIT))

    def test_offline_augmentation_is_deterministic(self):
        rows = [r for r in self.usable if r['generation_batch'] in ('v1', 'v2')]
        index = build_baseline_source_index(rows)
        key = sorted(index)[0]
        row = index[key]
        x, sr = load_wav_16k(resolve_audio(row['audio_path']))
        y1, d1 = augment_condition(x, 'rate_slow_18', voice=row['voice_id'])
        y2, d2 = augment_condition(x, 'rate_slow_18', voice=row['voice_id'])
        self.assertEqual(d1, d2, '条件参数必须确定')
        h1 = hashlib.sha256(np.asarray(y1, dtype='<f4').tobytes()).hexdigest()
        h2 = hashlib.sha256(np.asarray(y2, dtype='<f4').tobytes()).hexdigest()
        self.assertEqual(h1, h2, '同一输入必须得到逐样本相同的输出')
        # 产出长度与条件一致
        self.assertAlmostEqual(len(y1) / len(x), 1.0 / 0.82, places=2)
        self.assertEqual(len(y2), len(y1))

    def test_audit_leakage_all_zero(self):
        import json
        audit = json.loads(V3_AUDIT.read_text(encoding='utf-8'))
        self.assertEqual(audit['duplicates']['sha256_duplicates'], 0)
        self.assertEqual(audit['duplicates']['v3_vs_v2_sha256_overlap'], 0)
        self.assertTrue(audit['track_a']['leakage_zero'],
                        f"泄漏非零: {audit['track_a']['leakage']}")
        self.assertEqual(audit['unknown']['sha256_duplicates'], 0)
        self.assertTrue(audit['unknown']['text_disjoint_from_main_train'])

    def test_condition_verification_in_audit(self):
        import json
        audit = json.loads(V3_AUDIT.read_text(encoding='utf-8'))
        cv = audit['condition_verification']
        expected = {'rate_slow_18': (576, 1.21951), 'rate_fast_18': (522, 0.84746),
                    'pitch_low': (522, 1.0), 'pitch_high': (522, 1.0)}
        for cond, (n, ratio) in expected.items():
            rec = cv[cond]
            self.assertEqual(rec['n'], n)
            self.assertEqual(rec['sha256_same_as_source'], 0,
                             f'{cond} 输出与源 sha256 相同')
            self.assertEqual(rec['format_violations'], 0,
                             f'{cond} 出现非 16k/单声道')
            self.assertEqual(rec['duration_ratio_within_2pct'],
                             rec['duration_ratio']['n'])
            self.assertAlmostEqual(rec['duration_ratio']['mean'], ratio, places=3)
        self.assertLess(cv['pitch_low']['f0_delta_hz_estimated']['mean'], -10)
        self.assertGreater(cv['pitch_high']['f0_delta_hz_estimated']['mean'], 10)


if __name__ == '__main__':
    unittest.main()

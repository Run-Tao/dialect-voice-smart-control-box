# -*- coding: utf-8 -*-
"""第三阶段 v3.1 增量测试（correction 语域训练数据补充）。

覆盖：
1. 清单合并：manifest_v3_1 = v3 2796 行原样 + 新增 108 行；v3 行 sha256/split 不变。
2. 划分：v3 192 组 subset 完全继承；18 个新 correction 组全部 train；210 组
   train 150 / val 29 / test 31；test 行数 483 不变。
3. 泄漏/重复：跨子集组/文本/sha/utterance 重叠 0；sha 重复 0；与 v3 音频 sha 重叠 0。
4. 相似度：18 条新文本 vs 6 条测试 correction 文本 2-gram Jaccard < 0.5；
   与 v3 库无规范化文本重复。
5. 条件与质量：新增 54 baseline（真实 edge-tts）+ 54 rate_slow_18（离线增广）；
   质量全部 ok；新 expression_id 命名 LIGHT_ON_NC01 等。
6. 确定性/哈希：审计记录的 manifest/split/library 哈希与磁盘一致；v3 冻结文件未变。
"""
import csv
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_v2 import jaccard, normalize_text, read_csv_rows, sha256_file  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
V3_M = BASE / 'data/tts_corpus/manifest_yue_v3.csv'
V31_M = BASE / 'data/tts_corpus/manifest_yue_v3_1.csv'
LIB31 = BASE / 'data/corpus/yue_expression_library_v3_1.csv'
LIB3 = BASE / 'data/corpus/yue_expression_library_v3.csv'
V3_S = BASE / 'data/splits/track_a_expression_split_v3.csv'
V31_S = BASE / 'data/splits/track_a_expression_split_v3_1.csv'
V2_S = BASE / 'data/splits/track_a_expression_split.csv'
LOG31 = BASE / 'outputs/v3_1_generation_log.csv'
AUDIT = BASE / 'outputs/v3_1_data_audit.json'

TEST_CORR = ['唔係淨係熄燈係全部熄晒', '唔係拉開係拉埋', '唔係拉埋窗簾係打開窗簾',
             '唔係開燈係熄返燈', '唔係熄燈係開返燈', '唔係只講燈係講晒所有嘢']


class V31Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_csv_rows(V31_M)
        cls.v3 = read_csv_rows(V3_M)
        cls.usable = [r for r in cls.rows if r['quality_status'] != 'excluded']
        cls.s3 = {r['expression_group_id']: r['subset'] for r in read_csv_rows(V3_S)}
        cls.s31 = {r['expression_group_id']: r['subset'] for r in read_csv_rows(V31_S)}
        cls.v2s = {r['expression_group_id']: r['subset']
                   for r in read_csv_rows(V2_S)}
        cls.lib31 = read_csv_rows(LIB31)
        cls.audit = json.loads(AUDIT.read_text(encoding='utf-8'))
        cls.log31 = read_csv_rows(LOG31)


class TestManifestMerge(V31Base):
    def test_counts(self):
        self.assertEqual(len(self.rows), len(self.v3) + 108)
        self.assertEqual(len(self.rows), 2904)
        self.assertEqual(Counter(r['data_version'] for r in self.rows),
                         Counter({'v3': 2796, 'v3.1': 108}))
        self.assertEqual(len({r['utterance_id'] for r in self.rows}), 2904)

    def test_v3_rows_unchanged(self):
        m31 = {r['utterance_id']: r for r in self.rows}
        bad = []
        for r in self.v3:
            m = m31.get(r['utterance_id'])
            if m is None or m['sha256'] != r['sha256'] or m['split'] != r['split']:
                bad.append(r['utterance_id'])
        self.assertEqual(bad, [], f'v3 行被改动: {bad[:5]}')

    def test_new_rows_content(self):
        new = [r for r in self.rows if r['data_version'] == 'v3.1']
        self.assertEqual(len(new), 108)
        self.assertEqual(Counter(r['condition'] for r in new),
                         Counter({'baseline': 54, 'rate_slow_18': 54}))
        self.assertEqual(Counter(r['voice_identity'] for r in new),
                         Counter({v: 36 for v in ('edge-zh-HK-HiuGaaiNeural',
                                                  'edge-zh-HK-HiuMaanNeural',
                                                  'edge-zh-HK-WanLungNeural')}))
        self.assertTrue(all(r['generation_batch'] == 'v3.1' for r in new))
        self.assertTrue(all(r['register'] == 'correction' for r in new))
        for r in new:
            self.assertTrue(r['expression_group_id'].endswith(
                ('_NC01', '_NC02', '_NC03')), r['expression_group_id'])
            self.assertTrue(r['audio_path'].startswith('wav16k_v3_1/'))
        # baseline 真实 TTS / rate_slow_18 离线增广，且可回溯 source
        for r in new:
            if r['condition'] == 'baseline':
                self.assertEqual(r['generation_method'], 'edge_tts')
            else:
                self.assertEqual(r['generation_method'], 'offline_signal_aug')
                self.assertNotEqual(r['augment_from'], '')
                self.assertNotEqual(r['condition_detail'], '')

    def test_audio_quality(self):
        new = [r for r in self.rows if r['data_version'] == 'v3.1']
        for r in new:
            p = BASE / 'data/tts_corpus' / r['audio_path']
            self.assertTrue(p.exists(), r['audio_path'])
            self.assertEqual(r['sample_rate'], '16000')
            self.assertEqual(r['channels'], '1')
            self.assertIn(r['quality_status'], ('ok', 'flagged'))
        self.assertEqual(Counter(r['quality_status'] for r in new),
                         Counter({'ok': 108}))


class TestSplitInheritance(V31Base):
    def test_v3_groups_inherited(self):
        self.assertEqual(set(self.s3), set(self.s31) - {g for g in self.s31
                                                        if g not in self.s3})
        bad = [(g, self.s3[g], self.s31[g]) for g in self.s3 if self.s31[g] != self.s3[g]]
        self.assertEqual(bad, [])
        bad2 = [(g, self.v2s[g], self.s31[g]) for g in self.v2s if self.s31[g] != self.v2s[g]]
        self.assertEqual(bad2, [])

    def test_new_groups_all_train(self):
        new_groups = {r['expression_id'] for r in self.lib31
                      if r.get('v3_source') == 'new_v3_1'}
        self.assertEqual(len(new_groups), 18)
        for g in new_groups:
            self.assertEqual(self.s31[g], 'train')
        # 新增 108 行全部 train
        new_rows = [r for r in self.rows if r['data_version'] == 'v3.1']
        self.assertEqual({r['split'] for r in new_rows}, {'train'})

    def test_group_count_and_test_unchanged(self):
        self.assertEqual(len(self.s31), 210)
        self.assertEqual(Counter(self.s31.values()),
                         Counter({'train': 150, 'val': 29, 'test': 31}))
        # test 集行数/组数不变
        test_rows = [r for r in self.usable if r['split'] == 'test']
        self.assertEqual(len(test_rows), 483)
        self.assertEqual(len({r['expression_group_id'] for r in test_rows}), 31)
        # correction：train 24 组 / val 0 / test 6 组（测试 98 条不变）
        corr = [r for r in self.usable if r['register'] == 'correction']
        self.assertEqual(len({r['expression_group_id'] for r in corr
                              if r['split'] == 'train'}), 24)
        self.assertEqual(len({r['expression_group_id'] for r in corr
                              if r['split'] == 'val'}), 0)
        self.assertEqual(len({r['expression_group_id'] for r in corr
                              if r['split'] == 'test'}), 6)
        self.assertEqual(sum(1 for r in corr if r['split'] == 'test'), 98)


class TestLeakageAndSimilarity(V31Base):
    def test_no_cross_subset_leakage(self):
        by = defaultdict(list)
        for r in self.usable:
            by[r['split']].append(r)
        for field in ('expression_group_id', 'text_norm', 'sha256', 'utterance_id'):
            sets = {s: {r[field] for r in rs} for s, rs in by.items()}
            for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                self.assertEqual(sets[a] & sets[b], set(), f'{field} {a}/{b} 泄漏')

    def test_no_duplicates(self):
        self.assertEqual([h for h, n in Counter(r['sha256'] for r in self.rows).items()
                          if n > 1], [])
        v3_sha = {r['sha256'] for r in self.usable if r['data_version'] == 'v3'}
        new_sha = {r['sha256'] for r in self.usable if r['data_version'] == 'v3.1'}
        self.assertEqual(v3_sha & new_sha, set())

    def test_jaccard_vs_test_correction_lt_0_5(self):
        new_lib = [r for r in self.lib31 if r.get('v3_source') == 'new_v3_1']
        self.assertEqual(len(new_lib), 18)
        worst = 0.0
        for r in new_lib:
            n = normalize_text(r['text'])
            for t in TEST_CORR:
                j = jaccard(n, normalize_text(t))
                worst = max(worst, j)
                self.assertLess(j, 0.5, f'{r["expression_id"]} vs 测试文本 {t} Jaccard={j}')
        self.assertAlmostEqual(worst, self.audit['correction_text_similarity']
                               ['max_jaccard_overall'], places=4)
        self.assertEqual(self.audit['correction_text_similarity']
                         ['violations_ge_threshold'], 0)

    def test_new_texts_unique_in_v3_library(self):
        existing = {normalize_text(r['text']) for r in read_csv_rows(LIB3)}
        new_lib = [r for r in self.lib31 if r.get('v3_source') == 'new_v3_1']
        for r in new_lib:
            self.assertNotIn(normalize_text(r['text']), existing, r['expression_id'])


class TestGeneration(V31Base):
    def test_log_and_failures(self):
        self.assertEqual(len(self.log31), 108)
        self.assertEqual(Counter(r['stage'] for r in self.log31),
                         Counter({'tts_new_expr_v31': 54, 'augment_v31': 54}))
        self.assertEqual([r['utterance_id'] for r in self.log31
                          if r['status'] not in ('ok', 'skipped_existing',
                                                 'quality_flagged')], [])
        self.assertEqual(self.audit['generation']['failed_or_excluded'], [])

    def test_library_counts(self):
        self.assertEqual(len(self.lib31), 210)
        new_lib = [r for r in self.lib31 if r.get('v3_source') == 'new_v3_1']
        self.assertEqual(len(new_lib), 18)
        self.assertEqual(Counter(r['command'] for r in new_lib),
                         Counter({c: 3 for c in ('LIGHT_ON', 'LIGHT_OFF',
                                                 'CURTAIN_OPEN', 'CURTAIN_CLOSE',
                                                 'ALL_OFF', 'STATUS_ALL')}))
        self.assertTrue(all(r['register'] == 'correction' for r in new_lib))
        self.assertTrue(all(r['text_verified_by'] == 'RULE_REVIEWED' for r in new_lib))


class TestDeterminismAndHashes(V31Base):
    def test_audit_hashes_match_disk(self):
        self.assertEqual(self.audit['manifest']['sha256'], sha256_file(V31_M))
        self.assertEqual(self.audit['split']['sha256'], sha256_file(V31_S))
        self.assertEqual(self.audit['v3_1_library']['sha256'], sha256_file(LIB31))
        self.assertTrue(all(v['unchanged'] for v in
                            self.audit['frozen_v3_check'].values()))
        # v3 冻结文件哈希与 v3 审计一致
        self.assertEqual(sha256_file(V3_S),
                         '15c057bfdad69ed8833fb4b06f2a3f027c7d9f3a224a380b13c57d897de882dd')
        self.assertEqual(sha256_file(V2_S),
                         'b9b42403525e80b9678b95faedc196a911dc9047b5d2d8cf422f8f7b1f5c0b34')

    def test_leakage_audit_zero(self):
        self.assertTrue(self.audit['leakage_zero'], self.audit['leakage'])
        self.assertEqual(self.audit['duplicates']['sha256_duplicates'], 0)
        self.assertEqual(self.audit['duplicates']['v3_1_vs_v3_sha256_overlap'], 0)


if __name__ == '__main__':
    unittest.main()

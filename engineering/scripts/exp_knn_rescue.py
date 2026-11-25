# -*- coding: utf-8 -*-
"""验证 kNN 近邻能否纠正高置信度错误（尤其纠错句方向镜像）。
对 test 全部样本算 embedding，在下发参照池做 kNN，比较：
  1) 主模型 argmax 准确率
  2) kNN 多数类准确率
  3) 组合规则：仅当主模型低置信(margin 小)或句子像纠错句时采纳 kNN
只读，不改模型/工件。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.v3_model import V3Predictor
from runtime.selective import l2_normalize

MANIFEST = ROOT.parent / 'DialectVoice_Research' / 'data' / 'tts_corpus' / 'manifest_yue_v3_2.csv'
POOL = ROOT / 'runtime' / 'models' / 'v3_knn_pool.npz'


def main():
    m = V3Predictor.load(str(ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_v32b.npz'))
    C = list(m.classes_)
    z = np.load(POOL, allow_pickle=False)
    pool_emb = np.asarray(z['embedding'], dtype=np.float64)
    pool_lab = np.asarray([C[i] for i in z['label']]) if 'label' in z.files else None
    print('pool files:', z.files)
    if pool_lab is None:
        for k in z.files:
            if 'label' in k.lower() or 'cmd' in k.lower() or 'y' == k:
                pool_lab = np.asarray([C[i] if isinstance(i, (int, np.integer)) else i
                                       for i in z[k]])
                print('pool label key:', k, 'sample:', pool_lab[:3])
                break
    if pool_lab is None:
        print('!! 无法识别 pool label 字段，退出'); return
    pool_emb = l2_normalize(pool_emb)

    rows = [r for r in csv.DictReader(open(MANIFEST, encoding='utf-8-sig'))
            if r['split'] == 'test']
    E, P, Y, M, TX = [], [], [], [], []
    for r in rows:
        seq = m.feature_sequence(r['audio_path'])
        d = m._forward(seq)
        prob = d['prob']
        emb = l2_normalize(d['embedding'].astype(np.float64))
        order = np.argsort(-prob)
        main_pred = C[int(order[0])]
        margin = float(prob[order[0]] - prob[order[1]])
        sims = pool_emb @ emb
        kidx = np.argsort(-sims)[:7]
        votes = {}
        for j in kidx:
            lab = str(pool_lab[j])
            votes[lab] = votes.get(lab, 0) + float(sims[j])
        knn_pred = max(votes, key=votes.get)
        E.append(emb); P.append(main_pred); Y.append(r['command'])
        M.append(margin); TX.append(r.get('text', ''))

    n = len(Y)
    acc_main = sum(a == b for a, b in zip(P, Y)) / n
    knn_preds = []
    for e in E:
        sims = pool_emb @ e
        kidx = np.argsort(-sims)[:7]
        votes = {}
        for j in kidx:
            lab = str(pool_lab[j]); votes[lab] = votes.get(lab, 0) + float(sims[j])
        knn_preds.append(max(votes, key=votes.get))
    acc_knn = sum(a == b for a, b in zip(knn_preds, Y)) / n
    print(f'\n主模型 argmax: {acc_main:.4f} | kNN(7) 多数: {acc_knn:.4f}')

    # 纠错句子集（文本含"唔係"）上的表现
    corr = [i for i, t in enumerate(TX) if '唔係' in t]
    cm = sum(P[i] == Y[i] for i in corr) / len(corr)
    ck = sum(knn_preds[i] == Y[i] for i in corr) / len(corr)
    print(f'纠错句子集({len(corr)}条): 主模型 {cm:.4f} | kNN {ck:.4f}')
    for i in corr:
        if P[i] != Y[i]:
            print(f'  {Y[i]:14s} 主={P[i]:14s} kNN={knn_preds[i]:14s} m={M[i]:.3f} | {TX[i][:30]}')

    # 组合规则：主模型错且 kNN 对 -> 若 kNN 与 top2 一致则翻盘
    rescued = sum(1 for i in range(n) if P[i] != Y[i] and knn_preds[i] == Y[i])
    hurt = sum(1 for i in range(n) if P[i] == Y[i] and knn_preds[i] != Y[i])
    print(f'\nkNN 可救回 {rescued} 条错误，但会误伤 {hurt} 条正确（粗暴替换不可行）')


if __name__ == '__main__':
    main()

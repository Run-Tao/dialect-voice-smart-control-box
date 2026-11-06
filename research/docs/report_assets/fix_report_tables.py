# -*- coding: utf-8 -*-
"""收尾：把数据消融/OOD 两张表改为要点文字，使正式编号表回到 12 个，并统一编号。"""
from pathlib import Path

P = Path(__file__).resolve().parents[2] / 'docs' / 'v3_report.md'
s = P.read_text(encoding='utf-8')


def rep(old, new, tag, count=1):
    global s
    if old not in s:
        raise SystemExit(f'[MISS] {tag}')
    s = s.replace(old, new, count)


# 1) 数据消融表 -> 文字
old_abl = '''| test 视图 | v3-only（seed42） | v3.1（seed42） | 差值 |
|---|---:|---:|---:|
| full 483 | 0.8778 | **0.9130** | **+3.52pp** |
| correction 98 条 | 0.5408 | **0.6429** | **+10.21pp** |
| 非 correction 385 条 | 0.9636 | 0.9818 | +1.82pp |
| matched-v2 111 条 | 0.8649 | 0.9099 | +4.50pp |

**表 13：v3 vs v3.1 数据消融（同种子 42；v3-only 为单种子，v3.1 取集成工件中的 seed42）**
'''
new_abl = '''各测试视图的结果：full 483 条 **0.8778→0.9130（+3.52pp）**；correction 98 条
**0.5408→0.6429（+10.21pp）**；非 correction 385 条 0.9636→0.9818（+1.82pp）；
matched-v2 111 条 0.8649→0.9099（+4.50pp）。v3-only 为单种子（42），v3.1 取集成工件中的
同 seed42，保证随机种子一致。
'''
rep(old_abl, new_abl, 'ablation-table')

# 2) OOD 表 -> 要点
old_ood = '''| OOD 分数 | val AUC（已知 vs 未知校准） | val 选中阈值下的 FAR | test FAR | test 已知覆盖率 |
|---|---:|---:|---:|---:|
| `max_prob`（温度后最大类概率） | 0.8715 | 0.3333 | 0.3611 | 0.8841 |
| `cos_centroid`（类质心余弦） | **0.9517** | **0.0556** | 0.3611 | 0.9130 |
| `mahalanobis`（马氏距离，取负） | 0.8526 | 0.4167 | 0.4444 | 0.8509 |
| `knn_distance`（k=5 近邻距离，取负） | 0.8927 | 0.1944 | **0.1667** | 0.8923 |

**表 14：4 种 OOD 分数的可分性与阈值迁移（`ood.json`）**
'''
new_ood = '''- `max_prob`（温度后最大类概率）：AUC 0.8715；val 选中阈值下 FAR 0.3333 → test FAR
  0.3611（test 覆盖率 0.8841）；
- `cos_centroid`（类质心余弦）：AUC **0.9517**；val FAR **0.0556** → test FAR 0.3611
  （覆盖率 0.9130）；
- `mahalanobis`（马氏距离，取负）：AUC 0.8526；val FAR 0.4167 → test FAR 0.4444
  （覆盖率 0.8509）；
- `knn_distance`（k=5 近邻距离，取负）：AUC 0.8927；val FAR 0.1944 → test FAR **0.1667**
  （覆盖率 0.8923）。
'''
rep(old_ood, new_ood, 'ood-table')

# 3) 重新编号为 8..12
for old, new, tag in [
    ('**表 10：配对统计检验', '**表 9：配对统计检验', 'renum-stats'),
    ('**表 11：分层准确率', '**表 10：分层准确率', 'renum-strat'),
    ('**表 12：correction 语域', '**表 11：correction 语域', 'renum-corr'),
    ('**表 15：拒识指标', '**表 12：拒识指标', 'renum-rej'),
]:
    rep(old, new, tag)

P.write_text(s, encoding='utf-8', newline='')
print('tables renumbered; 字数=', len(s))

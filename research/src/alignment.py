# -*- coding: utf-8 -*-
"""固定长度处理和DTW对齐。两者不等价，分别用于不同环节。"""
import numpy as np


def fixed_length_sequence(x, target_len):
    """沿时间轴线性插值到target_len帧。输入[T,D]，输出[target_len,D]。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("输入必须是二维特征矩阵[T,D]")
    if target_len <= 0:
        raise ValueError("target_len必须为正数")
    if x.shape[0] == 0:
        raise ValueError("特征序列为空")
    if x.shape[0] == target_len:
        return x.astype(np.float32)
    if x.shape[0] == 1:
        return np.repeat(x, target_len, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, x.shape[0])
    dst = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, x.shape[1]), dtype=np.float64)
    for d in range(x.shape[1]):
        out[:, d] = np.interp(dst, src, x[:, d])
    return out.astype(np.float32)


def dtw_align_pair(x, y, metric='euclidean'):
    """DTW对齐两个序列，返回对齐后的序列、路径和累计距离。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x和y必须是二维特征矩阵[T,D]")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"特征维度不一致: {x.shape[1]} vs {y.shape[1]}")
    if metric != 'euclidean':
        raise ValueError("当前仅支持euclidean")

    n, m = x.shape[0], y.shape[0]
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = np.linalg.norm(x[i - 1] - y[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        choices = [(cost[i - 1, j - 1], i - 1, j - 1),
                   (cost[i - 1, j], i - 1, j),
                   (cost[i, j - 1], i, j - 1)]
        _, i, j = min(choices, key=lambda z: z[0])
    path.reverse()
    if not path or path[0] != (0, 0):
        path.insert(0, (0, 0))
    if path[-1] != (n - 1, m - 1):
        path.append((n - 1, m - 1))
    xi = [p[0] for p in path]
    yj = [p[1] for p in path]
    return x[xi], y[yj], path, float(cost[n, m])


def dtw_distance(x, y, metric='euclidean'):
    _, _, _, dist = dtw_align_pair(x, y, metric=metric)
    return dist


def _downsample_for_dtw(x, max_frames):
    x = np.asarray(x, dtype=np.float64)
    if max_frames is None or x.shape[0] <= max_frames:
        return x
    idx = np.linspace(0, x.shape[0]-1, int(max_frames)).astype(int)
    return x[idx]


def dtw_distance_nd(x, y, norm='path', max_frames=None):
    """向量化DTW距离，与 dtw_distance_general 语义完全一致但更快。

    仅做「计算累积代价矩阵+回溯步数」这一等价实现：
    - 局部距离同为欧氏距离 L2；
    - 递推同为 min(上、左、左上)；
    - 归一化分母同为回溯路径步数（path）或 max(n, m)（max_len）。
    逐元素代价用广播计算，避免 Python 双重循环，用于实验批量评估；
    端到端控制链路仍走原 dtw_distance_general，行为不变。
    """
    x = _downsample_for_dtw(x, max_frames).astype(np.float64)
    y = _downsample_for_dtw(y, max_frames).astype(np.float64)
    if x.shape[1] != y.shape[1]:
        raise ValueError('x/y特征维度不一致')
    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        raise ValueError('特征序列为空')
    # 局部距离矩阵 [n, m]
    diff = x[:, None, :] - y[None, :, :]
    D = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    # DP 表形状 [n+1, m+1]，形状与 dtw_distance_general 完全一致
    C = np.empty((n + 1, m + 1), dtype=np.float64)
    C[0, :] = np.inf
    C[:, 0] = np.inf
    C[0, 0] = 0.0
    # 注意：递推 C[i,j] = D[i-1,j-1] + min(C[i-1,j], C[i,j-1], C[i-1,j-1]) 中
    # C[i,j-1] 依赖同一行刚更新的值，因此必须按 j 递增顺序处理（不能整体向量化）。
    # 这里用逐元素 Python 循环保证与 dtw_distance_general 数值完全一致。
    for i in range(1, n + 1):
        prev = C[i - 1]           # C[i-1, :]，长度 m+1
        row = C[i]                # C[i, :]
        row[1:] = D[i - 1]
        for j in range(1, m + 1):
            row[j] = row[j] + min(prev[j], row[j - 1], prev[j - 1])
    # 回溯步数：与原实现相同的前向回溯
    i, j = n, m
    steps = 0
    while i > 0 or j > 0:
        steps += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            a, b, c = C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]
            if a <= b and a <= c:
                i -= 1
                j -= 1
            elif b <= c:
                i -= 1
            else:
                j -= 1
    if norm == 'path':
        denom = max(steps, 1)
    elif norm == 'max_len':
        denom = max(n, m)
    elif norm == 'none':
        denom = 1.0
    else:
        raise ValueError('norm必须是path/max_len/none')
    return float(C[n, m] / denom)


def _local_distance(a, b, metric='euclidean'):
    if metric == 'euclidean':
        return float(np.linalg.norm(a - b))
    if metric == 'cosine':
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))
    raise ValueError('metric必须是euclidean或cosine')


def dtw_distance_general(x, y, metric='euclidean', norm='path', max_frames=None):
    """可配置局部距离和归一化方式的DTW距离。"""
    x = _downsample_for_dtw(x, max_frames)
    y = _downsample_for_dtw(y, max_frames)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError('x/y必须是二维矩阵')
    if x.shape[1] != y.shape[1]:
        raise ValueError('x/y特征维度不一致')
    n, m = x.shape[0], y.shape[0]
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = _local_distance(x[i - 1], y[j - 1], metric)
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    i, j = n, m
    steps = 0
    while i > 0 or j > 0:
        steps += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choices = [(cost[i - 1, j - 1], i - 1, j - 1),
                       (cost[i - 1, j], i - 1, j),
                       (cost[i, j - 1], i, j - 1)]
            _, i, j = min(choices, key=lambda z: z[0])
    if norm == 'path':
        denom = max(steps, 1)
    elif norm == 'max_len':
        denom = max(n, m)
    elif norm == 'none':
        denom = 1.0
    else:
        raise ValueError('norm必须是path/max_len/none')
    return float(cost[n, m] / denom)

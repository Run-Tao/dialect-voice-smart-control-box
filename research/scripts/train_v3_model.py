# -*- coding: utf-8 -*-
"""第三阶段：训练 6 类命令识别主模型（log-mel + Δ + 1D-CNN，多种子）。

流程
----
1. 读取 ``config_v3.json`` 指定的清单与固定划分（默认 v3.1；缺失时回退 v3）；
2. 用 ``src.features_v3.extract_sequence`` 提取 log-mel+Δ 帧级特征（确定性、纯 numpy）；
3. 线性插值到固定 256 帧，训练 ``N`` 个种子的 1D-CNN（PyTorch，CPU），
   仅在验证集上做早停与模型选择；
4. 验证集上做温度标定（标定仅用验证集，测试集不参与任何选择）；
5. 导出为纯 numpy 推理参数（``src/v3_cnn_numpy.py``），保存：
   - ``data/models/v3_cnn_ensemble.npz``：模型工件的唯一来源（含 JSON meta）；
   - ``outputs/v3_experiments/train_report.json``：训练协议、数据哈希、
     每种子 val 指标、集成 val 指标、温度、耗时。**不写 test 指标**。

复现性：所有随机源（种子、PyTorch、numpy）固定；数据哈希与命令写入报告；
同一机器同一版本下可重复（CPU 浮点差异可能导致极小波动，报告中记录）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.features_v3 import FeatureSpec, extract_sequence, output_dim  # noqa: E402
from src.v3_cnn_numpy import resample_to_frames  # noqa: E402

INTENTS = ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL']


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_data_paths(cfg: dict):
    data = cfg['data']
    manifest = BASE / data['manifest']
    split = BASE / data['split_file']
    if not manifest.exists() and data.get('fallback_manifest'):
        manifest = BASE / data['fallback_manifest']
    if not split.exists() and data.get('fallback_split_file'):
        split = BASE / data['fallback_split_file']
    if not manifest.exists():
        raise FileNotFoundError(f'清单不存在: {manifest}')
    if not split.exists():
        raise FileNotFoundError(f'划分文件不存在: {split}')
    return manifest, split


def resolve_audio_path(raw: str, audio_root: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for cand in (BASE / p, audio_root / p):
        if cand.exists():
            return cand
    return BASE / p


def load_records(cfg: dict, manifest: Path, split: Path):
    with open(split, 'r', encoding='utf-8-sig', newline='') as f:
        group2subset = {r['expression_group_id']: r['subset'] for r in csv.DictReader(f)}
    audio_root = BASE / cfg['data'].get('audio_root', 'data/tts_corpus')
    records = []
    with open(manifest, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            gid = row.get('expression_group_id', '')
            if gid not in group2subset:
                # 划分文件是唯一的 subset 依据；不在划分里的样本不允许进入训练
                continue
            status = row.get('quality_status') or 'ok'
            if status != cfg['data'].get('min_quality_status', 'ok'):
                continue
            cmd = row['command']
            if cmd not in INTENTS:
                continue
            records.append({
                'utterance_id': row['utterance_id'],
                'audio_path': str(resolve_audio_path(row['audio_path'], audio_root)),
                'command': cmd,
                'subset': group2subset[gid],
                'expression_group_id': gid,
                'register': row.get('register', ''),
                'condition': row.get('condition', ''),
                'voice': row.get('voice_identity', row.get('speaker_id', '')),
                'text': row.get('text', ''),
            })
    return records


def feature_spec_from_config(cfg: dict) -> FeatureSpec:
    f = cfg['features']
    return FeatureSpec(
        sample_rate=int(f['sample_rate']), n_fft=int(f['n_fft']), hop=int(f['hop']),
        n_mels=int(f['n_mels']), n_mfcc=int(f.get('n_mfcc', 13)),
        fmin=float(f['fmin']), fmax=float(f['fmax']) if f.get('fmax') else None,
        preemphasis=float(f['preemphasis']), use_cmvn=bool(f['use_cmvn']),
        include_delta=bool(f['include_delta']), include_delta2=bool(f.get('include_delta2', False)),
        delta_width=int(f.get('delta_width', 2)),
        include_mfcc=bool(f.get('include_mfcc', True)),
        include_logmel=bool(f.get('include_logmel', False)),
        logmel_cmvn=bool(f.get('logmel_cmvn', True)),
    )


def build_arrays(records, spec: FeatureSpec, n_frames: int, log=print):
    from src.audio_io import load_wav
    X = np.empty((len(records), n_frames, spec_output_dim(spec)), dtype=np.float32)
    t0 = time.time()
    for i, r in enumerate(records):
        audio, sr = load_wav(r['audio_path'])
        seq = extract_sequence(audio, sr, spec)
        X[i] = resample_to_frames(seq, n_frames)
        if (i + 1) % 500 == 0:
            log(f'  特征 {i + 1}/{len(records)}  {time.time() - t0:.1f}s')
    log(f'  特征完成 {len(records)} 条，耗时 {time.time() - t0:.1f}s，dim={X.shape[2]}')
    y = np.asarray([INTENTS.index(r['command']) for r in records], dtype=np.int64)
    return X, y


def spec_output_dim(spec: FeatureSpec) -> int:
    return int(output_dim(spec))


class CmdCNN:
    """延迟导入 PyTorch 的训练模型（避免推理环境依赖）。"""

    @staticmethod
    def build(in_dim: int, n_cls: int, dropout: float):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(in_dim, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                )
                self.drop = nn.Dropout(dropout)
                self.fc = nn.Linear(256, n_cls)

            def embed(self, x):
                import torch
                h = self.net(x)
                return torch.cat([h.mean(-1), h.max(-1).values], dim=1)

            def forward(self, x):
                return self.fc(self.drop(self.embed(x)))

        return _Net()


def augment_batch(x, cfg):
    """SpecAugment + 时间平移 + 随机增益（训练时在线增广）。"""
    import torch
    aug = cfg['model']['augment']
    B, C, T = x.shape
    if aug.get('time_shift', 0) > 0:
        s = int(aug['time_shift'])
        shifts = torch.randint(-s, s + 1, (B,))
        x = torch.stack([torch.roll(x[i], int(shifts[i]), dims=-1) for i in range(B)])
    for i in range(B):
        if np.random.rand() < float(aug.get('p_freq_mask', 0.7)):
            w = int(np.random.randint(2, max(3, int(aug['freq_mask_max']) // 2 + 2)))
            f0 = int(np.random.randint(0, C - w))
            x[i, f0:f0 + w, :] = 0.0
        if np.random.rand() < float(aug.get('p_time_mask', 0.7)):
            w = int(np.random.randint(5, max(6, int(aug['time_mask_max']) // 2 + 2)))
            t0 = int(np.random.randint(0, T - w))
            x[i, :, t0:t0 + w] = 0.0
    if aug.get('gain_db', 0) > 0:
        g = float(aug['gain_db'])
        gain = torch.from_numpy(np.exp(np.random.uniform(-g, g, (B, 1, 1))).astype(np.float32))
        x = x * gain
    return x


def macro_f1(y, p, n_cls=6):
    out = []
    for c in range(n_cls):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(out))


def confusion(y, p, n_cls=6):
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for a, b in zip(y, p):
        cm[a, b] += 1
    return cm.tolist()


def train_one_seed(seed, Xtr, ytr, Xva, yva, cfg, log=print):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    model_cfg = cfg['model']
    model = CmdCNN.build(Xtr.shape[1], len(INTENTS), float(model_cfg.get('dropout', 0.3)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(model_cfg['lr']),
                            weight_decay=float(model_cfg['weight_decay']))
    epochs = int(model_cfg['max_epochs'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=float(model_cfg.get('label_smoothing', 0.05)))
    bs = int(model_cfg['batch_size'])
    patience = int(model_cfg['patience'])
    best = (-1.0, None, -1)
    history = []
    N = Xtr.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        losses = []
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb, yb = augment_batch(Xtr[idx].clone(), cfg), ytr[idx]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva).argmax(1).numpy()
        acc = float(np.mean(pv == yva.numpy()))
        history.append({'epoch': ep, 'loss': float(np.mean(losses)), 'val_accuracy': acc})
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= patience:
            break
    model.load_state_dict(best[1])
    model.eval()
    log(f'  seed={seed} best_val={best[0]:.4f} @ep{best[2]} '
        f'({time.time() - t0:.0f}s, {len(history)} epochs)')
    return model, {'seed': seed, 'best_val_accuracy': best[0], 'best_epoch': best[2],
                   'epochs_run': len(history), 'train_seconds': round(time.time() - t0, 1),
                   'history': history}


def ensemble_probs(models, X, bs=128):
    import torch
    probs = []
    Xt = torch.from_numpy(X) if not isinstance(X, torch.Tensor) else X
    with torch.no_grad():
        for m in models:
            out = []
            for i in range(0, Xt.shape[0], bs):
                out.append(torch.softmax(m(Xt[i:i + bs]), dim=1).numpy())
            probs.append(np.concatenate(out, 0))
    return np.mean(np.stack(probs, 0), axis=0), probs


def calibrate_temperature(logits, y, grid):
    """在验证集上最小化 NLL 选温度（只用于置信度/拒识，不改变 argmax）。"""
    z = logits.astype(np.float64)
    best = (np.inf, 1.0)
    for t in grid:
        p = np.exp((z / t) - (z / t).max(1, keepdims=True))
        p = p / p.sum(1, keepdims=True)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))
        if nll < best[0]:
            best = (nll, float(t))
    return best[1], best[0]


def export_params(model):
    from src.v3_cnn_numpy import collect_arrays
    return collect_arrays({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def main():
    ap = argparse.ArgumentParser(description='训练 v3 六类命令 CNN 主模型')
    ap.add_argument('--config', default=str(BASE / 'config_v3.json'))
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--split', default=None)
    ap.add_argument('--out', default=str(BASE / 'data/models/v3_cnn_ensemble.npz'))
    ap.add_argument('--report', default=None)
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--smoke', action='store_true', help='冒烟：少量样本 + 2 epoch')
    args = ap.parse_args()

    import torch
    if args.threads:
        torch.set_num_threads(int(args.threads))
    log = lambda *a: print(*a, flush=True)  # noqa: E731

    cfg = load_config(Path(args.config))
    if args.seeds:
        cfg['model']['seeds'] = args.seeds
    if args.max_epochs is not None:
        cfg['model']['max_epochs'] = args.max_epochs
    if args.patience is not None:
        cfg['model']['patience'] = args.patience
    manifest, split = resolve_data_paths(cfg)
    if args.manifest:
        manifest = Path(args.manifest)
    if args.split:
        split = Path(args.split)
    records = load_records(cfg, manifest, split)
    counts = {s: sum(1 for r in records if r['subset'] == s) for s in ('train', 'val', 'test')}
    log(f'数据：{manifest.name} + {split.name}，共 {len(records)} 条，split={counts}')
    spec = feature_spec_from_config(cfg)
    n_frames = int(cfg['features']['n_frames'])
    X, y = build_arrays(records, spec, n_frames, log=log)
    # build_arrays 返回 [N, n_frames, D]（便于逐条插值），网络输入需要 [N, D, n_frames]
    X = np.ascontiguousarray(X.transpose(0, 2, 1))
    assert X.shape[2] == n_frames and X.shape[1] == spec_output_dim(spec), \
        f'特征转置后形状异常: {X.shape}，期望 [N,{spec_output_dim(spec)},{n_frames}]'
    log(f'  网络输入: N={X.shape[0]}, in_dim={X.shape[1]}, n_frames={X.shape[2]}')

    idx = {s: np.array([r['subset'] == s for r in records]) for s in ('train', 'val', 'test')}
    if args.smoke:
        # 冒烟模式：每子集只取少量样本，验证流程
        for s in idx:
            take = np.where(idx[s])[0][:120]
            idx[s] = np.zeros(len(records), dtype=bool)
            idx[s][take] = True
        log(f'冒烟模式索引: { {s: int(idx[s].sum()) for s in idx} }')

    Xtr = torch.from_numpy(np.ascontiguousarray(X[idx['train']]))
    ytr = torch.from_numpy(y[idx['train']])
    Xva = torch.from_numpy(np.ascontiguousarray(X[idx['val']]))
    yva = torch.from_numpy(y[idx['val']])

    t_start = time.time()
    models, seed_reports = [], []
    for seed in cfg['model']['seeds']:
        m, rep = train_one_seed(seed, Xtr, ytr, Xva, yva, cfg, log=log)
        models.append(m)
        seed_reports.append(rep)
    elapsed = time.time() - t_start

    # 集成验证
    import torch as _torch
    Xva_t = _torch.from_numpy(np.ascontiguousarray(X[idx['val']]))
    probs_ens, probs_each = ensemble_probs(models, Xva_t)
    pv = probs_ens.argmax(1)
    yv = y[idx['val']]
    val_acc = float(np.mean(pv == yv))
    val_f1 = macro_f1(yv, pv)
    log(f'集成 val accuracy={val_acc:.4f} macro_f1={val_f1:.4f}')

    # 温度标定（集成 logits）
    with _torch.no_grad():
        logits_list = []
        for m in models:
            logits_list.append(m(Xva_t).numpy())
    mean_logits = np.mean(np.stack(logits_list, 0), 0)
    Tgrid = np.linspace(float(cfg['rejection']['temperature_grid_min']),
                        float(cfg['rejection']['temperature_grid_max']),
                        int(cfg['rejection']['temperature_grid_steps']))
    temperature, nll = calibrate_temperature(mean_logits, yv, Tgrid)
    log(f'温度标定 T={temperature:.3f} val_nll={nll:.4f}')

    # 导出
    payload = {}
    for si, m in enumerate(models):
        for k, v in export_params(m).items():
            payload[f'cnn_s{si}/{k}'] = v.astype(np.float32)
    meta = {
        'format': 'v3_cnn_ensemble/1.0',
        'arch': 'cmd_cnn_v1',
        'classes': INTENTS,
        'seeds': list(cfg['model']['seeds']),
        'n_seeds': len(models),
        'feature_spec': spec.to_dict(),
        'n_frames': n_frames,
        'in_dim': int(X.shape[1]),
        'temperature': float(temperature),
        'temperature_grid': [float(Tgrid[0]), float(Tgrid[-1]), len(Tgrid)],
        'val': {'accuracy': val_acc, 'macro_f1': val_f1, 'n': int(len(yv)),
                'confusion_matrix': confusion(yv, pv)},
        'data': {
            'manifest': str(manifest.relative_to(BASE)) if str(manifest).startswith(str(BASE)) else str(manifest),
            'manifest_sha256': sha256_file(manifest),
            'split': str(split.relative_to(BASE)) if str(split).startswith(str(BASE)) else str(split),
            'split_sha256': sha256_file(split),
            'n_train': int(idx['train'].sum()), 'n_val': int(idx['val'].sum()),
            'n_test': int(idx['test'].sum()),
            'n_expression_groups': len({r['expression_group_id'] for r in records}),
        },
        'training': {
            'max_epochs': int(cfg['model']['max_epochs']),
            'patience': int(cfg['model']['patience']),
            'batch_size': int(cfg['model']['batch_size']),
            'lr': float(cfg['model']['lr']),
            'weight_decay': float(cfg['model']['weight_decay']),
            'dropout': float(cfg['model'].get('dropout', 0.3)),
            'augment': cfg['model']['augment'],
            'torch_version': _torch.__version__,
            'python': platform.python_version(),
            'platform': platform.platform(),
            'threads': int(_torch.get_num_threads()),
        },
        'rejection': {
            'score': cfg['rejection']['score'],
            'threshold': None,
            'min_known_coverage': float(cfg['rejection']['min_known_coverage']),
            'target_far': float(cfg['rejection']['target_far']),
        },
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'train_seconds': round(elapsed, 1),
        'val_utterance_ids': [records[i]['utterance_id'] for i in np.where(idx['val'])[0]],
        'val_labels': yv.astype(np.int64).tolist(),
    }
    payload['val_probs'] = probs_ens.astype(np.float32)
    payload['__meta__'] = np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    log(f'模型已保存: {out} ({out.stat().st_size / 1e6:.2f} MB)')

    report = {
        'protocol': {'command': ' '.join(sys.argv), 'config': str(args.config),
                     'manifest': meta['data']['manifest'], 'split': meta['data']['split'],
                     'data_hashes': {k: meta['data'][k] for k in
                                     ('manifest_sha256', 'split_sha256')},
                     'selection': '早停与温度标定仅用验证集；本脚本不评估测试集',
                     'seeds': list(cfg['model']['seeds'])},
        'counts': {'records': len(records), **counts},
        'features': spec.to_dict(),
        'seeds': seed_reports,
        'ensemble_val': {'accuracy': val_acc, 'macro_f1': val_f1,
                         'confusion_matrix': confusion(yv, pv),
                         'n': int(len(yv))},
        'temperature': float(temperature),
        'train_seconds': round(elapsed, 1),
        'artifact': str(out),
        'created_at': meta['created_at'],
    }
    report_path = Path(args.report) if args.report else (
        BASE / cfg['paths']['outputs_dir'] / 'train_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'训练报告: {report_path}')


if __name__ == '__main__':
    main()

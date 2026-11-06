# -*- coding: utf-8 -*-
"""统一实验运行器。"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from .representations import (
    FeatureExtractor, sentence_vector,
    build_cosine_templates, build_dtw_templates
)
from .baselines import predict_cosine, predict_dtw
from .classifiers import StandardScalerNumpy, SoftmaxRegression, LinearSVM
from .metrics import classification_metrics
from .results import save_json, save_summary_csv
from .mapping import (
    OLSMapping, RidgeMapping, SVDFeatureTransformer, align_pair_sequences,
    pair_records_by_command, mapping_diagnostics
)


def _split_meta(split):
    return {
        'seed': split.seed,
        'train_speakers': split.train_speakers,
        'val_speakers': split.val_speakers,
        'test_speakers': split.test_speakers,
        'train_size': len(split.train),
        'val_size': len(split.val),
        'test_size': len(split.test),
    }


def _template_info(templates, method):
    if method == 'dtw':
        return {k: len(v) for k, v in templates.items()}
    return {k: list(v.shape) for k, v in templates.items()}


def _predictions_to_rows(method, records, preds):
    rows = []
    for r, p in zip(records, preds):
        rows.append({
            'method': method,
            'utterance_id': r.utterance_id,
            'speaker_id': r.speaker_id,
            'true_command': r.command,
            'pred_command': p.command,
            'score': p.score,
            'distance': '' if p.distance is None else p.distance,
            'scores': json.dumps(p.scores, ensure_ascii=False),
        })
    return rows


def _save_result(out_dir, method, result, prediction_rows):
    out_dir = Path(out_dir) / method
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / 'result.json', result)
    save_summary_csv(
        out_dir / 'predictions.csv',
        prediction_rows,
        ['method', 'utterance_id', 'speaker_id', 'true_command',
         'pred_command', 'score', 'distance', 'scores']
    )
    return out_dir


def run_template_method(method, split, config):
    feat_cfg = config['features']
    extractor = FeatureExtractor(feat_cfg)
    target_len = config['alignment']['fixed_length_frames']
    template_language = config.get('baselines', {}).get('template_language', None)

    t0 = time.perf_counter()
    if method == 'cosine':
        templates = build_cosine_templates(split.train, extractor, target_len, template_language)
        test_preds = []
        for r in split.test:
            q = extractor.fixed_sequence_from_record(r, target_len)
            test_preds.append(predict_cosine(q, templates))
    elif method == 'dtw':
        templates = build_dtw_templates(split.train, extractor, template_language)
        dtw_cfg = config.get('baselines', {}).get('dtw', {})
        normalize = dtw_cfg.get('normalize', 'path')
        max_frames = dtw_cfg.get('max_frames', None)
        test_preds = []
        for r in split.test:
            q = extractor.sequence_from_record(r)
            test_preds.append(predict_dtw(q, templates, normalize=normalize, max_frames=max_frames))
    else:
        raise ValueError(f"未知模板方法: {method}")
    infer_time = time.perf_counter() - t0

    y_true = [r.command for r in split.test]
    y_pred = [p.command for p in test_preds]
    metrics = classification_metrics(y_true, y_pred)
    return {
        'method': method,
        'split': _split_meta(split),
        'data_protocol': {
            'templates_from': 'train',
            'evaluation_on': 'test',
            'val_used_for': 'not_used',
        },
        'templates': _template_info(templates, method),
        'metrics': metrics,
        'timing': {
            'train_seconds': 0.0,
            'inference_seconds_total': infer_time,
            'inference_seconds_per_sample': infer_time / max(len(split.test), 1),
        },
    }, test_preds


def _extract_xy(records, extractor):
    X, y = [], []
    for r in records:
        seq = extractor.sequence_from_record(r)
        X.append(sentence_vector(seq))
        y.append(r.command)
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def _make_model(method, cfg):
    if method == 'logreg':
        return SoftmaxRegression(
            l2=cfg.get('l2', 1e-3),
            learning_rate=cfg.get('learning_rate', 0.1),
            epochs=cfg.get('epochs', 300),
        )
    if method == 'svm':
        return LinearSVM(
            l2=cfg.get('l2', 1e-3),
            learning_rate=cfg.get('learning_rate', 0.05),
            epochs=cfg.get('epochs', 500),
        )
    raise ValueError(method)


def run_classifier_method(method, split, config):
    extractor = FeatureExtractor(config['features'])
    X_train, y_train = _extract_xy(split.train, extractor)
    X_val, y_val = _extract_xy(split.val, extractor)
    X_test, y_test = _extract_xy(split.test, extractor)

    # 标准化只在训练集拟合
    scaler = StandardScalerNumpy().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    cfg = config.get('baselines', {}).get(method, {})
    tune = bool(cfg.get('tune_on_validation', False))
    best_l2 = cfg.get('l2', 1e-3)
    tuning_log = []
    if tune:
        grid = cfg.get('l2_grid', [best_l2])
        best_acc = -1.0
        for l2 in grid:
            m = _make_model(method, {**cfg, 'l2': l2})
            m.fit(X_train_s, y_train)
            val_acc = float(np.mean(m.predict(X_val_s) == y_val))
            tuning_log.append({'l2': l2, 'val_accuracy': val_acc})
            if val_acc > best_acc:
                best_acc = val_acc
                best_l2 = l2

    t0 = time.perf_counter()
    model = _make_model(method, {**cfg, 'l2': best_l2})
    model.fit(X_train_s, y_train)
    train_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    y_pred = model.predict(X_test_s)
    infer_time = time.perf_counter() - t1

    probs = model.predict_proba(X_test_s)
    preds = []
    from .baselines import Prediction
    for i, cmd in enumerate(y_pred):
        score = float(np.max(probs[i]))
        score_map = {str(c): float(p) for c, p in zip(model.classes_, probs[i])}
        preds.append(Prediction(command=str(cmd), score=score, scores=score_map, confidence=score))

    metrics = classification_metrics(y_test, y_pred)
    return {
        'method': method,
        'split': _split_meta(split),
        'data_protocol': {
            'fit_on': 'train',
            'scaler_fit_on': 'train',
            'tuning_on': 'validation' if tune else 'not_used',
            'evaluation_on': 'test',
        },
        'hyperparameters': {
            'selected_l2': best_l2,
            'tuning_log': tuning_log,
            'sentence_feature_dim': int(X_train.shape[1]),
        },
        'metrics': metrics,
        'timing': {
            'train_seconds': train_time,
            'inference_seconds_total': infer_time,
            'inference_seconds_per_sample': infer_time / max(len(split.test), 1),
        },
    }, preds


def run_experiment(method, split, config, output_root='outputs/baselines'):
    if method in ('cosine', 'dtw'):
        result, preds = run_template_method(method, split, config)
    elif method in ('logreg', 'svm'):
        result, preds = run_classifier_method(method, split, config)
    elif method in ('ols', 'ridge'):
        result, preds = run_mapping_method(method, split, config)
    else:
        raise ValueError(f"未知方法: {method}")

    result['run_id'] = f"{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows = _predictions_to_rows(method, split.test, preds)
    out_dir = _save_result(output_root, method, result, rows)
    return result, out_dir


def run_mapping_method(method, split, config):
    """OLS/Ridge映射 + 余弦模板匹配。"""
    mapping_cfg = config.get('mapping', {})
    extractor = FeatureExtractor(config['features'])
    target_len = config['alignment']['fixed_length_frames']
    source_lang = mapping_cfg.get('source_language', None)
    target_lang = mapping_cfg.get('target_language', None)
    alignment = mapping_cfg.get('alignment', 'fixed_length')
    svd_mode = mapping_cfg.get('svd_mode', 'none')
    svd_rank = mapping_cfg.get('svd_rank', None)
    fit_intercept = bool(mapping_cfg.get('fit_intercept', True))
    alpha = float(mapping_cfg.get('alpha', 1e-3))

    # 1) 仅使用训练集构造配对
    pairs = pair_records_by_command(split.train, source_lang, target_lang)
    src_seqs, tgt_seqs = [], []
    for srec, trec in pairs:
        sx = extractor.sequence_from_record(srec)
        ty = extractor.sequence_from_record(trec)
        ax, ay = align_pair_sequences(sx, ty, alignment, target_len)
        src_seqs.append(ax)
        tgt_seqs.append(ay)
    X_pairs = np.stack(src_seqs, axis=0)
    Y_pairs = np.stack(tgt_seqs, axis=0)

    # 2) SVD变换器只在训练配对序列上拟合
    svd = SVDFeatureTransformer(svd_mode, svd_rank).fit(src_seqs + tgt_seqs)
    X_svd = np.stack([svd.transform(x) for x in src_seqs], axis=0)
    Y_svd = np.stack([svd.transform(y) for y in tgt_seqs], axis=0)

    # 3) 映射训练
    t0 = time.perf_counter()
    if method == 'ols':
        model = OLSMapping(fit_intercept=fit_intercept).fit(X_svd, Y_svd)
    elif method == 'ridge':
        model = RidgeMapping(alpha=alpha, fit_intercept=fit_intercept).fit(X_svd, Y_svd)
    else:
        raise ValueError(method)
    train_time = time.perf_counter() - t0
    diag = mapping_diagnostics(model, X_svd, Y_svd)

    # 4) 目标模板：来自训练集目标语言记录
    from .representations import select_template_records
    target_records = select_template_records(split.train, target_lang)
    grouped = {}
    for r in target_records:
        seq = extractor.fixed_sequence_from_record(r, target_len)
        seq = svd.transform(seq)
        grouped.setdefault(r.command, []).append(seq)
    templates = {
        cmd: np.mean(np.stack(seqs, axis=0), axis=0).astype(np.float32)
        for cmd, seqs in grouped.items()
    }
    if not templates:
        raise ValueError("没有构造出目标语言模板")

    # 5) 测试：方言 -> SVD -> 映射 -> 余弦模板
    t1 = time.perf_counter()
    test_preds = []
    for r in split.test:
        q = extractor.fixed_sequence_from_record(r, target_len)
        q = svd.transform(q)
        q_mapped = model.transform(q)
        test_preds.append(predict_cosine(q_mapped, templates))
    infer_time = time.perf_counter() - t1

    y_true = [r.command for r in split.test]
    y_pred = [p.command for p in test_preds]
    metrics = classification_metrics(y_true, y_pred)
    return {
        'method': method,
        'split': _split_meta(split),
        'data_protocol': {
            'mapping_fit_on': 'train_pairs',
            'svd_fit_on': 'train_pairs',
            'templates_from': 'train_target_records',
            'evaluation_on': 'test',
            'val_used_for': 'not_used',
        },
        'mapping': {
            'method': method,
            'fit_intercept': fit_intercept,
            'alpha': alpha if method == 'ridge' else None,
            'svd_mode': svd_mode,
            'svd_rank': svd_rank,
            'alignment': alignment,
            'source_language': source_lang,
            'target_language': target_lang,
            'num_pairs': len(pairs),
            'train_rmse': diag['rmse'],
            'train_mae': diag['mae'],
            'train_mean_cosine': diag['mean_cosine'],
            'omega_shape': list(model.omega_.shape),
        },
        'templates': {k: list(v.shape) for k, v in templates.items()},
        'metrics': metrics,
        'timing': {
            'train_seconds': train_time,
            'inference_seconds_total': infer_time,
            'inference_seconds_per_sample': infer_time / max(len(split.test), 1),
        },
    }, test_preds

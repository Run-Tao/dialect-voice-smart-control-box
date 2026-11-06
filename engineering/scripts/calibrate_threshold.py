# -*- coding: utf-8 -*-
"""拒识阈值标定：只在验证集与未知校准集上选阈值。

原则：阈值不能看测试集。本脚本用
- 已知命令的**验证集**（按表达组划分，来自 split 文件的 subset=val）；
- 未知指令的**校准集**（unknown 的 calibration 部分）；
扫阈值曲线，按「已知覆盖率不低于 min_coverage 且未知误接受率不高于 target_far」
选阈值，两条都达不到时退到 FAR 最小且覆盖率最高的一档。

示例：
    python scripts/calibrate_threshold.py \
        --known-manifest "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\manifest_yue_v3_1_vad.csv" \
        --split "D:\\Run_Tao\\DialectVoice_Research\\data\\splits\\track_a_expression_split_v3_1.csv" \
        --unknown-manifest "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\unknown_v3_vad.csv" \
        --model "D:\\Run_Tao\\DialectVoice_Research\\data\\models\\v3_cnn_ensemble_vad.npz" \
        --out "D:\\Run_Tao\\DialectVoice_Research\\data\\models\\v3_thresholds_vad.json"

阈值文件写在 Research 的 data/models/ 下作为唯一权威副本，
再由 scripts/sync_runtime.py 同步进 runtime/models/ 供部署使用。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ENGINEERING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINEERING_ROOT))


def load_split_subset(split_path: Path, subset: str):
    with open(split_path, 'r', encoding='utf-8-sig', newline='') as f:
        return {r['expression_group_id']: r['subset'] for r in csv.DictReader(f)}


def resolve_audio(raw: str, bases):
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for base in bases:
        for cand in (base / p, base / 'data' / 'tts_corpus' / p):
            if cand.exists():
                return cand
    return p


def score_files(predictor, paths):
    from runtime.audio_io import load_wav
    from runtime.features_v3 import extract_sequence
    out = []
    for path in paths:
        audio, sr = load_wav(path)
        pred = predictor.predict_features(
            extract_sequence(audio, sr, predictor.feature_spec))
        out.append({'path': str(path), 'max_prob': float(pred.confidence),
                    'command': pred.command})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--known-manifest', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--unknown-manifest', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True,
                        help='阈值 JSON 的权威副本路径（放在 Research 的 data/models/ 下）')
    parser.add_argument('--min-coverage', type=float, default=0.80,
                        help='已知命令最低覆盖率下限；低于此值用户需反复重说')
    parser.add_argument('--target-far', type=float, default=0.15,
                        help='未知指令误接受率上限；超过此值可能误动作')
    parser.add_argument('--curve-out', default=None, help='把完整阈值曲线写到该 JSON')
    args = parser.parse_args()

    from runtime.v3_model import V3Predictor
    predictor = V3Predictor.load(args.model)

    known_manifest = Path(args.known_manifest).resolve()
    unknown_manifest = Path(args.unknown_manifest).resolve()
    split_path = Path(args.split).resolve()
    bases = [known_manifest.parent, known_manifest.parent.parent,
             known_manifest.parent.parent.parent, unknown_manifest.parent]

    group2subset = load_split_subset(split_path, 'val')
    known_paths = []
    with open(known_manifest, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if group2subset.get(row.get('expression_group_id', '')) != 'val':
                continue
            p = resolve_audio(row['audio_path'], bases)
            if p.exists():
                known_paths.append(p)

    unknown_paths = []
    with open(unknown_manifest, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('text_split', 'calibration') != 'calibration':
                continue
            p = resolve_audio(row['audio_path'], bases)
            if p.exists():
                unknown_paths.append(p)

    if not known_paths or not unknown_paths:
        sys.exit(f'验证集 {len(known_paths)} 条 / 未知校准集 {len(unknown_paths)} 条，数据不足')

    known = score_files(predictor, known_paths)
    unknown = score_files(predictor, unknown_paths)

    grid = [i / 200.0 for i in range(1, 199)]
    curve = []
    for t in grid:
        cov = sum(1 for s in known if s['max_prob'] >= t) / len(known)
        far = sum(1 for s in unknown if s['max_prob'] >= t) / len(unknown)
        curve.append({'threshold': round(t, 4), 'known_coverage': round(cov, 4),
                      'unknown_false_accept_rate': round(far, 4)})

    feasible = [c for c in curve
                if c['known_coverage'] >= args.min_coverage
                and c['unknown_false_accept_rate'] <= args.target_far]
    if feasible:
        chosen = min(feasible, key=lambda c: (c['unknown_false_accept_rate'],
                                              -c['known_coverage']))
    else:
        chosen = min(curve, key=lambda c: (c['unknown_false_accept_rate'],
                                           -c['known_coverage']))
    payload = {
        'format': 'v3_thresholds/2.0-vad',
        'selected_threshold': chosen['threshold'],
        'score': 'max_prob',
        'temperature': predictor.temperature,
        'selected_on': 'validation (VAD-trimmed) + unknown calibration (VAD-trimmed)',
        'known_coverage_on_validation': chosen['known_coverage'],
        'unknown_false_accept_rate_on_calibration': chosen['unknown_false_accept_rate'],
        'n_known_val': len(known),
        'n_unknown_calibration': len(unknown),
        'min_known_coverage_target': args.min_coverage,
        'target_far': args.target_far,
        'feasible': bool(feasible),
        'note': ('阈值仅在验证集与未知校准集上选择；测试集未参与。'
                 '覆盖率与误接受率存在取舍，工作点由 min-coverage / target-far 两个'
                 '业务约束确定。'),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding='utf-8')
    if args.curve_out:
        Path(args.curve_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.curve_out).write_text(
            json.dumps({'curve': curve, 'chosen': chosen}, ensure_ascii=False, indent=2),
            encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

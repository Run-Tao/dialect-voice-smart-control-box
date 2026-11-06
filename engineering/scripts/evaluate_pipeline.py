# -*- coding: utf-8 -*-
"""管线评估：已知命令准确率、拒识后的接受样本准确率、未知指令误接受率。

与 scripts/train_* 的实验不同，这里评估的是**部署管线**（VAD + 分类 + 拒识），
用于工程学导论项目的测试报告。

示例：
    python scripts/evaluate_pipeline.py \
        --known "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\wav16k_v3\\edge" \
        --unknown "D:\\Run_Tao\\DialectVoice_Research\\data\\tts_corpus\\unknown_v3" \
        --out docs/pipeline_eval.json
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.realtime import DialectEngine
from src.vad import EnergyVad
from runtime.audio_io import load_wav

ROOT = Path(__file__).resolve().parents[1]


def label_from_name(name: str, classes) -> str:
    for c in classes:
        if name.startswith(c + '_'):
            return c
    return 'UNKNOWN'


def evaluate(engine, files, use_vad: bool, classes):
    n = 0
    correct = 0
    accepted = 0
    accepted_correct = 0
    skipped = 0
    latencies = []
    per_command = Counter()
    per_command_total = Counter()
    confusion = Counter()
    for path in files:
        audio, sr = load_wav(path)
        if use_vad:
            vad = EnergyVad(fs=sr)
            utterances = vad.feed(audio) + vad.flush()
            if not utterances:
                skipped += 1
                continue
            audio = utterances[0]
        result = engine.process_utterance(audio, sr)
        truth = label_from_name(path.name, classes)
        n += 1
        latencies.append(result.latency_ms)
        per_command_total[truth] += 1
        confusion[(truth, result.command)] += 1
        hit = int(result.command == truth)
        correct += hit
        accepted += int(result.accepted)
        accepted_correct += int(result.accepted and result.command == truth)
        if hit:
            per_command[truth] += 1
    return {
        'n': n,
        'skipped': skipped,
        'argmax_accuracy': round(correct / n, 4) if n else None,
        'acceptance_rate': round(accepted / n, 4) if n else None,
        'accepted_accuracy': round(accepted_correct / accepted, 4) if accepted else None,
        'per_command_argmax_accuracy': {
            c: round(per_command[c] / per_command_total[c], 4)
            for c in sorted(per_command_total)
        },
        'per_command_n': dict(sorted(per_command_total.items())),
        'confusion': {
            truth: {pred: confusion[(truth, pred)] for pred in sorted(classes)
                    if confusion[(truth, pred)]}
            for truth in sorted(per_command_total)
        },
        'mean_latency_ms': round(sum(latencies) / len(latencies), 1) if latencies else None,
    }


def files_from_manifest(manifest: Path, subset: str):
    """按 manifest 的 split 列取文件，保证评估落在没被训练过的子集上。"""
    out = []
    with manifest.open(encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            if subset and row.get('split') != subset:
                continue
            path = Path(row['audio_path'])
            if path.exists():
                out.append(path)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--known', default=None, help='已知命令 WAV 目录')
    parser.add_argument('--manifest', default=None, help='语料 manifest（含 split 列）')
    parser.add_argument('--subset', default='test', help='manifest 中的子集名，如 test')
    parser.add_argument('--unknown', default=None)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--model', default=None, help='模型工件路径，默认用 runtime 里的部署模型')
    parser.add_argument('--thresholds', default=None)
    parser.add_argument('--out', default=None)
    # 默认关闭：本脚本产出的 docs/pipeline_eval.json 是「部署单阈值 0.945」口径的
    # 历史工件，被测试报告直接引用。静默换成双信号会让那份报告里的数字对不上来源。
    parser.add_argument('--selective', action='store_true',
                        help='用双信号 AND 规则评估（阈值与参照池来自 gate 文件）')
    parser.add_argument('--gate', default=None, help='双信号规则文件路径')
    parser.add_argument('--pool', default=None, help='kNN 参照池 npz 路径')
    args = parser.parse_args()

    if args.model:
        engine = DialectEngine.load(args.model, args.thresholds, threshold=args.threshold,
                                    gate_path=args.gate, pool_path=args.pool,
                                    selective=args.selective)
    else:
        engine = DialectEngine.load(threshold=args.threshold,
                                    gate_path=args.gate, pool_path=args.pool,
                                    selective=args.selective)
    classes = engine.classes_

    if args.manifest:
        known_files = files_from_manifest(Path(args.manifest), args.subset)
    elif args.known:
        known_files = sorted(Path(args.known).rglob('*.wav'))
    else:
        sys.exit('需要 --manifest 或 --known 之一')
    unknown_files = sorted(Path(args.unknown).rglob('*.wav')) if args.unknown else []
    if args.limit:
        known_files = known_files[:args.limit]
        unknown_files = unknown_files[:args.limit]
    if not known_files:
        sys.exit(f'没有找到 WAV: {args.manifest or args.known}')

    report = {
        'threshold': engine.threshold,
        # 判定口径必须写进工件：同一份 pipeline_eval.json 在单阈值与双信号下
        # 数字完全不同，不标注就等于两个来源共用一个文件名。
        'rejection_rule': (engine.gate.as_dict() if engine.gate is not None
                           else {'kind': 'single_threshold',
                                 'threshold': round(float(engine.threshold), 6)}),
        'classes': classes,
        'source': {
            'manifest': args.manifest,
            'subset': args.subset if args.manifest else None,
            'known_dir': args.known,
            'unknown_dir': args.unknown,
        },
        'variants': {},
    }
    for use_vad in (True, False):
        key = 'with_vad' if use_vad else 'no_vad'
        entry = {'known': evaluate(engine, known_files, use_vad, classes)}
        if unknown_files:
            unk = evaluate(engine, unknown_files, use_vad, classes)
            entry['unknown'] = {
                'n': unk['n'],
                'false_accept_rate': round(unk['acceptance_rate'], 4)
                if unk['acceptance_rate'] is not None else None,
                'mean_latency_ms': unk['mean_latency_ms'],
            }
        report['variants'][key] = entry

    report['note'] = '语料为 edge-tts 合成粤语，非真人语音；结论不外推为真人准确率。'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding='utf-8')
        print('已写入:', args.out)


if __name__ == '__main__':
    main()

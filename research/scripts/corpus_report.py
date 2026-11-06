# -*- coding: utf-8 -*-
"""语料统计、重复检查。"""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.manifest import load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/tts_corpus/manifest.csv')
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    print('total:', len(records))
    print('by language:', dict(Counter(r.language for r in records)))
    print('by source:', dict(Counter(r.source for r in records)))
    print('by speaker:', dict(Counter(r.speaker_id for r in records)))
    print('by command:', dict(Counter(r.command for r in records)))
    print('by expression:', dict(Counter(r.expression_type for r in records)))
    keys = [(r.text, r.speaker_id, r.source) for r in records]
    dup = [k for k, v in Counter(keys).items() if v > 1]
    print('duplicate_text_speaker_source_groups:', len(dup))
    if dup:
        for k in dup[:10]:
            print('DUP:', k)


if __name__ == '__main__':
    main()

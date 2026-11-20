# -*- coding: utf-8 -*-
"""从 DialectVoice_Research 同步推理运行时到本项目 runtime/。

runtime/ 必须自包含（要能整体拷到树莓派），所以这里复制纯 numpy 推理所需
的模块与模型工件，并把 Research 里的 `from src.X import` 改写为包内相对导入。

重复运行是幂等的；复制后用 sha256 校验模型工件，避免手工替换出错。
"""
import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ENGINEERING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH_ROOT = ENGINEERING_ROOT.parent / 'DialectVoice_Research'

# (源文件相对 Research 根, 目标文件相对 runtime/)
SOURCE_FILES = [
    ('src/features.py', 'features.py'),
    ('src/features_v3.py', 'features_v3.py'),
    ('src/v3_cnn_numpy.py', 'v3_cnn_numpy.py'),
    ('src/v3_model.py', 'v3_model.py'),
    ('src/selective.py', 'selective.py'),
    ('src/audio_io.py', 'audio_io.py'),
    ('src/serial_protocol.py', 'serial_protocol.py'),
    # S4(b)/S5 两遍识别与决策层：流式预触发、整句复核、状态源、决策、对话
    ('src/streaming.py', 'streaming.py'),
    ('src/verify.py', 'verify.py'),
    ('src/state.py', 'state.py'),
    ('src/decision.py', 'decision.py'),
    ('src/dialog.py', 'dialog.py'),
]

MODEL_FILES = [
    # 未裁剪基线模型：用于「VAD 前 / 后」消融对比
    ('data/models/v3_cnn_ensemble.npz', 'models/v3_cnn_ensemble.npz'),
    ('data/models/v3_thresholds.json', 'models/v3_thresholds.json'),
    # 部署模型：在 VAD 裁剪后的同源语料上重训练，与线上输入分布一致
    ('data/models/v3_cnn_ensemble_vad.npz', 'models/v3_cnn_ensemble_vad.npz'),
    ('data/models/v3_thresholds_vad.json', 'models/v3_thresholds_vad.json'),
    # S4 双信号拒识：规则 JSON 与 kNN 参照池是**一对**工件，必须同时同步。
    # 池由 scripts/build_selective_gate.py 生成（阈值已按该池重标定），
    # 单独替换任何一个都会让线上判定与实测 FAR 脱钩。
    ('data/models/v3_selective_gate.json', 'models/v3_selective_gate.json'),
    ('data/models/v3_knn_pool.npz', 'models/v3_knn_pool.npz'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rewrite_imports(text):
    """把 Research 的 `from src.X import` 改成 runtime 包内相对导入。"""
    return re.sub(r'from src\.([A-Za-z_][A-Za-z0-9_]*) import', r'from .\1 import', text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--research', default=str(DEFAULT_RESEARCH_ROOT))
    parser.add_argument('--runtime', default=str(ENGINEERING_ROOT / 'runtime'))
    parser.add_argument('--check', action='store_true',
                        help='只校验不同步，来源缺失或哈希不符时退出码非零')
    args = parser.parse_args()

    research = Path(args.research).resolve()
    runtime = Path(args.runtime).resolve()
    if not research.is_dir():
        sys.exit(f'找不到 Research 目录: {research}')

    (runtime / 'models').mkdir(parents=True, exist_ok=True)
    (runtime / '__init__.py').write_text(
        '# -*- coding: utf-8 -*-\n"""自包含推理运行时（由 scripts/sync_runtime.py 生成）。"""\n',
        encoding='utf-8',
    )

    problems = []
    for src_rel, dst_rel in SOURCE_FILES:
        src = research / src_rel
        dst = runtime / dst_rel
        if not src.exists():
            problems.append(f'来源缺失: {src}')
            continue
        text = rewrite_imports(src.read_text(encoding='utf-8'))
        if args.check:
            if not dst.exists() or dst.read_text(encoding='utf-8') != text:
                problems.append(f'内容不一致: {dst}')
            continue
        dst.write_text(text, encoding='utf-8')
        print(f'同步 {dst_rel}')

    for src_rel, dst_rel in MODEL_FILES:
        src = research / src_rel
        dst = runtime / dst_rel
        if not src.exists():
            problems.append(f'来源缺失: {src}')
            continue
        if args.check:
            if not dst.exists() or sha256(src) != sha256(dst):
                problems.append(f'哈希不一致: {dst}')
            continue
        shutil.copy2(src, dst)
        print(f'同步 {dst_rel}  sha256={sha256(dst)[:16]}')

    if problems:
        for p in problems:
            print('问题:', p)
        sys.exit(1)
    print('runtime 就绪:', runtime)


if __name__ == '__main__':
    main()

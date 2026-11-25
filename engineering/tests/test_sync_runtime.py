# -*- coding: utf-8 -*-
"""sync_runtime.py 测试。

它是 runtime/ 正确性的唯一守卫，此前无测试。重点验证三件事：
1. 导入改写只动 ``from src.X``，不误伤同前缀模块；
2. ``--check`` 是只读的（此前会顺手建目录并覆盖 ``__init__.py``）；
3. 内容/哈希/来源缺失都能让退出码非零。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sync_runtime


# ----------------------------------------------------------------------
# 纯函数
# ----------------------------------------------------------------------
def test_rewrite_imports_converts_src_package_imports():
    text = 'from src.features import extract\nfrom src.v3_model import V3Predictor\n'
    assert sync_runtime.rewrite_imports(text) == (
        'from .features import extract\nfrom .v3_model import V3Predictor\n')


def test_rewrite_imports_keeps_other_imports_intact():
    text = ('import src.features\n'
            'from srcx import thing\n'
            'from .local import y\n')
    assert sync_runtime.rewrite_imports(text) == text


def test_rewrite_imports_handles_multiline_and_indent():
    text = 'def f():\n    from src.dialog import COMMAND_ZH\n'
    assert sync_runtime.rewrite_imports(text) == (
        'def f():\n    from .dialog import COMMAND_ZH\n')


def test_sha256_matches_hashlib(tmp_path):
    payload = b'x' * (2 << 20)
    path = tmp_path / 'blob.bin'
    path.write_bytes(payload)
    assert sync_runtime.sha256(path) == hashlib.sha256(payload).hexdigest()


# ----------------------------------------------------------------------
# main()：用最小的自定义清单，不碰真实 Research 目录
# ----------------------------------------------------------------------
@pytest.fixture
def layout(tmp_path, monkeypatch):
    research = tmp_path / 'Research'
    runtime = tmp_path / 'runtime'
    (research / 'src').mkdir(parents=True)
    (research / 'data' / 'models').mkdir(parents=True)
    (research / 'src' / 'a.py').write_text('from src.b import x\n', encoding='utf-8')
    (research / 'data' / 'models' / 'm.npz').write_bytes(b'\x00\x01\x02')
    monkeypatch.setattr(sync_runtime, 'SOURCE_FILES', [('src/a.py', 'a.py')])
    monkeypatch.setattr(sync_runtime, 'MODEL_FILES',
                        [('data/models/m.npz', 'models/m.npz')])
    return research, runtime


def run_main(monkeypatch, research, runtime, check=False):
    argv = ['sync_runtime.py', '--research', str(research), '--runtime', str(runtime)]
    if check:
        argv.append('--check')
    monkeypatch.setattr(sys, 'argv', argv)
    sync_runtime.main()


def test_sync_copies_rewritten_source_and_model(layout, monkeypatch):
    research, runtime = layout
    run_main(monkeypatch, research, runtime)
    assert (runtime / 'a.py').read_text(encoding='utf-8') == 'from .b import x\n'
    assert (runtime / 'models' / 'm.npz').read_bytes() == b'\x00\x01\x02'
    assert (runtime / '__init__.py').exists()


def test_check_passes_after_sync(layout, monkeypatch, capsys):
    research, runtime = layout
    run_main(monkeypatch, research, runtime)
    run_main(monkeypatch, research, runtime, check=True)
    assert 'runtime 就绪' in capsys.readouterr().out


def test_check_is_read_only(layout, monkeypatch):
    """回归：--check 此前会建目录并覆盖 __init__.py，属于只读校验的副作用。"""
    research, runtime = layout
    run_main(monkeypatch, research, runtime)
    init_before = (runtime / '__init__.py').read_text(encoding='utf-8')
    (runtime / 'a.py').write_text('hand edited\n', encoding='utf-8')

    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, research, runtime, check=True)

    assert exc.value.code == 1
    # 不一致时不得把手工改动覆盖回去
    assert (runtime / 'a.py').read_text(encoding='utf-8') == 'hand edited\n'
    assert (runtime / '__init__.py').read_text(encoding='utf-8') == init_before


def test_check_does_not_create_missing_runtime_dir(layout, monkeypatch):
    research, runtime = layout
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, research, runtime, check=True)
    assert exc.value.code == 1
    assert not runtime.exists()


def test_check_detects_hash_mismatch(layout, monkeypatch, capsys):
    research, runtime = layout
    run_main(monkeypatch, research, runtime)
    (runtime / 'models' / 'm.npz').write_bytes(b'\xff\xfe')
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, research, runtime, check=True)
    assert exc.value.code == 1
    assert '哈希不一致' in capsys.readouterr().out


def test_missing_source_exits_nonzero(layout, monkeypatch, capsys):
    research, runtime = layout
    (research / 'src' / 'a.py').unlink()
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, research, runtime)
    assert exc.value.code == 1
    assert '来源缺失' in capsys.readouterr().out


def test_missing_research_dir_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [
        'sync_runtime.py', '--research', str(tmp_path / 'nope'),
        '--runtime', str(tmp_path / 'runtime')])
    with pytest.raises(SystemExit):
        sync_runtime.main()

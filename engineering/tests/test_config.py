# -*- coding: utf-8 -*-
"""配置加载与 CLI 覆盖测试。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_points_at_deployed_artifacts():
    cfg = load_config()
    assert cfg.model_path.exists(), cfg.model_path
    assert cfg.thresholds_path.exists(), cfg.thresholds_path
    assert cfg.threshold == pytest.approx(0.945)
    assert cfg.sample_rate == 16000


def test_relative_paths_resolve_against_project_root(tmp_path):
    cfg = load_config()
    assert cfg.model_path.is_absolute()
    assert cfg.broadcast_dir.is_absolute()
    assert cfg.log_path.parent.name == 'logs'
    partial = tmp_path / 'partial.yaml'
    partial.write_text('recognition:\n  confidence_threshold: 0.5\n', encoding='utf-8')
    assert load_config(partial).model_path == cfg.model_path


def test_missing_config_falls_back_to_defaults(tmp_path):
    cfg = load_config(tmp_path / 'nope.yaml')
    assert cfg.model_path.name == 'v3_cnn_ensemble_vad.npz'
    assert cfg.threshold is None


def test_cli_defaults_follow_config(tmp_path):
    from scripts.run_replay import build_parser
    baseline = ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble.npz'
    yml = tmp_path / 'alt.yaml'
    yml.write_text(
        f"""
audio:
  samplerate: 8000
recognition:
  model: {baseline}
  thresholds: {ROOT / 'runtime' / 'models' / 'v3_thresholds.json'}
  confidence_threshold: 0.33
control:
  executor: mock_esp32
""", encoding='utf-8')
    cfg = load_config(yml)
    args = build_parser(cfg).parse_args(['--audio', 'x.wav'])
    assert args.threshold == pytest.approx(0.33)
    assert args.fs == 8000
    assert args.executor == 'mock_esp32'
    assert args.model == str(baseline)


def test_cli_overrides_config(tmp_path):
    from scripts.run_replay import build_parser
    cfg = load_config()
    args = build_parser(cfg).parse_args(['--audio', 'x.wav', '--threshold', '0.7'])
    assert args.threshold == pytest.approx(0.7)


def test_vad_kwargs_from_config(tmp_path):
    yml = tmp_path / 'vad.yaml'
    yml.write_text('vad:\n  frame_ms: 20\n  energy_margin_db: 12\n', encoding='utf-8')
    cfg = load_config(yml)
    assert cfg.vad_kwargs() == {'frame_ms': 20, 'energy_margin_db': 12}


def test_monitor_parser_follows_config(tmp_path):
    from scripts.run_monitor import build_parser
    cfg = load_config()
    args = build_parser(cfg).parse_args([])
    assert args.threshold == pytest.approx(cfg.threshold)
    assert args.model == str(cfg.model_path)

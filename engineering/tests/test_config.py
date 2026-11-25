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
    assert cfg.fast_silence_ms == pytest.approx(240.0)
    assert cfg.red_soft_enabled is True
    assert cfg.force_confirm_commands == []
    assert cfg.high_risk_min_ood['ALL_OFF'] == pytest.approx(-0.019)
    assert ['ALL_OFF', 'LIGHT_OFF'] in cfg.confirmation_pairs


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
    assert cfg.model_path.name == 'v3_cnn_ensemble_v32b.npz'
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


# ---------------------------------------------------------------------------
# 麦克风入口的口径守卫。
# 本次审查发现的缺陷：run_monitor 调 run_stream 时不传 two_pass（默认 False），
# 也不传 state_source / 决策参数 —— 于是现场跑的是 S4 之前的单遍路径，
# config 的 streaming.pretrigger 与 state_source 全部失效，
# 报告第 6 节的两遍指标在演示里复现不出来。下面几条把它锁死。
# ---------------------------------------------------------------------------
def test_monitor_defaults_to_two_pass():
    from scripts.run_monitor import build_parser
    cfg = load_config()
    assert cfg.streaming_enabled is True, '出厂 config 应启用流式两遍'
    args = build_parser(cfg).parse_args([])
    assert args.two_pass is True


def test_monitor_pass_switch_overrides_config():
    from scripts.run_monitor import build_parser
    cfg = load_config()
    assert build_parser(cfg).parse_args(['--single-pass']).two_pass is False
    assert build_parser(cfg).parse_args(['--two-pass']).two_pass is True
    with pytest.raises(SystemExit):
        build_parser(cfg).parse_args(['--two-pass', '--single-pass'])


def test_monitor_mock_relay_can_be_disabled():
    """原来 --mock-relay 是 store_true + default=True，真实继电器永远起不来。"""
    from scripts.run_monitor import build_parser
    cfg = load_config()
    assert build_parser(cfg).parse_args([]).mock_relay is True, '安全默认仍是 mock'
    assert build_parser(cfg).parse_args(['--no-mock-relay']).mock_relay is False


def test_monitor_main_wires_deployed_path(monkeypatch):
    """main() 必须把两遍开关、状态先验与决策参数真的传进引擎。"""
    import scripts.run_monitor as rm
    from runtime.state import MockStateSource

    seen = {}

    class FakeEngine:
        gate = None
        threshold = 0.945

        @classmethod
        def load(cls, *a, **kw):
            seen['load_kwargs'] = kw
            return cls()

        def run_stream(self, capture, fs=16000, chunk_frames=1024, two_pass=False):
            seen['two_pass'] = two_pass
            return iter(())

    class FakeCapture:
        def __init__(self, **kw):
            seen['capture_kwargs'] = kw

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(rm, 'DialectEngine', FakeEngine)
    monkeypatch.setattr(rm, 'build_executor', lambda args: object())
    monkeypatch.setattr('src.audio_capture.AudioCapture', FakeCapture)

    rm.main(['--no-speak', '--executor', 'log'])

    kw = seen['load_kwargs']
    assert seen['two_pass'] is True, '麦克风入口必须走两遍流式路径'
    assert isinstance(kw['state_source'], MockStateSource), '状态先验未接线'
    cfg = load_config()
    # 决策参数必须来自 config，而不是引擎默认值（改 config 要能生效）
    assert kw['decision_costs'] == cfg.decision_costs
    assert kw['debounce_ms'] == cfg.debounce_ms
    assert kw['max_clarify'] == cfg.max_clarify
    assert kw['consistency_margin'] == pytest.approx(cfg.consistency_margin)
    assert kw['confusable_pairs'] == cfg.confusable_pairs
    assert kw['broadcast_command'] == cfg.broadcast_command
    assert kw['fast_silence_ms'] == pytest.approx(cfg.fast_silence_ms)
    assert kw['fast_endpoint_threshold'] == pytest.approx(cfg.fast_endpoint_threshold)
    assert kw['red_soft_enabled'] == cfg.red_soft_enabled
    assert kw['red_soft_tolerances'] == cfg.red_soft_tolerances
    assert kw['force_confirm_commands'] == cfg.force_confirm_commands
    assert kw['confirmation_pairs'] == cfg.confirmation_pairs
    assert kw['high_risk_min_ood'] == cfg.high_risk_min_ood


def test_monitor_single_pass_flag_reaches_run_stream(monkeypatch):
    import scripts.run_monitor as rm

    seen = {}

    class FakeEngine:
        gate = None
        threshold = 0.945

        @classmethod
        def load(cls, *a, **kw):
            return cls()

        def run_stream(self, capture, fs=16000, chunk_frames=1024, two_pass=False):
            seen['two_pass'] = two_pass
            return iter(())

    class FakeCapture:
        def __init__(self, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(rm, 'DialectEngine', FakeEngine)
    monkeypatch.setattr(rm, 'build_executor', lambda args: object())
    monkeypatch.setattr('src.audio_capture.AudioCapture', FakeCapture)
    rm.main(['--no-speak', '--executor', 'log', '--single-pass'])
    assert seen['two_pass'] is False

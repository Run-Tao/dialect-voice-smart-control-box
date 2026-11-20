# -*- coding: utf-8 -*-
"""统一配置加载：config.yaml 是唯一配置源。

CLI 参数优先于配置；配置里的相对路径一律按项目根目录解析，
这样 `--config` 指向任何位置的局部覆盖文件都不会把相对路径带偏。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

ENGINEERING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ENGINEERING_ROOT / 'config.yaml'

_VAD_KEYS = ('frame_ms', 'energy_margin_db', 'min_speech_ms',
             'min_silence_ms', 'pre_roll_ms', 'max_utterance_ms')


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)
    root: Path = ENGINEERING_ROOT

    def get(self, dotted: str, default=None):
        node: Any = self.raw
        for part in dotted.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str, default=None) -> Optional[Path]:
        value = self.get(dotted, default)
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def model_path(self) -> Path:
        return self.path('recognition.model', 'runtime/models/v3_cnn_ensemble_vad.npz')

    @property
    def thresholds_path(self) -> Path:
        return self.path('recognition.thresholds', 'runtime/models/v3_thresholds_vad.json')

    @property
    def threshold(self) -> Optional[float]:
        value = self.get('recognition.confidence_threshold')
        return None if value is None else float(value)

    # ------------------------------------------------------------------
    # S4 双信号拒识（maxprob AND ood）。默认**关闭**：没有显式写
    # recognition.selective.enabled: true 的配置文件行为与 S3 之前完全一致，
    # 这样旧配置、旧测试、旧日志都不会因为这次改动而悄悄换判定口径。
    @property
    def selective_enabled(self) -> bool:
        return bool(self.get('recognition.selective.enabled', False))

    @property
    def selective_gate_path(self) -> Optional[Path]:
        return self.path('recognition.selective.gate',
                         'runtime/models/v3_selective_gate.json')

    @property
    def selective_pool_path(self) -> Optional[Path]:
        # 缺省 None：由规则文件里的 ood.pool + 规则文件所在目录推出，
        # 避免同一个池路径在两个地方各写一遍然后不一致。
        return self.path('recognition.selective.pool', None)

    @property
    def selective_fallback_mode(self) -> str:
        mode = str(self.get('recognition.selective.fallback_mode', 'single'))
        if mode not in ('single', 'reject'):
            raise ValueError(
                f"recognition.selective.fallback_mode 只能是 'single' 或 'reject'，收到 {mode!r}")
        return mode

    @property
    def sample_rate(self) -> int:
        return int(self.get('audio.samplerate', 16000))

    # ------------------------------------------------------------------
    # S5：状态源。默认 mock —— 课程环境里 HA/MQTT 未必能部署，
    # 用 mock 状态机跑通闭环，结论标注为「仿真状态源」（方案 §11 R-9）。
    @property
    def state_source_kind(self) -> str:
        return str(self.get('state_source', 'mock') or 'mock')

    @property
    def state_source_options(self) -> Dict[str, Any]:
        """状态源构造参数（url/token/entities/host/topics…）。"""
        opts = self.get('state_source_options', {}) or {}
        return dict(opts) if isinstance(opts, dict) else {}

    # ------------------------------------------------------------------
    # S5：期望代价三态决策。默认与 runtime/decision.py 的 DecisionCosts 一致，
    # 唯一改动是 cost_reject：方案 §8.3 的草稿写 1.0，但「拒识」意味着用户
    # 要把整句重说一遍，比「二选一澄清」贵，所以这里取 4.0。
    # 1.0 会让 E_clarify(=1.6) 永远大于 E_reject(=1.0)，澄清机制根本触发不了。
    @property
    def decision_costs(self) -> Dict[str, float]:
        d = self.get('decision', {}) or {}
        out = {
            'act_wrong': float(d.get('cost_act_wrong', 10.0)),
            'reject': float(d.get('cost_reject', 4.0)),
            'clarify': float(d.get('cost_clarify', 1.5)),
        }
        if 'retry_success' in d:
            out['retry_success'] = float(d['retry_success'])
        if 'yellow_reliability' in d:
            out['yellow_reliability'] = float(d['yellow_reliability'])
        return out

    @property
    def prior_temperature(self) -> float:
        return float(self.get('decision.prior_temperature', 0.3))

    @property
    def competitor_floor(self) -> float:
        return float(self.get('decision.competitor_floor', 0.005))

    @property
    def prior_override_margin(self) -> float:
        return float(self.get('decision.prior_override_margin', 0.15))

    @property
    def consistency_margin(self) -> float:
        """两遍不一致时，Pass-2 要有多大的 margin 才仍然算「分歧」。

        Pass-1 的 1 s 滑窗看的是半句话，实测常把「全部關晒」判成 LIGHT_OFF，
        而 Pass-2 整句后验 margin 高达 0.998。margin 低于这个值才降级。
        """
        return float(self.get('decision.consistency_margin', 0.15))

    @property
    def redundancy_weight(self) -> float:
        return float(self.get('decision.redundancy_weight', 0.25))

    @property
    def confusable_pairs(self):
        pairs = self.get('decision.confusable_pairs')
        if not pairs:
            return None
        return [list(p) for p in pairs]

    @property
    def debounce_ms(self) -> int:
        return int(self.get('decision.debounce_ms', 2000))

    @property
    def max_clarify(self) -> int:
        return int(self.get('decision.max_clarify', 1))

    # ------------------------------------------------------------------
    # S4(b)：流式两遍识别的参数。
    @property
    def streaming_enabled(self) -> bool:
        return bool(self.get('streaming.pretrigger', True))

    @property
    def streaming_window_ms(self) -> float:
        return float(self.get('streaming.window_ms', 1000.0))

    @property
    def streaming_hop_ms(self) -> float:
        return float(self.get('streaming.hop_ms', 100.0))

    @property
    def pretrigger_threshold(self) -> float:
        return float(self.get('streaming.pretrigger_threshold', 0.5))

    @property
    def fast_silence_ms(self) -> float:
        """预触发之后的静音挂起；压得越短，动作延迟越低。"""
        return float(self.get('streaming.fast_silence_ms', 120.0))

    @property
    def fast_endpoint_threshold(self) -> float:
        """快速端点的单窗后验门槛。

        与 ``pretrigger_threshold`` 分开：预触发要连续两窗一致（给用户看的
        反馈不能抖），快速端点只要一个窗（延迟预算不能等两窗）。绑在一起
        时短命令结构上等不到预触发，静音挂起压在默认 400 ms，
        「说完 -> 动作」进不了 300 ms 预算。
        """
        return float(self.get('streaming.fast_endpoint_threshold',
                              self.pretrigger_threshold))

    @property
    def keyword_bias(self) -> float:
        return float(self.get('streaming.keyword_bias', 0.3))

    @property
    def executor(self) -> str:
        return str(self.get('control.executor', 'log'))

    @property
    def broadcast_command(self) -> bool:
        """两遍路径是否**额外**播一遍普通话命令。

        默认 False：执行器已经把动作做掉了，再播一遍会让智能音箱重复执行。
        只有「靠普通话播报驱动外部音箱」的场景才需要打开。
        """
        return bool(self.get('control.broadcast_command', False))

    @property
    def relay_pin(self) -> int:
        return int(self.get('control.relay_pin', 17))

    @property
    def wake_word(self) -> str:
        return str(self.get('broadcast.wake_word', '') or '')

    @property
    def broadcast_dir(self) -> Path:
        return self.path('broadcast.audio_dir', 'data/audio/mandarin')

    @property
    def log_path(self) -> Path:
        return self.root / 'logs' / 'control_log.jsonl'

    def vad_kwargs(self) -> Dict[str, Any]:
        vad = self.raw.get('vad') or {}
        return {k: vad[k] for k in _VAD_KEYS if k in vad}


def load_config(path=None) -> Config:
    """读取配置文件；文件不存在时返回内置默认值，保证离线可跑。

    相对路径始终相对项目根目录解析，与配置文件所在位置无关。
    """
    p = Path(path).resolve() if path else DEFAULT_CONFIG
    if not p.exists():
        return Config(raw={})
    import yaml
    data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    return Config(raw=data)

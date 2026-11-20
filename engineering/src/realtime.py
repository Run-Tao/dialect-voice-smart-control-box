# -*- coding: utf-8 -*-
"""实时识别管线：VAD 切段 -> 特征 -> 分类 -> 拒识 -> 执行 -> 普通话播报。

S4(b)/S5 之后这里是**两遍**管线：

    Pass-1（流式滑窗 runtime/streaming.py） -> 预触发 + 候选
    Pass-2（整句复核 runtime/verify.py）    -> 关键词偏置 + 两遍一致性
    决策层（runtime/decision.py）          -> 状态先验 + 期望代价三态
    执行层（src/control.py）               -> dispatch / wait_ack（ACK 闭环）
    对话层（runtime/dialog.py）            -> 去抖 + 澄清

三条入口并存，语义各不相同，**不能互相替代**：

- ``process_utterance``    单遍，行为与 S4 之前逐字一致。离线批量评估
  （``scripts/evaluate_pipeline.py``、``scripts/eval_real_far.py``）与既有测试
  走这条；S0/S3/S4 公布的所有指标都是这条路径的口径，改它等于让历史数字失效。
- ``process_segment``      两遍。候选已知（``candidate`` 可以是 None）。
- ``process_stream_block`` 流式：VAD 切段 + Pass-1 预触发 + 两遍复核。

动作优先与异步播报
------------------
「动作 P95 ≤ 300 ms」量的是**用户说完到动作下发**这一段，不是从开口算起。
做法：Pass-1 预触发一响，就把 VAD 的静音挂起从 400 ms 压到 ``fast_silence_ms``
（默认 120 ms），语音一停就切段复核下发；播报放到后台线程，不占用动作时间。
"""
from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .vad import EnergyVad

ENGINEERING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ENGINEERING_ROOT / 'runtime' / 'models' / 'v3_cnn_ensemble_vad.npz'
DEFAULT_THRESHOLDS = ENGINEERING_ROOT / 'runtime' / 'models' / 'v3_thresholds_vad.json'
DEFAULT_GATE = ENGINEERING_ROOT / 'runtime' / 'models' / 'v3_selective_gate.json'


def _runtime(name: str):
    """按名字取 runtime 模块。runtime/ 必须能整体拷到端侧，所以只在用到时导入。"""
    root = str(ENGINEERING_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f'runtime.{name}')


@dataclass
class UtteranceResult:
    command: str
    confidence: float
    accepted: bool
    scores: Dict[str, float] = field(default_factory=dict)
    # S4(b)：**无偏**后验（关键词偏置被解析还原后的结果）。
    # scores 是带偏置的，决策层看它；门限看 acoustic_scores；
    # 评估「模型本身的 argmax」时也必须看这一份，否则关键词偏置会污染基线。
    acoustic_scores: Dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    duration_sec: float = 0.0
    latency_ms: float = 0.0
    reason: str = ''
    executed: bool = False
    spoken: bool = False
    # S4：这一条究竟走的是双信号规则还是降级路径。必须落进日志——
    # 降级不是错误，但看不见的降级等于把 FAR 悄悄退回 0.3110 那一档。
    rule: str = ''
    signals: Dict[str, float] = field(default_factory=dict)
    # --- S4(b)/S5：三态决策与两遍一致性 ---
    action: str = ''              # execute / clarify / reject（单遍路径为空）
    zone: str = ''                # green / yellow / red
    gate_accepted: bool = False   # 门限是否放行（与最终动作分开记）
    pass1_command: str = ''       # Pass-1 给出的候选
    consistency: str = ''         # consistent / inconsistent / no_candidate
    expected_costs: Dict[str, float] = field(default_factory=dict)
    debounced: bool = False       # 被去抖窗口抑制，没有执行
    action_latency_ms: float = 0.0  # 复核开始 -> 动作下发完成
    hangover_ms: float = 0.0      # 切段时的静音挂起（用户说完到切段）
    pretrigger_ms: float = 0.0    # Pass-1 预触发处理耗时
    clarify: Dict[str, object] = field(default_factory=dict)
    ack: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            'command': self.command,
            'confidence': round(float(self.confidence), 4),
            'accepted': bool(self.accepted),
            'margin': round(float(self.margin), 4),
            'duration_sec': round(float(self.duration_sec), 3),
            'latency_ms': round(float(self.latency_ms), 1),
            'reason': self.reason,
            'executed': bool(self.executed),
            'spoken': bool(self.spoken),
            'rule': self.rule,
            'scores': {k: round(float(v), 4) for k, v in self.scores.items()},
            'acoustic_scores': {k: round(float(v), 4)
                                for k, v in self.acoustic_scores.items()},
            'signals': {k: round(float(v), 6) for k, v in self.signals.items()
                        if v is not None},
            'action': self.action,
            'zone': self.zone,
            'gate_accepted': bool(self.gate_accepted),
            'pass1_command': self.pass1_command,
            'consistency': self.consistency,
            'expected_costs': {k: round(float(v), 4)
                               for k, v in self.expected_costs.items()},
            'debounced': bool(self.debounced),
            'action_latency_ms': round(float(self.action_latency_ms), 1),
            'hangover_ms': round(float(self.hangover_ms), 1),
            'pretrigger_ms': round(float(self.pretrigger_ms), 1),
            'clarify': dict(self.clarify),
            'ack': {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                    for k, v in self.ack.items()},
        }


class DialectEngine:
    """把一段方言语音转成控制动作。"""

    def __init__(self, predictor, executor=None, speaker=None,
                 vad: Optional[EnergyVad] = None,
                 threshold: Optional[float] = None,
                 gate=None,
                 on_result: Optional[Callable[[UtteranceResult], None]] = None,
                 log_path: Optional[str] = None,
                 # --- S4(b)/S5 ---
                 state_source=None,
                 decision_costs=None,
                 confusable_pairs=None,
                 competitor_floor: float = 0.005,
                 prior_temperature: float = 0.3,
                 redundancy_weight: float = 0.25,
                 prior_override_margin: float = 0.15,
                 consistency_margin: float = 0.15,
                 debounce_ms: int = 2000,
                 max_clarify: int = 1,
                 window_ms: float = 1000.0,
                 hop_ms: float = 100.0,
                 pretrigger_threshold: float = 0.5,
                 fast_silence_ms: float = 120.0,
                 fast_endpoint_threshold: Optional[float] = None,
                 bias: float = 0.3,
                 broadcast_async: bool = True,
                 broadcast_command: bool = False,
                 ack_timeout: float = 1.0,
                 on_pretrigger=None):
        self.predictor = predictor
        self.executor = executor
        self.speaker = speaker
        self.vad = vad if vad is not None else EnergyVad(fs=16000)
        self.threshold = float(threshold if threshold is not None
                              else getattr(predictor, 'threshold', 0.0))
        # gate 为 None => 单阈值路径，与 S3 之前逐字一致
        self.gate = gate
        self.on_result = on_result
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.state_source = state_source
        self.competitor_floor = float(competitor_floor)
        self.prior_temperature = float(prior_temperature)
        self.redundancy_weight = float(redundancy_weight)
        self.prior_override_margin = float(prior_override_margin)
        self.consistency_margin = float(consistency_margin)
        self.window_ms = float(window_ms)
        self.hop_ms = float(hop_ms)
        self.pretrigger_threshold = float(pretrigger_threshold)
        self.fast_silence_ms = float(fast_silence_ms)
        # 快速端点的门槛与预触发**分开**：预触发要连续两窗一致，
        # 快速端点只要一个窗后验够高。绑在一起时短命令永远等不到预触发，
        # 静音挂起压在默认 390 ms，「说完 -> 动作」进不了 300 ms 预算。
        self.fast_endpoint_threshold = float(
            pretrigger_threshold if fast_endpoint_threshold is None
            else fast_endpoint_threshold)
        self.bias = float(bias)
        self.broadcast_async = bool(broadcast_async)
        # 两遍路径里执行器**已经**把动作做掉了，再播一遍普通话命令会让
        # 智能音箱把同一件事再做一次。所以默认只播确认语（已完成/失败）。
        # 单遍路径语义不同：那里的普通话播报就是执行本身，保持原样。
        self.broadcast_command = bool(broadcast_command)
        self.ack_timeout = float(ack_timeout)
        self.on_pretrigger = on_pretrigger

        decision_mod = _runtime('decision')
        dialog_mod = _runtime('dialog')
        if isinstance(decision_costs, dict):
            decision_costs = decision_mod.DecisionCosts.from_dict(decision_costs)
        self.decision_costs = decision_costs or decision_mod.DecisionCosts()
        self.confusable_pairs = (list(confusable_pairs) if confusable_pairs is not None
                                 else list(decision_mod.CONFUSABLE_PAIRS))
        self.debouncer = dialog_mod.Debouncer(window_ms=int(debounce_ms))
        self.dialog = dialog_mod.ClarifyDialog(max_clarify=int(max_clarify))

        self._verifier = None
        self._pass1 = None
        self._candidate: Optional[str] = None
        self._pretrigger_ms = 0.0
        self._default_min_silence_frames = self.vad.min_silence_frames
        # 与 VAD 的 _stream_pos 同坐标系：都是「累计喂进来的样本数」。
        # 延迟基准用它把预触发事件对齐到语音起点。
        self._stream_pos = 0

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_path=DEFAULT_MODEL, thresholds_path=DEFAULT_THRESHOLDS,
             gate_path=None, pool_path=None, selective=None,
             fallback_mode: Optional[str] = None, **kwargs) -> 'DialectEngine':
        if str(ENGINEERING_ROOT) not in sys.path:
            sys.path.insert(0, str(ENGINEERING_ROOT))
        from runtime.v3_model import V3Predictor
        predictor = V3Predictor.load(str(model_path), str(thresholds_path))
        gate = cls.build_gate(
            gate_path=gate_path, pool_path=pool_path, selective=selective,
            fallback_threshold=float(getattr(predictor, 'threshold', 0.945)),
            fallback_mode=fallback_mode)
        return cls(predictor, gate=gate, **kwargs)

    @staticmethod
    def build_gate(gate_path=None, pool_path=None, selective=None,
                   fallback_threshold: float = 0.945,
                   fallback_mode: Optional[str] = None):
        """按开关装配双信号门限；返回 ``None`` 表示走单阈值路径。

        ``selective`` 语义（**默认关闭**）：
        - ``None`` / ``False``：不启用，行为与 S3 之前逐字一致。默认必须关闭，
          否则 ``scripts/eval_real_far.py``、``tests/test_pipeline.py`` 这些既有
          调用点会在无人察觉的情况下换掉判定口径——S0 公布的 FAR 0.3110 就是
          单阈值口径，被静默改成双信号后那个数字再也复现不出来。
        - ``True``：启用。规则文件缺失直接报错，**不静默降级**，否则
          「以为开了双信号、实际在跑 0.945 单阈值」这种最坏情况看不出来。

        参照池缺失是另一回事：它属于「模型先到、池后到」的正常分步下发状态，
        ``SelectiveGate`` 会自己降级并在每条结果的 ``rule`` 字段上标明，
        所以这里只打印一行警告，不抛异常。
        """
        if str(ENGINEERING_ROOT) not in sys.path:
            sys.path.insert(0, str(ENGINEERING_ROOT))
        from runtime.selective import SelectiveGate

        if not selective:
            return None
        gp = Path(gate_path) if gate_path else DEFAULT_GATE
        if not gp.exists():
            raise FileNotFoundError(
                f'已显式启用双信号拒识，但规则文件不存在: {gp}\n'
                f'先跑 scripts/build_selective_gate.py 生成它，'
                f'或把 recognition.selective.enabled 设为 false')
        gate = SelectiveGate.load(gp, pool_path=pool_path, models_dir=gp.parent)
        if fallback_mode:
            gate.fallback_mode = str(fallback_mode)
        if not gate.meta.get('pool_path'):
            print(f'[warn] 双信号门限缺参照池，退回 {gate.fallback_mode} 路径'
                  f'（单阈值 {gate.fallback_threshold}）: {gp}', file=sys.stderr, flush=True)
        return gate

    @property
    def classes_(self) -> List[str]:
        return list(self.predictor.classes_)

    # ------------------------------------------------------------------
    # 单遍路径：行为与 S4 之前逐字一致，公布的历史指标都以它为准。
    # ------------------------------------------------------------------
    def process_utterance(self, audio: np.ndarray, fs: int = 16000) -> UtteranceResult:
        """对一段已经切好的语音做识别并执行。"""
        from runtime.features_v3 import extract_sequence

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        started = time.perf_counter()
        if audio.size == 0:
            return UtteranceResult(command='', confidence=0.0, accepted=False,
                                   reason='empty_audio')
        if int(fs) != int(self.predictor.feature_spec.sample_rate):
            return UtteranceResult(command='', confidence=0.0, accepted=False,
                                   reason=f'sample_rate_mismatch:{fs}')

        seq = extract_sequence(audio, int(fs), self.predictor.feature_spec)
        # 门限开着就走 predict_detailed：一次前向同时拿部署概率与嵌入，
        # 不能为了算 ood 再跑一遍网络（端侧延迟预算见 docs/bench_latency.json）。
        # 预测器没有 predict_detailed（例如测试里的替身）时退回普通前向，
        # gate.decide 拿不到嵌入会自行降级，判定路径仍然闭合。
        extras = None
        if self.gate is not None and hasattr(self.predictor, 'predict_detailed'):
            pred, extras = self.predictor.predict_detailed(seq)
        else:
            pred = self.predictor.predict_features(seq)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if self.gate is not None:
            prob = np.asarray([pred.scores[c] for c in self.predictor.classes_],
                              dtype=np.float64)
            decision = self.gate.decide(prob, (extras or {}).get('embedding'))
            accepted = bool(decision['accepted'])
            rule = str(decision['rule'])
            reason = str(decision['reason'])
            signals = {k: float(v) for k, v in decision['signals'].items()
                       if v is not None}
        else:
            accepted = bool(pred.confidence >= self.threshold)
            rule = 'single_threshold'
            reason = '' if accepted else f'below_threshold:{self.threshold:.2f}'
            signals = {'maxprob': float(pred.confidence)}

        result = UtteranceResult(
            command=pred.command,
            confidence=float(pred.confidence),
            accepted=accepted,
            scores=dict(pred.scores),
            acoustic_scores=dict(pred.scores),
            margin=float(pred.margin),
            duration_sec=audio.size / float(fs),
            latency_ms=latency_ms,
            rule=rule,
            signals=signals,
        )
        if not accepted:
            result.reason = reason

        if accepted:
            if self.executor is not None:
                try:
                    self.executor.execute(pred.command)
                    result.executed = True
                except Exception as exc:  # 执行失败不应打断识别循环
                    result.executed = False
                    result.reason = f'executor_error:{exc}'
            if self.speaker is not None:
                try:
                    result.spoken = bool(self.speaker.speak(pred.command))
                except Exception as exc:
                    result.spoken = False
                    result.reason = (result.reason + f'|speaker_error:{exc}').strip('|')

        self._log(result)
        if self.on_result is not None:
            self.on_result(result)
        return result

    # ------------------------------------------------------------------
    # 两遍路径
    # ------------------------------------------------------------------
    def _verifier_obj(self):
        if self._verifier is None:
            self._verifier = _runtime('verify').Pass2Verifier(self.predictor,
                                                              bias=self.bias)
        return self._verifier

    def _pass1_obj(self):
        if self._pass1 is None:
            self._pass1 = _runtime('streaming').Pass1Detector(
                self.predictor, fs=16000, window_ms=self.window_ms,
                hop_ms=self.hop_ms, pretrigger_threshold=self.pretrigger_threshold,
                soft_threshold=self.fast_endpoint_threshold)
        return self._pass1

    def _read_state(self):
        if self.state_source is None:
            return None
        try:
            return self.state_source.read()
        except Exception:
            # 状态源不可达不该拖垮识别：先验没了，决策照常走（只是少一个信号）。
            return None

    def process_segment(self, audio: np.ndarray, fs: int = 16000,
                        candidate: Optional[str] = None) -> UtteranceResult:
        """两遍路径：Pass-2 整句复核 -> 门限 -> 决策层 -> 执行 / 澄清 / 拒识。"""
        started = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return self._emit(UtteranceResult(command='', confidence=0.0,
                                              accepted=False, reason='empty_audio'))
        if int(fs) != int(self.predictor.feature_spec.sample_rate):
            return self._emit(UtteranceResult(command='', confidence=0.0, accepted=False,
                                              reason=f'sample_rate_mismatch:{fs}'))

        v = self._verifier_obj().verify(audio, int(fs), candidate=candidate)

        # 门限用**无偏**后验：门限是在无偏口径上标定的工件，
        # 拿带关键词偏置的后验去比阈值等于换了口径。
        acoustic = np.asarray(v.acoustic_probs, dtype=np.float64)
        if self.gate is not None:
            g = self.gate.decide(acoustic, v.embedding)
            gate_accepted = bool(g['accepted'])
            gate_rule = str(g['rule'])
            gate_reason = str(g['reason'])
            signals = {k: float(val) for k, val in g['signals'].items()
                       if val is not None}
        else:
            top = float(acoustic.max()) if acoustic.size else 0.0
            gate_accepted = bool(top >= self.threshold)
            gate_rule = 'single_threshold'
            gate_reason = '' if gate_accepted else f'below_threshold:{self.threshold:.2f}'
            signals = {'maxprob': top}

        retry_mode = self.dialog.awaiting_retry
        clarify_allowed = (not retry_mode) and (not self.dialog.exhausted)

        decision = _runtime('decision').decide(
            v.probs, v.classes,
            state=self._read_state(),
            gate_accepted=gate_accepted,
            gate_reason=gate_reason,
            costs=self.decision_costs,
            confusable_pairs=self.confusable_pairs,
            competitor_floor=self.competitor_floor,
            prior_temperature=self.prior_temperature,
            redundancy_weight=self.redundancy_weight,
            prior_override_margin=self.prior_override_margin,
            seed_agree=v.seed_agree,
            seed_std=v.seed_std,
            consistency=v.consistency,
            consistency_margin=self.consistency_margin,
            clarify_allowed=clarify_allowed,
        )

        result = UtteranceResult(
            command=decision.command or v.top1,
            confidence=float(np.max(v.probs)) if v.probs.size else 0.0,
            accepted=(decision.action == 'execute'),
            scores={c: float(v.probs[i]) for i, c in enumerate(v.classes)},
            acoustic_scores={c: float(v.acoustic_probs[i])
                             for i, c in enumerate(v.classes)},
            margin=float(decision.margin),
            duration_sec=audio.size / float(fs),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            reason=decision.reason,
            rule=gate_rule,
            # 门限信号（maxprob / ood）+ 决策层信号（seed_agree / seed_std /
            # consistency_degraded ...）。合并进来是因为「为什么降级」必须
            # 在日志里看得见，否则降级就等于把 FAR 悄悄退回单信号那一档。
            # maxprob 不重复带：决策层算的是**带偏置**的后验，与门限用的
            # 无偏 maxprob 不是同一个口径，它记在 result.confidence 上。
            signals={**signals,
                     **{k: float(val) for k, val in decision.signals.items()
                        if k != 'maxprob' and val is not None}},
            action=decision.action,
            zone=decision.zone,
            gate_accepted=gate_accepted,
            pass1_command=candidate or '',
            consistency=v.consistency,
            expected_costs=dict(decision.expected_costs),
        )

        if decision.action == 'execute':
            self._do_execute(result, decision.command)
        elif decision.action == 'clarify':
            self._do_clarify(result, decision)
        else:
            result.action = 'reject'
            result.zone = 'red'
            result.executed = False
            self._speak_reject(result)

        if retry_mode:
            # 这一句是用户对上一次澄清的重说：不管成没成，这一轮澄清都结束。
            self.dialog.on_retry(success=(result.action == 'execute'))

        return self._emit(result)

    # ------------------------------------------------------------------
    def _do_execute(self, result: UtteranceResult, command: str) -> None:
        if not self.debouncer.allow(command):
            # 抑制的是**执行**，不是识别：命令照样记进日志。
            result.debounced = True
            result.executed = False
            result.reason = 'debounced'
            return
        if self.executor is None:
            result.executed = False
            result.reason = 'no_executor'
            return
        t0 = time.perf_counter()
        try:
            info = self.executor.dispatch(command)
        except Exception as exc:
            result.executed = False
            result.reason = f'executor_error:{exc}'
            self._speak_error(result)
            return
        result.action_latency_ms = (time.perf_counter() - t0) * 1000.0
        result.ack = dict(info or {})
        result.executed = bool((info or {}).get('ok', True))
        if self.state_source is not None and hasattr(self.state_source, 'apply'):
            try:
                self.state_source.apply(command)
            except Exception:
                pass
        self._speak_after_action(command, result)

    def _do_clarify(self, result: UtteranceResult, decision) -> None:
        prompt = self.dialog.request(decision.candidates)
        if prompt is None:
            # 澄清额度用尽：按方案 §6.7 拒识并播报可用命令。
            result.action = 'reject'
            result.zone = 'red'
            result.reason = 'clarify_exhausted'
            result.executed = False
            self._speak_reject(result)
            return
        result.action = 'clarify'
        result.zone = 'yellow'
        result.reason = decision.reason
        result.clarify = prompt.as_dict()
        self._speak_clarify(result, prompt)

    # ------------------------------------------------------------------
    def _speak_after_action(self, command: str, result: UtteranceResult) -> None:
        """动作已下发，播报放后台：动作不能等播报。"""
        if self.speaker is None:
            return
        if self.broadcast_async:
            threading.Thread(target=self._speak_worker, args=(command, result),
                             daemon=True).start()
        else:
            self._speak_worker(command, result)

    def _speak_worker(self, command: str, result: UtteranceResult) -> None:
        try:
            if self.executor is not None and getattr(self.executor, 'async_ack', False):
                ack = self.executor.wait_ack(timeout=self.ack_timeout)
                result.ack.update(ack or {})
                ok = bool((ack or {}).get('ok', True))
            else:
                ok = bool(result.ack.get('ok', True))
            if ok:
                if self.broadcast_command:
                    result.spoken = bool(self.speaker.speak(command))
                result.spoken = bool(self.speaker.speak_done()) or result.spoken
            else:
                result.spoken = bool(self.speaker.speak_error())
        except Exception as exc:
            result.spoken = False
            result.reason = (result.reason + f'|speak_error:{exc}').strip('|')
        self._log_extra({'event': 'broadcast', 'command': command,
                         'ok': bool(result.ack.get('ok', True)),
                         'spoken': bool(result.spoken)})

    def _speak_clarify(self, result: UtteranceResult, prompt) -> None:
        if self.speaker is None:
            return
        try:
            result.spoken = bool(self.speaker.speak_clarify(prompt.candidates))
        except Exception as exc:
            result.spoken = False
            result.reason = (result.reason + f'|speaker_error:{exc}').strip('|')

    def _speak_reject(self, result: UtteranceResult) -> None:
        if self.speaker is None:
            return
        try:
            result.spoken = bool(self.speaker.speak_reject())
        except Exception as exc:
            result.spoken = False
            result.reason = (result.reason + f'|speaker_error:{exc}').strip('|')

    def _speak_error(self, result: UtteranceResult) -> None:
        if self.speaker is None:
            return
        try:
            result.spoken = bool(self.speaker.speak_error())
        except Exception as exc:
            result.spoken = False
            result.reason = (result.reason + f'|speaker_error:{exc}').strip('|')

    # ------------------------------------------------------------------
    # 流式路径
    # ------------------------------------------------------------------
    def process_stream_block(self, block, fs: int = 16000) -> List[UtteranceResult]:
        """喂一块流式音频，返回本次切出的语音段的结果列表。

        顺序是「先 Pass-1 预触发、后 VAD 切段」：预触发要在切段之前拿到，
        因为它负责把静音挂起压短（快速端点），否则动作延迟永远差一个 400 ms。

        快速端点有**两条**通路：严格预触发（连续两窗一致），以及软候选
        （单个窗后验够高）。软候选是必要的：Pass-1 的窗长 1 s，短命令的
        语音段往往在第一个窗被打分之前就结束了，两窗一致的预触发结构上
        永远等不到，挂起压在默认 390 ms，300 ms 预算就破了。
        """
        results: List[UtteranceResult] = []
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return results
        self._stream_pos += int(block.size)

        detector = self._pass1_obj()
        event = detector.feed(block)
        if event is not None:
            self._candidate = event.command
            self._pretrigger_ms = event.latency_ms
            self._fast_endpoint()
            if self.on_pretrigger is not None:
                try:
                    self.on_pretrigger(event)
                except Exception:
                    pass
        elif detector.soft_candidate is not None:
            self._fast_endpoint()

        segments = self.vad.feed(block)
        # 切段时生效的静音挂起：快速端点生效后是 fast_silence_ms，
        # 否则是默认的 400 ms。
        hangover = self._hangover_ms()
        for seg in segments:
            result = self.process_segment(seg, fs, candidate=self._candidate)
            result.pretrigger_ms = self._pretrigger_ms
            result.hangover_ms = hangover
            results.append(result)
            self._after_utterance()
        return results

    def _hangover_ms(self) -> float:
        frame_ms = 1000.0 * float(self.vad.frame_len) / max(int(self.vad.fs), 1)
        return float(self.vad.min_silence_frames) * frame_ms

    def reset_stream(self) -> None:
        """复位流式状态（VAD / Pass-1 / 候选）。

        基准脚本按「一条语音一段流」测量延迟时需要它：否则相邻语句会被 VAD
        连成一段，测出来的就不是单条语音的延迟。噪声底一并复位——连续监听
        时它该自适应，但逐条回放时每条是独立的流，复位后才与「裁剪语料时
        每条用一个新 VAD」的条件一致。
        """
        self.vad.reset()
        self.vad.reset_noise_floor()
        self._pass1_obj().reset()
        self._candidate = None
        self._pretrigger_ms = 0.0
        self.vad.min_silence_frames = self._default_min_silence_frames

    def _fast_endpoint(self) -> None:
        """预触发之后把静音挂起压短，让「说完 -> 动作」落进 300 ms 预算。"""
        frame_ms = 1000.0 * float(self.vad.frame_len) / max(int(self.vad.fs), 1)
        fast = max(1, int(self.fast_silence_ms / max(frame_ms, 1e-6)))
        if fast < self.vad.min_silence_frames:
            self.vad.min_silence_frames = fast

    def _after_utterance(self) -> None:
        self._candidate = None
        self._pretrigger_ms = 0.0
        self._pass1_obj().reset()
        self.vad.min_silence_frames = self._default_min_silence_frames

    # ------------------------------------------------------------------
    def run_stream(self, capture, fs: int = 16000, chunk_frames: int = 1024,
                   two_pass: bool = False):
        """从音频采集对象持续取块，切段并逐段处理，产出结果迭代器。

        ``two_pass=False``（默认）走单遍路径，与 S4 之前一致；
        ``two_pass=True`` 走流式两遍路径（预触发 + 复核 + 决策层）。
        """
        while True:
            block = capture.read().reshape(-1)
            if two_pass:
                for result in self.process_stream_block(block, fs):
                    yield result
            else:
                for utterance in self.vad.feed(block):
                    yield self.process_utterance(utterance, fs)

    # ------------------------------------------------------------------
    def _emit(self, result: UtteranceResult) -> UtteranceResult:
        self._log(result)
        if self.on_result is not None:
            self.on_result(result)
        return result

    def _log(self, result: UtteranceResult) -> None:
        self._log_extra(result.as_dict())

    def _log_extra(self, row: dict) -> None:
        if not self.log_path:
            return
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

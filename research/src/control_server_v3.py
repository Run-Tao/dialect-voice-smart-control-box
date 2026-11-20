# -*- coding: utf-8 -*-
"""v3 控制服务：可插拔识别器 + 六类命令模拟设备 + JSONL 审计日志。

设计目标：与二期 ``src/control_server.py`` 的 ``AudioControlServer`` /
``DTWControlServer`` 保持同样的字段风格，同时把「识别器」抽象为可插拔对象，
以便 v2 DTW pickle 与后续 v3 模型可以无缝切换。

约定：识别器（predictor）只要实现 ``predict(audio_path)`` 并返回带有
``command`` / ``confidence`` / ``scores`` / ``distance`` 任意子集的对象即可；
也可直接返回 ``(command, confidence, scores)`` 元组或同名字典。
"""
import json
import math
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# 六类受控命令（与二期串口协议 src/serial_protocol.py 保持一致）
SUPPORTED_COMMANDS = (
    'LIGHT_ON',
    'LIGHT_OFF',
    'CURTAIN_OPEN',
    'CURTAIN_CLOSE',
    'ALL_OFF',
    'STATUS_ALL',
)

# 界面/日志使用的中文命令名
COMMAND_LABELS = {
    'LIGHT_ON': '開燈',
    'LIGHT_OFF': '關燈',
    'CURTAIN_OPEN': '打開窗簾',
    'CURTAIN_CLOSE': '關閉窗簾',
    'ALL_OFF': '全屋關閉',
    'STATUS_ALL': '查詢全屋狀態',
}

DEFAULT_STATE = {'light': 'off', 'curtain': 'closed'}

DEFAULT_FEATURE_CONFIG = {
    'sample_rate': 16000,
    'n_fft': 512,
    'hop_length': 160,
    'n_mfcc': 13,
    'n_mels': 26,
    'fmin': 50.0,
    'fmax': 8000.0,
    'preemphasis': 0.97,
    'use_cmvn': True,
}


def state_to_text(state):
    """把设备状态 dict 转成中文文案，如「燈：開；窗簾：關」。"""
    light = '開' if state.get('light') == 'on' else '關'
    curtain = '開' if state.get('curtain') == 'open' else '關'
    return f'燈：{light}；窗簾：{curtain}'


def normalize_state(state=None):
    """规范化设备状态：只保留 light/curtain 两个键，非法值回落到默认值。"""
    base = dict(DEFAULT_STATE)
    if not state:
        return base
    if state.get('light') in ('on', 'off'):
        base['light'] = state['light']
    if state.get('curtain') in ('open', 'closed'):
        base['curtain'] = state['curtain']
    return base


class MockDeviceV3:
    """六类命令的模拟设备状态机。

    语义（与二期真实设备/串口协议一致）：
      * LIGHT_ON / LIGHT_OFF   只改灯，不动窗帘；
      * CURTAIN_OPEN / CURTAIN_CLOSE 只改窗帘，不动灯；
      * ALL_OFF                同时关灯 + 关窗帘；
      * STATUS_ALL             只查询，不改变任何状态。
    拒绝执行未知命令时抛 ``ProtocolError``（ValueError 子类），且不改状态。
    """

    def __init__(self, state=None):
        self._state = normalize_state(state)
        self.seen_seq = set()
        self.history = []

    @property
    def state(self):
        """当前设备状态副本。"""
        return dict(self._state)

    def snapshot(self):
        return {'light': self._state['light'], 'curtain': self._state['curtain'],
                'text': state_to_text(self._state)}

    def execute(self, seq, intent):
        """执行一条命令，返回中文设备反馈文案；不修改状态则返回查询文案。"""
        if not isinstance(seq, int) or seq < 0:
            raise ValueError('seq 必须是非负整数')
        if intent not in SUPPORTED_COMMANDS:
            raise ValueError(f'未知命令: {intent}')
        if seq in self.seen_seq:
            # 与串口模拟设备一致：重复 seq 不重复执行（幂等保护）
            return f'重複指令（seq={seq}），已忽略'
        self.seen_seq.add(seq)

        if intent == 'LIGHT_ON':
            self._state['light'] = 'on'
            response = '已開燈'
        elif intent == 'LIGHT_OFF':
            self._state['light'] = 'off'
            response = '已關燈'
        elif intent == 'CURTAIN_OPEN':
            self._state['curtain'] = 'open'
            response = '窗簾已打開'
        elif intent == 'CURTAIN_CLOSE':
            self._state['curtain'] = 'closed'
            response = '窗簾已關閉'
        elif intent == 'ALL_OFF':
            self._state['light'] = 'off'
            self._state['curtain'] = 'closed'
            response = '全屋已關閉'
        else:  # STATUS_ALL
            response = f'狀態查詢：{state_to_text(self._state)}'

        self.history.append({'seq': seq, 'intent': intent,
                             'state': self.state, 'response': response})
        return response

    def reset(self, state=None):
        self._state = normalize_state(state)
        self.seen_seq = set()
        self.history = []
        return self.state


@dataclass
class PredictionResult:
    """统一后的识别结果。"""
    command: str = None
    confidence: float = 0.0
    scores: dict = field(default_factory=dict)
    distance: float = None

    def as_dict(self):
        return {'command': self.command, 'confidence': self.confidence,
                'scores': self.scores, 'distance': self.distance}


def coerce_prediction(raw):
    """把识别器的任意返回形式统一成 ``PredictionResult``。

    支持：
      * ``PredictionResult`` 本身；
      * 带 ``command`` / ``confidence`` / ``scores`` / ``distance`` 属性的对象
        （如 ``src.baselines.Prediction``）；
      * 同名字典；
      * ``(command, confidence, scores)`` 或至少两元素的元组/列表。
    """
    if isinstance(raw, PredictionResult):
        return raw
    if raw is None:
        raise ValueError('识别器返回了空结果')

    command = confidence = scores = distance = None
    if isinstance(raw, dict):
        command = raw.get('command', raw.get('predicted_intent', raw.get('intent')))
        confidence = raw.get('confidence', raw.get('score'))
        scores = raw.get('scores')
        distance = raw.get('distance')
    elif isinstance(raw, (tuple, list)):
        if len(raw) < 2:
            raise ValueError(f'识别结果元组至少需要 (command, confidence)，收到: {raw!r}')
        command = raw[0]
        confidence = raw[1]
        if len(raw) >= 3:
            scores = raw[2]
    else:
        # 属性风格对象；也兼容只有 predict_sequence 的旧模型输出
        command = getattr(raw, 'command', getattr(raw, 'predicted_intent', None))
        confidence = getattr(raw, 'confidence', getattr(raw, 'score', None))
        scores = getattr(raw, 'scores', None)
        distance = getattr(raw, 'distance', None)

    scores = {str(k): float(v) for k, v in (scores or {}).items()}
    if confidence is None:
        if scores:
            confidence = max(scores.values())
        else:
            confidence = 0.0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    try:
        distance = float(distance) if distance is not None else None
    except (TypeError, ValueError):
        distance = None
    return PredictionResult(command=(str(command) if command is not None else None),
                            confidence=confidence, scores=scores, distance=distance)


def _project_root():
    return Path(__file__).resolve().parents[1]


def load_feature_config(config_path=None):
    """读取项目 config.json 的 features；失败时回落到内置默认值。"""
    if config_path is None:
        candidate = _project_root() / 'config.json'
        config_path = candidate if candidate.exists() else None
    if config_path is not None:
        try:
            from .config import load_config
        except ImportError:  # pragma: no cover - 独立脚本导入场景
            from src.config import load_config
        return dict(load_config(config_path)['features'])
    return dict(DEFAULT_FEATURE_CONFIG)


def _extract_sequence(audio_path, feature_config):
    """按 v2 声学前端提取 MFCC 帧序列（DTW pickle 模型使用）。"""
    from .audio_io import load_wav
    from .features import extract_mfcc

    audio, sr = load_wav(audio_path)
    cfg = feature_config
    if int(sr) != int(cfg['sample_rate']):
        raise ValueError(f'采样率不匹配：{sr} vs {cfg["sample_rate"]}')
    return extract_mfcc(
        audio, sr,
        n_fft=cfg['n_fft'], hop_length=cfg['hop_length'],
        n_mfcc=cfg['n_mfcc'], n_mels=cfg['n_mels'],
        fmin=cfg['fmin'], fmax=cfg['fmax'],
        preemphasis=cfg['preemphasis'], use_cmvn=cfg['use_cmvn'],
    )


class DTWPredictorAdapter:
    """把二期 ``DTWControlModel`` 包装成统一的 predict(audio_path) 接口。"""

    model_type = 'dtw_v2'

    def __init__(self, model, feature_config=None):
        self.model = model
        self.feature_config = feature_config or load_feature_config()
        self.threshold = float(getattr(model, 'threshold', 0.9))
        self.classes_ = list(getattr(model, 'classes_', []))
        self.model_name = f'dtw_v2({len(self.classes_)}类)'

    def predict(self, audio_path):
        seq = _extract_sequence(audio_path, self.feature_config)
        pred = self.model.predict_sequence(seq)
        return coerce_prediction(pred)

    def predict_sequence(self, seq):
        """兼容二期 DTWControlModel 风格调用。"""
        return coerce_prediction(self.model.predict_sequence(seq))


class _SequencePredictorAdapter:
    """把只有 predict_sequence(seq) 的模型适配成 predict(audio_path)。"""

    def __init__(self, model, feature_config=None, threshold=None):
        self.model = model
        self.feature_config = feature_config or load_feature_config()
        self.threshold = float(threshold if threshold is not None
                               else getattr(model, 'threshold', 0.9))
        self.model_name = type(model).__name__

    def predict(self, audio_path):
        seq = _extract_sequence(audio_path, self.feature_config)
        return coerce_prediction(self.model.predict_sequence(seq))


def _try_load_v3(path):
    """动态探测 src/v3_model.py 的 V3Predictor.load(path)。

    返回 ``(predictor, None)`` 或 ``(None, 错误信息)``。模块不存在不算错误。
    """
    module = None
    try:
        from . import v3_model as module  # type: ignore
    except Exception:
        try:
            import src.v3_model as module  # type: ignore
        except Exception:
            module = None
    if module is None:
        return None, '未发现 src/v3_model.py（可继续使用 v2 DTW 模型）'
    loader = getattr(module, 'V3Predictor', None)
    if loader is None or not hasattr(loader, 'load'):
        return None, 'src/v3_model.py 中没有 V3Predictor.load()'
    try:
        predictor = loader.load(Path(path))
    except Exception as exc:  # 模型文件不匹配时回落到 v2
        return None, f'v3 模型加载失败: {exc}'
    if predictor is None:
        return None, 'v3 模型加载返回空对象'
    if not hasattr(predictor, 'predict'):
        if hasattr(predictor, 'predict_sequence'):
            predictor = _SequencePredictorAdapter(predictor,
                                                  threshold=getattr(predictor, 'threshold', None))
        else:
            return None, 'v3 预测器缺少 predict()/predict_sequence() 方法'
    if getattr(predictor, 'model_name', None) is None:
        try:
            predictor.model_name = type(predictor).__name__
        except Exception:
            pass
    return predictor, None


def load_predictor(path, feature_config=None, config_path=None):
    """加载识别器：优先 v3 模型，其次 v2 DTW pickle。

    参数
    ----
    path : 模型文件路径（v3 模型或 ``DTWControlModel.save`` 生成的 pickle）。
    feature_config : 可选，v2 DTW 模型需要的声学前端配置；默认读 config.json。
    config_path : 可选，自定义 config.json 路径。

    返回带 ``predict(audio_path)`` 与 ``threshold`` 属性的预测器对象。
    文件不存在或格式无法识别时抛出带中文提示的异常。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'模型文件不存在：{path}\n'
            '请确认路径；v2 DTW 演示可用 data/models/tts_control_model_dtw_demo.pkl，'
            'v3 模型请向 Lead 索取。')
    if path.is_dir():
        raise IsADirectoryError(f'模型路径是目录而不是文件：{path}')

    # 1) 优先尝试 v3 模型（Lead 后续提供 src/v3_model.py）
    v3_predictor, v3_note = _try_load_v3(path)
    if v3_predictor is not None:
        return v3_predictor

    # 2) 回落到 v2 DTW pickle
    try:
        with open(path, 'rb') as f:
            payload = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f'无法加载模型文件：{path}\n'
            f'  - v3 探测：{v3_note}\n'
            f'  - pickle 读取错误：{exc}\n'
            '支持 v3 V3Predictor.load() 模型或 v2 DTWControlModel pickle。') from exc

    if isinstance(payload, dict) and ('templates' in payload or payload.get('model_type') == 'dtw'):
        from .dtw_model import DTWControlModel
        model = DTWControlModel(payload.get('templates', {}),
                                payload.get('threshold', 0.9),
                                payload.get('max_frames', 60),
                                payload.get('temperature', 0.2))
        if not model.templates:
            raise RuntimeError(f'DTW 模型模板为空：{path}')
        return DTWPredictorAdapter(model, feature_config or load_feature_config(config_path))

    if hasattr(payload, 'predict_sequence'):
        return _SequencePredictorAdapter(payload, feature_config or load_feature_config(config_path))

    raise RuntimeError(
        f'无法识别的模型格式：{path}\n'
        f'  - v3 探测：{v3_note}\n'
        f'  - pickle 顶层类型：{type(payload).__name__}\n'
        '当前支持 v3 V3Predictor 与 v2 DTW 模板 pickle。')


class V3ControlServer:
    """音频 -> 识别/拒识 -> 模拟设备 -> JSONL 日志 的 v3 控制服务。

    与旧版兼容字段：``predicted_intent`` / ``confidence`` / ``threshold`` /
    ``accepted`` / ``device_response`` / ``device_state`` / ``scores`` /
    ``latency_sec``；v3 额外提供 ``predicted`` / ``state`` / ``response`` /
    ``latency`` 等别名，方便前端直接使用。
    """

    def __init__(self, predictor, threshold=None, device=None, log_path=None,
                 model_name=None):
        if predictor is None or not hasattr(predictor, 'predict'):
            raise TypeError('predictor 必须实现 predict(audio_path) 方法')
        self.predictor = predictor
        default_threshold = getattr(predictor, 'threshold', None)
        if threshold is None:
            threshold = 0.9 if default_threshold is None else default_threshold
        self.threshold = float(threshold)
        self.device = device or MockDeviceV3()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or getattr(predictor, 'model_name', None) \
            or type(predictor).__name__
        self.model_type = getattr(predictor, 'model_type', 'unknown')
        self._last_seq = 0

    # ---------------------------------------------------------------- 识别
    def predict(self, audio_path):
        return coerce_prediction(self.predictor.predict(audio_path))

    # ---------------------------------------------------------------- 处理
    def process(self, audio_path, seq=None, threshold=None):
        """处理一条音频：识别、阈值判定、执行设备命令、写日志。

        ``seq`` 省略时自动递增；``threshold`` 传入时覆盖本次调用的阈值。
        返回结构化结果 dict，并追加一行 JSONL 到 ``log_path``。
        """
        if seq is None:
            self._last_seq += 1
            seq = self._last_seq
        seq = int(seq)

        t0 = time.perf_counter()
        pred = self.predict(audio_path)

        effective_threshold = self.threshold if threshold is None else float(threshold)
        known = pred.command in SUPPORTED_COMMANDS
        accepted = bool(known and pred.confidence >= effective_threshold)
        if known:
            rejection_reason = None if accepted else 'low_confidence'
        else:
            rejection_reason = 'unknown_command'

        state_before = self.device.state
        if accepted:
            device_response = self.device.execute(seq, pred.command)
            response = device_response
        else:
            # 与旧版一致：拒识不执行设备命令、不改变状态
            device_response = 'REJECTED'
            if rejection_reason == 'unknown_command':
                raw = pred.command if pred.command else '（空）'
                response = f'已拒識：未知指令「{raw}」，未執行任何操作'
            else:
                response = (f'已拒識：置信度 {pred.confidence:.3f} 低於閾值 '
                            f'{effective_threshold:.3f}，未執行任何操作')

        state_after = self.device.state
        latency = round(time.perf_counter() - t0, 4)
        result = {
            # —— 与二期控制服务兼容的字段 ——
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'audio_path': str(audio_path),
            'seq': seq,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'predicted_intent': pred.command,
            'confidence': round(pred.confidence, 6),
            'threshold': effective_threshold,
            'accepted': accepted,
            'device_response': device_response,
            'device_state': state_after,
            'scores': pred.scores,
            'latency_sec': latency,
            # —— v3 语义别名（前端/Audit 友好）——
            'predicted': pred.command,
            'command_label': COMMAND_LABELS.get(pred.command, '未知指令'),
            'distance': pred.distance,
            'rejection_reason': rejection_reason,
            'state': state_after,
            'state_text': state_to_text(state_after),
            'previous_state': state_before,
            'state_changed': state_before != state_after,
            'response': response,
            'latency': latency,
        }
        self._append_log(result)
        return result

    # ---------------------------------------------------------------- 状态
    @property
    def state(self):
        return self.device.state

    def reset(self, state=None):
        self.device.reset(state)
        return self.device.state

    # ---------------------------------------------------------------- 日志
    def _append_log(self, result):
        if not self.log_path:
            return
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def read_log(self, limit=20):
        """读取最近 ``limit`` 条日志（最新在前），供 Web 演示台使用。"""
        if not self.log_path or not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding='utf-8').splitlines()
        records = []
        for line in lines[-int(limit):]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(records))

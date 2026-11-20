# -*- coding: utf-8 -*-
"""v3 控制服务 / 设备状态机 / 演示目录 / Web API 测试（不依赖真实模型文件）。"""
import csv
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control_server_v3 import (  # noqa: E402
    DTWPredictorAdapter,
    MockDeviceV3,
    V3ControlServer,
    coerce_prediction,
    load_predictor,
    state_to_text,
)
from src.dtw_model import DTWControlModel  # noqa: E402
from web.demo_catalog import build_demo_catalog  # noqa: E402

# 以文件方式加载 scripts/run_web_demo.py（scripts 不是包）
_SPEC = importlib.util.spec_from_file_location(
    'run_web_demo', PROJECT_ROOT / 'scripts' / 'run_web_demo.py')
run_web_demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_web_demo)


class MockPredictor:
    """最小可插拔识别器：不读音频，直接返回预设结果。"""

    model_type = 'mock'
    model_name = 'mock-predictor'

    def __init__(self, command='LIGHT_ON', confidence=0.95,
                 scores=None, threshold=0.7):
        self.command = command
        self.confidence = confidence
        self.scores = scores if scores is not None else {command: confidence}
        self.threshold = threshold
        self.calls = []

    def predict(self, audio_path):
        self.calls.append(str(audio_path))
        return SimpleNamespace(command=self.command, confidence=self.confidence,
                               scores=dict(self.scores), distance=0.123)


class FailingPredictor:
    threshold = 0.7

    def predict(self, audio_path):
        raise FileNotFoundError(f'模拟音频缺失: {audio_path}')


def make_wav(path, seconds=0.2, sample_rate=16000):
    frames = int(seconds * sample_rate)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b'\x00\x00' * frames)
    return path


def make_server(predictor, log_path=None, threshold=None):
    return V3ControlServer(predictor, threshold=threshold,
                           device=MockDeviceV3(), log_path=log_path)


class TestMockDeviceV3(unittest.TestCase):
    def test_light_commands_only_touch_light(self):
        device = MockDeviceV3()
        response = device.execute(1, 'LIGHT_ON')
        self.assertEqual(response, '已開燈')
        self.assertEqual(device.state, {'light': 'on', 'curtain': 'closed'})
        self.assertEqual(state_to_text(device.state), '燈：開；窗簾：關')
        response = device.execute(2, 'LIGHT_OFF')
        self.assertEqual(response, '已關燈')
        self.assertEqual(device.state, {'light': 'off', 'curtain': 'closed'})
        # 燈操作不能改動窗簾
        self.assertEqual(device.state['curtain'], 'closed')

    def test_curtain_commands_only_touch_curtain(self):
        device = MockDeviceV3()
        device.execute(1, 'LIGHT_ON')
        response = device.execute(2, 'CURTAIN_OPEN')
        self.assertEqual(response, '窗簾已打開')
        self.assertEqual(device.state, {'light': 'on', 'curtain': 'open'})
        response = device.execute(3, 'CURTAIN_CLOSE')
        self.assertEqual(response, '窗簾已關閉')
        self.assertEqual(device.state, {'light': 'on', 'curtain': 'closed'})

    def test_all_off_closes_light_and_curtain(self):
        device = MockDeviceV3()
        device.execute(1, 'LIGHT_ON')
        device.execute(2, 'CURTAIN_OPEN')
        response = device.execute(3, 'ALL_OFF')
        self.assertEqual(response, '全屋已關閉')
        self.assertEqual(device.state, {'light': 'off', 'curtain': 'closed'})

    def test_status_all_only_queries(self):
        device = MockDeviceV3()
        device.execute(1, 'LIGHT_ON')
        device.execute(2, 'CURTAIN_OPEN')
        before = device.state
        response = device.execute(3, 'STATUS_ALL')
        self.assertIn('狀態查詢', response)
        self.assertIn('燈：開', response)
        self.assertIn('窗簾：開', response)
        self.assertEqual(device.state, before)

    def test_unknown_intent_rejected_without_state_change(self):
        device = MockDeviceV3()
        with self.assertRaises(ValueError):
            device.execute(1, 'TURN_ON_TV')
        self.assertEqual(device.state, {'light': 'off', 'curtain': 'closed'})
        with self.assertRaises(ValueError):
            device.execute(-1, 'LIGHT_ON')

    def test_duplicate_seq_is_idempotent(self):
        device = MockDeviceV3()
        device.execute(1, 'LIGHT_ON')
        response = device.execute(1, 'LIGHT_OFF')
        self.assertIn('重複', response)
        self.assertEqual(device.state['light'], 'on')

    def test_reset(self):
        device = MockDeviceV3()
        device.execute(1, 'ALL_OFF')
        device.reset()
        self.assertEqual(device.state, {'light': 'off', 'curtain': 'closed'})
        self.assertEqual(device.seen_seq, set())


class TestV3ControlServer(unittest.TestCase):
    def test_accept_updates_state_and_exposes_compatible_fields(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / 'log.jsonl'
            predictor = MockPredictor('LIGHT_ON', 0.95)
            server = make_server(predictor, log_path)
            result = server.process(Path(d) / 'a.wav', 1)
            self.assertTrue(result['accepted'])
            self.assertEqual(result['predicted_intent'], 'LIGHT_ON')
            self.assertEqual(result['predicted'], 'LIGHT_ON')
            self.assertEqual(result['command_label'], '開燈')
            self.assertEqual(result['device_response'], '已開燈')
            self.assertEqual(result['response'], '已開燈')
            self.assertEqual(result['device_state'], {'light': 'on', 'curtain': 'closed'})
            self.assertEqual(result['state'], {'light': 'on', 'curtain': 'closed'})
            self.assertEqual(result['state_text'], '燈：開；窗簾：關')
            self.assertTrue(result['state_changed'])
            self.assertEqual(result['seq'], 1)
            self.assertAlmostEqual(result['threshold'], 0.7)
            self.assertGreaterEqual(result['latency_sec'], 0)
            self.assertEqual(result['latency'], result['latency_sec'])
            self.assertEqual(result['scores'], {'LIGHT_ON': 0.95})
            self.assertIsNone(result['rejection_reason'])
            self.assertEqual(predictor.calls, [str(Path(d) / 'a.wav')])

    def test_low_confidence_rejected_without_state_change(self):
        with tempfile.TemporaryDirectory() as d:
            server = make_server(MockPredictor('CURTAIN_OPEN', 0.4), Path(d) / 'log.jsonl')
            result = server.process(Path(d) / 'a.wav', 1)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['rejection_reason'], 'low_confidence')
            self.assertEqual(result['device_response'], 'REJECTED')
            self.assertIn('已拒識', result['response'])
            self.assertFalse(result['state_changed'])
            self.assertEqual(server.state, {'light': 'off', 'curtain': 'closed'})

    def test_unknown_command_rejected_even_with_high_confidence(self):
        with tempfile.TemporaryDirectory() as d:
            server = make_server(MockPredictor('PLAY_MUSIC', 0.99), Path(d) / 'log.jsonl')
            result = server.process(Path(d) / 'a.wav', 1)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['rejection_reason'], 'unknown_command')
            self.assertIn('未知指令', result['response'])
            self.assertEqual(server.state, {'light': 'off', 'curtain': 'closed'})

    def test_per_call_threshold_override(self):
        with tempfile.TemporaryDirectory() as d:
            server = make_server(MockPredictor('LIGHT_ON', 0.6), Path(d) / 'log.jsonl')
            rejected = server.process(Path(d) / 'a.wav', 1)          # 默认阈值 0.7
            self.assertFalse(rejected['accepted'])
            accepted = server.process(Path(d) / 'b.wav', 2, threshold=0.5)
            self.assertTrue(accepted['accepted'])
            self.assertEqual(accepted['threshold'], 0.5)
            self.assertEqual(server.state['light'], 'on')

    def test_constructor_threshold_override_model_default(self):
        server = make_server(MockPredictor('LIGHT_ON', 0.6, threshold=0.9), threshold=0.5)
        self.assertEqual(server.threshold, 0.5)
        self.assertTrue(server.process('x.wav', 1)['accepted'])
        default_server = make_server(MockPredictor('LIGHT_ON', 0.6, threshold=0.9))
        self.assertEqual(default_server.threshold, 0.9)

    def test_state_machine_semantics_through_server(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / 'log.jsonl'
            predictor = MockPredictor('LIGHT_ON', 1.0)
            server = make_server(predictor, log_path)
            steps = [
                ('LIGHT_ON', {'light': 'on', 'curtain': 'closed'}, '已開燈'),
                ('CURTAIN_OPEN', {'light': 'on', 'curtain': 'open'}, '窗簾已打開'),
                ('CURTAIN_CLOSE', {'light': 'on', 'curtain': 'closed'}, '窗簾已關閉'),
                ('STATUS_ALL', {'light': 'on', 'curtain': 'closed'}, None),
                ('LIGHT_OFF', {'light': 'off', 'curtain': 'closed'}, '已關燈'),
                ('CURTAIN_OPEN', {'light': 'off', 'curtain': 'open'}, '窗簾已打開'),
                ('ALL_OFF', {'light': 'off', 'curtain': 'closed'}, '全屋已關閉'),
            ]
            for seq, (command, expected_state, expected_response) in enumerate(steps, 1):
                predictor.command = command
                predictor.scores = {command: 1.0}
                result = server.process(Path(d) / f'{seq}.wav', seq)
                self.assertTrue(result['accepted'])
                self.assertEqual(result['state'], expected_state)
                if expected_response:
                    self.assertEqual(result['response'], expected_response)
                else:
                    self.assertIn('狀態查詢', result['response'])

    def test_log_jsonl_complete_and_read_reversed(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / 'nested' / 'v3_demo_log.jsonl'
            server = make_server(MockPredictor('LIGHT_ON', 0.95), log_path)
            server.process(Path(d) / 'a.wav', 1)
            server.process(Path(d) / 'b.wav', 2, threshold=0.5)
            lines = [ln for ln in log_path.read_text(encoding='utf-8').splitlines() if ln]
            self.assertEqual(len(lines), 2)
            record = json.loads(lines[0])
            for key in ('timestamp', 'audio_path', 'seq', 'predicted', 'predicted_intent',
                        'accepted', 'threshold', 'state', 'state_text', 'response',
                        'latency', 'latency_sec', 'scores', 'confidence'):
                self.assertIn(key, record)
            self.assertEqual(json.loads(lines[1])['threshold'], 0.5)
            latest = server.read_log(limit=5)
            self.assertEqual(len(latest), 2)
            self.assertEqual(latest[0]['seq'], 2)  # 最新在前
            self.assertEqual(server.read_log(limit=1)[0]['seq'], 2)

    def test_auto_increment_seq(self):
        with tempfile.TemporaryDirectory() as d:
            server = make_server(MockPredictor('LIGHT_ON', 0.95), Path(d) / 'log.jsonl')
            self.assertEqual(server.process('a.wav')['seq'], 1)
            self.assertEqual(server.process('b.wav')['seq'], 2)

    def test_prediction_exception_propagates_without_log(self):
        with tempfile.TemporaryDirectory() as d:
            log_path = Path(d) / 'log.jsonl'
            server = make_server(FailingPredictor(), log_path)
            with self.assertRaises(FileNotFoundError):
                server.process(Path(d) / 'missing.wav', 1)
            self.assertFalse(log_path.exists())
            self.assertEqual(server.state, {'light': 'off', 'curtain': 'closed'})

    def test_reset_restores_device(self):
        with tempfile.TemporaryDirectory() as d:
            server = make_server(MockPredictor('ALL_OFF', 1.0), Path(d) / 'log.jsonl')
            server.process('a.wav', 1)
            self.assertEqual(server.state, {'light': 'off', 'curtain': 'closed'})
            server.device.execute(2, 'LIGHT_ON')
            server.reset()
            self.assertEqual(server.state, {'light': 'off', 'curtain': 'closed'})


class TestCoercePrediction(unittest.TestCase):
    def test_dict_tuple_and_object_forms(self):
        self.assertEqual(coerce_prediction(
            {'command': 'LIGHT_ON', 'confidence': 0.8,
             'scores': {'LIGHT_ON': 0.8}}).command, 'LIGHT_ON')
        from_tuple = coerce_prediction(('ALL_OFF', 0.7, {'ALL_OFF': 0.7}))
        self.assertEqual(from_tuple.command, 'ALL_OFF')
        self.assertAlmostEqual(from_tuple.confidence, 0.7)
        self.assertEqual(coerce_prediction(
            SimpleNamespace(command='STATUS_ALL', scores={'STATUS_ALL': 0.6})).confidence,
            0.6)
        self.assertEqual(coerce_prediction(
            SimpleNamespace(predicted_intent='LIGHT_OFF', condition=0.5)).command,
            'LIGHT_OFF')

    def test_none_rejected(self):
        with self.assertRaises(ValueError):
            coerce_prediction(None)


class TestV3ModelProbe(unittest.TestCase):
    def test_v3_predictor_takes_priority_when_module_exists(self):
        """模拟 Lead 提供 src/v3_model.py 后的优先级（不依赖真实模型文件）。"""
        import types
        import src as src_pkg

        fake_module = types.ModuleType('src.v3_model')
        fake_predictor = MockPredictor('ALL_OFF', 1.0)

        class FakeV3Predictor:
            @staticmethod
            def load(path):
                return fake_predictor

        fake_module.V3Predictor = FakeV3Predictor
        old_module = sys.modules.get('src.v3_model')
        old_attr = getattr(src_pkg, 'v3_model', None)
        sys.modules['src.v3_model'] = fake_module
        setattr(src_pkg, 'v3_model', fake_module)
        try:
            with tempfile.TemporaryDirectory() as d:
                model_path = Path(d) / 'fake_v3.pkl'
                model_path.write_bytes(b'fake')
                loaded = load_predictor(model_path)
                self.assertIs(loaded, fake_predictor)
        finally:
            if old_module is None:
                sys.modules.pop('src.v3_model', None)
                if hasattr(src_pkg, 'v3_model'):
                    delattr(src_pkg, 'v3_model')
            else:
                sys.modules['src.v3_model'] = old_module
                if old_attr is not None:
                    setattr(src_pkg, 'v3_model', old_attr)


class TestLoadPredictor(unittest.TestCase):
    def test_missing_model_file(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_predictor(PROJECT_ROOT / 'data' / 'models' / '不存在的模型.pkl')
        self.assertIn('模型文件不存在', str(ctx.exception))

    def test_load_v2_dtw_pickle_and_predict(self):
        with tempfile.TemporaryDirectory() as d:
            templates = {
                'LIGHT_ON': [np.random.randn(8, 13).astype(np.float32)],
                'LIGHT_OFF': [np.random.randn(8, 13).astype(np.float32)],
            }
            model = DTWControlModel(templates, threshold=0.5, max_frames=20)
            model_path = Path(d) / 'dtw.pkl'
            model.save(model_path)
            predictor = load_predictor(model_path)
            self.assertIsInstance(predictor, DTWPredictorAdapter)
            self.assertAlmostEqual(predictor.threshold, 0.5)
            self.assertEqual(sorted(predictor.classes_), ['LIGHT_OFF', 'LIGHT_ON'])
            wav = make_wav(Path(d) / 'a.wav')
            result = predictor.predict(wav)
            self.assertIn(result.command, ('LIGHT_OFF', 'LIGHT_ON'))
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertEqual(set(result.scores), {'LIGHT_OFF', 'LIGHT_ON'})

    def test_invalid_pickle_gives_chinese_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'bad.pkl'
            with open(path, 'wb') as f:
                pickle.dump({'foo': 1}, f)
            with self.assertRaises(RuntimeError) as ctx:
                load_predictor(path)
            self.assertIn('无法识别的模型格式', str(ctx.exception))

    def test_directory_path_rejected(self):
        with self.assertRaises(IsADirectoryError):
            load_predictor(PROJECT_ROOT / 'data')


class TestDemoCatalog(unittest.TestCase):
    def _build_fixture(self, base):
        (base / 'data' / 'splits').mkdir(parents=True)
        (base / 'data' / 'tts_corpus' / 'wav16k_v2').mkdir(parents=True)
        (base / 'outputs').mkdir(parents=True)
        groups = []
        manifest_rows = []
        command_of_group = {}
        for i, command in enumerate(
                ['LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN', 'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL'],
                start=1):
            group_id = f'{command}_{i:02d}'
            command_of_group[group_id] = command
            groups.append({'expression_group_id': group_id, 'subset': 'test',
                           'command': command, 'register': 'direct', 'n_samples': '1',
                           'voices': 'edge-x', 'text': f'測試文本{i}'})
            # 每个命令放两条：先 edge 后 canto，验证音色优先排序
            for voice in ('edge-x', 'canto-tts-default'):
                audio_rel = f'wav16k_v2/{group_id}_{voice}.wav'
                wav = base / 'data' / 'tts_corpus' / audio_rel
                wav.write_bytes(b'RIFF')
                manifest_rows.append({
                    'audio_path': audio_rel, 'speaker_id': voice,
                    'voice_identity': voice, 'command': command, 'split': 'test',
                    'expression_group_id': group_id, 'text': f'測試文本{i}',
                    'register': 'direct'})
        groups.append({'expression_group_id': 'LIGHT_ON_99', 'subset': 'train',
                       'command': 'LIGHT_ON', 'register': 'direct', 'n_samples': '1',
                       'voices': 'edge-x', 'text': '訓練文本'})
        split_path = base / 'data' / 'splits' / 'track_a_expression_split.csv'
        with open(split_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(groups[0].keys()))
            writer.writeheader()
            writer.writerows(groups)
        manifest_path = base / 'data' / 'tts_corpus' / 'manifest_yue_v2.csv'
        with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        unknown_wav = base / 'data' / 'tts_corpus' / 'wav16k_v2' / 'unknown_x.wav'
        unknown_wav.write_bytes(b'RIFF')
        with open(base / 'outputs' / 'unknown_tts_log.csv', 'w',
                  encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['utt_id', 'text', 'audio_path', 'status'])
            writer.writeheader()
            writer.writerow({'utt_id': 'U1', 'text': '開冷氣',
                             'audio_path': str(unknown_wav), 'status': 'ok'})
        return unknown_wav

    def test_build_catalog_selects_test_split_and_preferred_voice(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            unknown_wav = self._build_fixture(base)
            catalog = build_demo_catalog(base)
            self.assertEqual(len(catalog['commands']), 6)
            by_command = {item['command']: item for item in catalog['commands']}
            self.assertEqual(set(by_command), {'LIGHT_ON', 'LIGHT_OFF', 'CURTAIN_OPEN',
                                               'CURTAIN_CLOSE', 'ALL_OFF', 'STATUS_ALL'})
            for command, item in by_command.items():
                self.assertTrue(Path(item['audio_path']).exists())
                self.assertEqual(item['voice'], 'canto-tts-default')  # 音色优先
                self.assertTrue(Path(item['audio_path']).name.startswith(command + '_'))
                self.assertEqual(item['candidate_count'], 2)
            self.assertIsNotNone(catalog['unknown'])
            self.assertEqual(Path(catalog['unknown']['audio_path']), unknown_wav)
            self.assertEqual(catalog['unknown']['text'], '開冷氣')

    def test_unknown_v3_csv_has_priority(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            self._build_fixture(base)
            v3_wav = base / 'data' / 'tts_corpus' / 'wav16k_v2' / 'unknown_v3_x.wav'
            v3_wav.write_bytes(b'RIFF')
            with open(base / 'data' / 'tts_corpus' / 'unknown_v3.csv', 'w',
                      encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['audio_path', 'text', 'status'])
                writer.writeheader()
                writer.writerow({'audio_path': str(v3_wav), 'text': '幫我開冷氣',
                                 'status': 'ok'})
            catalog = build_demo_catalog(base)
            self.assertEqual(Path(catalog['unknown']['audio_path']), v3_wav)
            self.assertEqual(catalog['unknown']['source'], 'data/tts_corpus/unknown_v3.csv')

    def test_missing_data_reports_warnings(self):
        with tempfile.TemporaryDirectory() as d:
            catalog = build_demo_catalog(Path(d))
            self.assertEqual(catalog['commands'], [])
            self.assertIsNone(catalog['unknown'])
            self.assertTrue(any('test 划分' in w for w in catalog['warnings']))


class TestWebDemoApi(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.wav = make_wav(self.base / 'sample.wav')
        self.log_path = self.base / 'v3_demo_log.jsonl'
        self.predictor = MockPredictor('LIGHT_ON', 0.95)
        self.server = V3ControlServer(self.predictor, device=MockDeviceV3(),
                                      log_path=self.log_path)
        self.catalog = {
            'base_dir': str(self.base),
            'commands': [{'kind': 'command', 'command': 'LIGHT_ON', 'label': '開燈',
                          'text': '幫我開燈', 'group': 'LIGHT_ON_01', 'voice': 'canto',
                          'audio_path': str(self.wav), 'source': 'test'}],
            'unknown': {'kind': 'unknown', 'label': '未知指令（應拒識）',
                        'text': '開冷氣', 'audio_path': str(self.wav),
                        'source': 'test'},
            'warnings': [],
        }
        self.app = run_web_demo.create_app(self.server, self.catalog,
                                           base_dir=self.base, log_path=self.log_path)
        self.client = self.app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def test_index_page_offline_inline(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/html', resp.headers['Content-Type'])
        html = resp.get_data(as_text=True)
        self.assertIn('方言智控盒', html)
        self.assertIn('操作日誌', html)
        self.assertIn('開燈', html)
        self.assertIn('未知指令（應拒識）', html)
        self.assertIn('cmd-btn', html)
        self.assertNotIn('https://cdn', html.lower())
        self.assertNotIn('__CATALOG_JSON__', html)
        self.assertNotIn('__COMMAND_BUTTONS__', html)
        self.assertNotIn('http://', html)
        self.assertNotIn('https://', html)
        self.assertIn('id="catalog-data"', html)

    def test_predict_accept_updates_state_and_log(self):
        resp = self.client.post('/api/predict', json={'audio_path': str(self.wav)})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['accepted'])
        self.assertEqual(data['predicted_intent'], 'LIGHT_ON')
        self.assertEqual(data['state'], {'light': 'on', 'curtain': 'closed'})
        self.assertIn('latency_sec', data)
        state = self.client.get('/api/state').get_json()
        self.assertEqual(state['state']['light'], 'on')
        self.assertTrue(self.log_path.exists())
        self.assertEqual(len(self.log_path.read_text(encoding='utf-8').splitlines()), 1)

    def test_predict_threshold_rejects(self):
        data = self.client.post('/api/predict',
                                json={'audio_path': str(self.wav),
                                      'threshold': 0.99}).get_json()
        self.assertFalse(data['accepted'])
        self.assertEqual(data['rejection_reason'], 'low_confidence')
        self.assertEqual(data['device_response'], 'REJECTED')
        state = self.client.get('/api/state').get_json()
        self.assertEqual(state['state'], {'light': 'off', 'curtain': 'closed'})

    def test_predict_invalid_threshold(self):
        for bad in (1.5, -0.1, 'high', True):
            resp = self.client.post('/api/predict',
                                    json={'audio_path': str(self.wav), 'threshold': bad})
            self.assertEqual(resp.status_code, 400, f'threshold={bad!r}')
            self.assertIn('error', resp.get_json())

    def test_predict_invalid_payload(self):
        self.assertEqual(self.client.post('/api/predict', json={}).status_code, 400)
        self.assertEqual(self.client.post('/api/predict',
                                          json={'audio_path': ' '}).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', data='not json',
                                          content_type='text/plain').status_code, 400)
        missing = self.client.post('/api/predict',
                                   json={'audio_path': str(self.base / 'nope.wav')})
        self.assertEqual(missing.status_code, 404)

    def test_unknown_command_rejected_by_api(self):
        self.predictor.command = 'PLAY_MUSIC'
        self.predictor.confidence = 0.99
        self.predictor.scores = {'PLAY_MUSIC': 0.99}
        data = self.client.post('/api/predict',
                                json={'audio_path': str(self.wav)}).get_json()
        self.assertFalse(data['accepted'])
        self.assertEqual(data['rejection_reason'], 'unknown_command')
        self.assertEqual(data['state'], {'light': 'off', 'curtain': 'closed'})

    def test_reset_and_log_endpoints(self):
        self.client.post('/api/predict', json={'audio_path': str(self.wav)})
        self.client.post('/api/predict', json={'audio_path': str(self.wav)})
        log = self.client.get('/api/log?limit=1').get_json()
        self.assertEqual(len(log['records']), 1)
        self.assertEqual(log['records'][0]['seq'], 2)
        self.assertIn('log_path', log)
        reset = self.client.post('/api/reset').get_json()
        self.assertEqual(reset['state'], {'light': 'off', 'curtain': 'closed'})
        self.assertEqual(reset['state_text'], '燈：關；窗簾：關')
        self.assertEqual(len(self.server.read_log(10)), 2)  # 重置不清审计日志

    def test_log_limit_validation_and_catalog(self):
        self.assertEqual(self.client.get('/api/log?limit=abc').status_code, 400)
        catalog = self.client.get('/api/catalog').get_json()
        self.assertEqual(catalog['commands'][0]['command'], 'LIGHT_ON')
        health = self.client.get('/api/health').get_json()
        self.assertEqual(health['status'], 'ok')


if __name__ == '__main__':
    unittest.main()

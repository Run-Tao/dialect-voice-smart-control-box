# -*- coding: utf-8 -*-
"""普通话播报层：把识别出的命令转成普通话音频，播放给智能音箱。

建议书里这一环是「装置播放预录的唤醒词 + 普通话命令音频，智能音箱接收后执行」，
此前仓库完全没有实现。转换发生在**语义层（命令标签 -> 普通话指令）**，
而不是特征层（方言 MFCC 向量 -> 普通话 MFCC 向量）——后者经实测无效，
理由见 docs/技术路线修订说明.md。

S4(b)/S5 之后这里除了播命令，还要播**反馈类**音频：

- ``ACK_DONE.wav``    动作已下发并收到回执；
- ``ACK_FAIL.wav``    动作下发失败；
- ``REJECT_UNKNOWN.wav`` 拒识（不属于任何命令）后播报可用命令；
- ``CLARIFY_<A>_<B>.wav`` 黄区澄清的二选一问句；
- ``PRETRIGGER.wav``  Pass-1 预触发提示音。

这些音频由 ``scripts/build_mandarin_audio.py`` 生成（需要 edge-tts）。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, Sequence

from .commands import COMMANDS, spoken_text

ENGINEERING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = ENGINEERING_ROOT / 'data' / 'audio' / 'mandarin'

# 反馈类音频：文件名 -> 播报文本（mock 模式打印用；TTS 生成时也用它）
# ACK_DONE 用「已完成」而不是「好的，已经完成」：对话完成延迟的验收预算是
# 1.2 s，动作延迟已经要占掉 ~150 ms，确认语再长就顶不住了。
FEEDBACK_TEXT = {
    'ACK_DONE': '已完成',
    'ACK_FAIL': '执行失败，请再说一次',
    'REJECT_UNKNOWN': '没听懂，请说：打开灯、关灯、打开窗帘、关上窗帘、全部关闭、查询设备状态',
}


def _runtime(name: str):
    root = str(ENGINEERING_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f'runtime.{name}')


class MandarinSpeaker:
    """按命令播放预录的普通话指令音频。"""

    def __init__(self, audio_dir=DEFAULT_AUDIO_DIR, wake_word: str = '',
                 mock: bool = False, device=None):
        self.audio_dir = Path(audio_dir)
        self.wake_word = wake_word
        self.mock = mock
        self.device = device
        self._player = None

    # ------------------------------------------------------------------
    def path_for(self, command: str) -> Path:
        return self.audio_dir / f'{command}.wav'

    def asset_path(self, asset: str) -> Path:
        return self.audio_dir / f'{asset}.wav'

    def available(self, command: str) -> bool:
        return self.path_for(command).exists()

    def available_asset(self, asset: str) -> bool:
        return self.asset_path(asset).exists()

    # ------------------------------------------------------------------
    def speak(self, command: str) -> bool:
        """播放命令对应的普通话音频，返回是否真的播了。"""
        if command not in COMMANDS:
            return False
        text = spoken_text(command, self.wake_word)
        if self.mock:
            print(f'[mock 播报] {text}')
            return True
        path = self.path_for(command)
        if not path.exists():
            print(f'[播报缺失] {command} -> {path}，先运行 scripts/build_mandarin_audio.py')
            return False
        self._play(path)
        print(f'[播报] {text}')
        return True

    # ------------------------------------------------------------------
    def speak_done(self, command: str = '') -> bool:
        """动作完成后的确认。"""
        return self._speak_asset('ACK_DONE', FEEDBACK_TEXT['ACK_DONE'])

    def speak_error(self, command: str = '') -> bool:
        """动作下发失败的提示。"""
        return self._speak_asset('ACK_FAIL', FEEDBACK_TEXT['ACK_FAIL'])

    def speak_reject(self) -> bool:
        """拒识后播报可用命令。"""
        return self._speak_asset('REJECT_UNKNOWN', FEEDBACK_TEXT['REJECT_UNKNOWN'])

    def speak_clarify(self, candidates: Sequence[str]) -> bool:
        """黄区澄清：把两个候选摆出来让用户二选一。"""
        clarify_mod = _runtime('dialog')
        asset = clarify_mod.clarify_asset(candidates)
        names = [clarify_mod.COMMAND_ZH.get(str(c), str(c)) for c in candidates]
        text = (f'你是要{names[0]}，还是{names[1]}？' if len(names) >= 2
                else '请再说一次')
        return self._speak_asset(asset, text)

    def speak_pretrigger(self, command: str = '') -> bool:
        """Pass-1 预触发提示音（只给反馈，不播命令，避免与 ACK 播报抢麦）。"""
        return self._speak_asset('PRETRIGGER', '')

    # ------------------------------------------------------------------
    def _speak_asset(self, asset: str, text: str) -> bool:
        if self.mock:
            print(f'[mock 播报] {text or asset}')
            return True
        path = self.asset_path(asset)
        if not path.exists():
            print(f'[播报缺失] {asset} -> {path}，先运行 scripts/build_mandarin_audio.py')
            return False
        self._play(path)
        if text:
            print(f'[播报] {text}')
        return True

    # ------------------------------------------------------------------
    def _play(self, path: Path) -> None:
        if self._player is None:
            self._player = _make_player(self.device)
        self._player(path)


def _make_player(device=None):
    """返回一个可调用对象：给定 wav 路径即播放。优先 sounddevice。"""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            '播放需要 sounddevice 与 soundfile: pip install sounddevice soundfile'
        ) from exc

    def play(path: Path) -> None:
        audio, sr = sf.read(str(path), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sd.play(audio, samplerate=sr, device=device)
        sd.wait()

    return play

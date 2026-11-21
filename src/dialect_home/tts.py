"""语音反馈：pyttsx3 离线优先，edge-tts 在线可选。"""
from __future__ import annotations

import threading


class Tts:
    def __init__(self, enabled: bool = True, engine: str = "pyttsx3",
                 edge_voice: str = "zh-CN-XiaoxiaoNeural"):
        self.enabled = enabled
        self.engine_kind = engine
        self.edge_voice = edge_voice
        self._engine = None
        self._lock = threading.Lock()

    def _get_pyttsx3(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 180)
            for voice in self._engine.getProperty("voices"):
                if "zh" in (voice.languages or [""])[0].lower() or "chinese" in voice.name.lower():
                    self._engine.setProperty("voice", voice.id)
                    break
        return self._engine

    def speak(self, text: str, block: bool = False) -> None:
        if not self.enabled or not text:
            return
        if self.engine_kind == "edge":
            target = self._speak_edge
        else:
            target = self._speak_local
        if block:
            target(text)
        else:
            threading.Thread(target=target, args=(text,), daemon=True).start()

    def _speak_local(self, text: str) -> None:
        try:
            with self._lock:
                engine = self._get_pyttsx3()
                engine.say(text)
                engine.runAndWait()
        except Exception as exc:  # 无声卡/无语音包时降级为静默
            print(f"[tts] 离线语音不可用，跳过播报: {exc}")

    def _speak_edge(self, text: str) -> None:
        try:
            import asyncio
            import tempfile

            import edge_tts

            async def _gen():
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                    path = fp.name
                await edge_tts.Communicate(text, self.edge_voice).save(path)
                return path

            path = asyncio.run(_gen())
            self._play_mp3(path)
        except Exception as exc:
            print(f"[tts] 在线语音失败，回退离线: {exc}")
            self._speak_local(text)

    @staticmethod
    def _play_mp3(path: str) -> None:
        import subprocess

        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nodisp", "-autoexit",
                        "-loglevel", "quiet", path], check=False)
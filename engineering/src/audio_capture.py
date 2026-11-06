# -*- coding: utf-8 -*-
"""麦克风采集模块。"""
import queue
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioCapture:
    def __init__(self, samplerate=16000, channels=1, blocksize=1024):
        if sd is None:
            raise ImportError("请先安装 sounddevice: pip install sounddevice")
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.q = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time, status):
        if status:
            print("audio status:", status)
        self.q.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self.stream.start()

    def read(self, timeout=1.0):
        return self.q.get(timeout=timeout)

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def list_devices():
    if sd is None:
        raise ImportError("请先安装 sounddevice")
    print(sd.query_devices())

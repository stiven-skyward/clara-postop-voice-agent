"""Silero VAD v5 (ONNX) en streaming: detecta fin de habla en el servidor.

Consume chunks de 512 muestras float32 @16 kHz (32 ms). Mantiene estado
recurrente; expone eventos: speech_start, speech_end (con el audio completo).
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from app import config

SR = 16000
CHUNK = 512  # muestras por inferencia (exigido por silero v5 a 16 kHz)


class StreamingVAD:
    def __init__(self) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(
            str(config.VAD_MODEL), opts, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(64, dtype=np.float32)  # contexto exigido por silero v5
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech: list[np.ndarray] = []
        self._pre: list[np.ndarray] = []          # pre-roll ~300 ms
        self._in_speech = False
        self._silence_ms = 0.0

    def _prob(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self._ctx, chunk])[None, :]
        out, self._state = self.sess.run(
            None,
            {"input": x, "state": self._state,
             "sr": np.array(SR, dtype=np.int64)},
        )
        self._ctx = chunk[-64:]
        return float(out[0, 0])

    def feed(self, samples: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
        """Alimenta audio; devuelve eventos [('speech_start', None) | ('speech_end', audio)]."""
        events: list[tuple[str, np.ndarray | None]] = []
        self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
        while self._buf.size >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            p = self._prob(chunk)
            if self._in_speech:
                self._speech.append(chunk)
                if p < config.VAD_THRESHOLD - 0.15:
                    self._silence_ms += 1000 * CHUNK / SR
                    if self._silence_ms >= config.VAD_SILENCE_MS:
                        audio = np.concatenate(self._pre + self._speech)
                        events.append(("speech_end", audio))
                        self._speech, self._pre = [], []
                        self._in_speech = False
                        self._silence_ms = 0.0
                else:
                    self._silence_ms = 0.0
            else:
                self._pre.append(chunk)
                if len(self._pre) > 10:
                    self._pre.pop(0)
                if p >= config.VAD_THRESHOLD:
                    self._in_speech = True
                    self._silence_ms = 0.0
                    self._speech = []
                    events.append(("speech_start", None))
        return events

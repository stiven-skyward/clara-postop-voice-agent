"""STT con whisper.cpp (pywhispercpp), CPU-only, español."""
from __future__ import annotations

import threading

import numpy as np

from app import config

_lock = threading.Lock()
_model = None


def get_model():
    global _model
    if _model is None:
        from pywhispercpp.model import Model
        _model = Model(
            str(config.STT_MODEL),
            n_threads=config.STT_THREADS,
            print_progress=False,
            print_realtime=False,
            redirect_whispercpp_logs_to=None,
        )
    return _model


def transcribe(audio_f32_16k: np.ndarray) -> str:
    """Transcribe audio mono float32 @16 kHz a texto en español."""
    if audio_f32_16k.size < 1600:  # <0.1 s
        return ""
    m = get_model()
    with _lock:  # whisper.cpp context no es thread-safe
        segments = m.transcribe(
            audio_f32_16k,
            language="es",
            initial_prompt=config.STT_PROMPT,
            translate=False,
        )
    text = " ".join(s.text.strip() for s in segments).strip()
    # whisper alucina texto en silencios: filtra marcas típicas
    if text.lower() in ("", "gracias.", "subtítulos por la comunidad de amara.org",
                        "¡gracias por ver el vídeo!", "[música]", "(música)"):
        return ""
    return text

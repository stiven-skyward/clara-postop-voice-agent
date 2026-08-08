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


def _enhance(audio: np.ndarray) -> np.ndarray:
    """Acondicionamiento para condiciones no ideales:
    - remoción de componente DC (micrófonos baratos con offset);
    - normalización de ganancia a pico 0.9 (voces débiles, micrófono lejano).
    No se amplifica lo que es prácticamente silencio (evita subir ruido puro)."""
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) or 1.0
    if peak < 0.05:
        return audio
    return np.clip(audio * (0.9 / peak), -1.0, 1.0)


def transcribe(audio_f32_16k: np.ndarray, context: str | None = None) -> str:
    """Transcribe audio mono float32 @16 kHz a texto en español.

    `context` (la última pregunta del agente) sesga el decodificador de whisper
    hacia el vocabulario esperado: si se preguntó por "hinchazón", una respuesta
    corta y confusa se resuelve hacia esos términos.
    """
    if audio_f32_16k.size < 1600:  # <0.1 s
        return ""
    audio_f32_16k = _enhance(audio_f32_16k)
    prompt = config.STT_PROMPT
    if context:
        prompt = f"{prompt} Agente: {context[:160]}"
    m = get_model()
    with _lock:  # whisper.cpp context no es thread-safe
        segments = m.transcribe(
            audio_f32_16k,
            language="es",
            initial_prompt=prompt,
            translate=False,
        )
    text = " ".join(s.text.strip() for s in segments).strip()
    # whisper alucina texto en silencios: filtra marcas típicas
    if text.lower() in ("", "gracias.", "subtítulos por la comunidad de amara.org",
                        "¡gracias por ver el vídeo!", "[música]", "(música)"):
        return ""
    return text

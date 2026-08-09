"""STT con whisper.cpp (pywhispercpp), CPU-only, español."""
from __future__ import annotations

import re
import threading
import unicodedata

import numpy as np

from app import config

_lock = threading.Lock()
_model = None


_load_lock = threading.Lock()


def get_model():
    global _model
    if _model is not None:
        return _model
    with _load_lock:  # double-checked: evita cargar dos veces el modelo
        if _model is not None:
            return _model
        from pywhispercpp.model import Model
        _model = Model(
            str(config.STT_MODEL),
            n_threads=config.STT_THREADS,
            print_progress=False,
            print_realtime=False,
            redirect_whispercpp_logs_to=None,
        )
    return _model


_HP_SOS = None


def _enhance(audio: np.ndarray) -> np.ndarray:
    """Acondicionamiento para condiciones no ideales:
    - paso-alto a 90 Hz: quita el retumbe (ventiladores, golpes de mesa, el
      propio cuerpo del micrófono) que en micrófonos de portátil llega a
      dominar el 66 % de la energía y enmascara la voz;
    - normalización de ganancia (voces débiles o micrófono lejano);
    - limitador suave en vez de recorte duro, que introduce distorsión.
    """
    global _HP_SOS
    audio = audio - float(np.mean(audio))
    if audio.size > 64:
        try:
            from scipy import signal
            if _HP_SOS is None:
                _HP_SOS = signal.butter(4, 90, btype="high", fs=16000, output="sos")
            audio = signal.sosfiltfilt(_HP_SOS, audio.astype(np.float64)).astype(np.float32)
        except Exception:
            pass
    peak = float(np.max(np.abs(audio))) or 1.0
    if peak < 0.02:          # prácticamente silencio: no amplificar ruido puro
        return audio
    # normaliza a 0.85 y comprime picos con tanh (evita el recorte cuadrado)
    g = 0.85 / peak
    return np.tanh(audio * g * 1.1).astype(np.float32)


def _norm_basura(s: str) -> str:
    """Normaliza para comparar con la lista de alucinaciones típicas."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w ]", "", s).strip()


_BASURA = {
    "", "gracias", "subtitulos por la comunidad de amaraorg",
    "gracias por ver el video", "musica", "silencio", "silence",
    "subtitulos realizados por la comunidad de amaraorg", "suscribete",
    "amaraorg", "aplausos", "risas", "ruido", "subtitulado por la comunidad",
    "mas informacion en wwwamaraorg", "gracias por su atencion",
}

# velocidad de habla imposible: por encima de esto, whisper está "continuando"
# el prompt en vez de transcribir (alucinación típica con audio corto o ruidoso)
_MAX_CAR_POR_SEG = 25.0


def transcribe(audio_f32_16k: np.ndarray, context: str | None = None) -> str:
    """Transcribe audio mono float32 @16 kHz a texto en español.

    Defensas contra la alucinación de whisper (que en audio corto o con ruido
    inventa frases largas y plausibles, p. ej. «seis» → «saber si el hombre
    está en el círculo»):
      1. El prompt de contexto solo se usa con audio suficientemente largo.
      2. Decodificación golosa (temperature 0) y umbrales estrictos.
      3. Relleno de silencio: whisper es inestable por debajo de ~1 s.
      4. Rechazo por velocidad de habla imposible.
      5. Lista de frases basura típicas.
    """
    if audio_f32_16k.size < 1600:  # <0.1 s
        return ""
    audio = _enhance(audio_f32_16k)
    dur = len(audio) / 16000

    # (3) whisper trabaja mejor con al menos ~1 s de señal
    if dur < 1.0:
        audio = np.concatenate([audio, np.zeros(int(16000 * (1.0 - dur)), np.float32)])

    # (1) con respuestas cortas ("seis", "sí") el prompt largo domina y alucina
    if dur >= 1.6:
        prompt = config.STT_PROMPT + (f" Agente: {context[:140]}" if context else "")
    elif dur >= 0.8:
        prompt = config.STT_PROMPT_CORTO
    else:
        prompt = ""

    m = get_model()
    with _lock:  # whisper.cpp context no es thread-safe
        segments = m.transcribe(
            audio,
            language="es",
            initial_prompt=prompt,
            translate=False,
            temperature=0.0,          # (2) sin muestreo aleatorio
            temperature_inc=0.0,      # sin escalada de temperatura al fallar
            entropy_thold=2.2,
            logprob_thold=-0.8,
            no_speech_thold=0.5,
            suppress_nst=True,        # suprime tokens que no son de habla
        )
    text = " ".join(s.text.strip() for s in segments).strip()

    if _norm_basura(text) in _BASURA or not text.strip(" .…"):
        return ""
    # (4) más caracteres de los que caben en el audio → es invención
    if len(text) / max(dur, 0.3) > _MAX_CAR_POR_SEG:
        return ""
    return text

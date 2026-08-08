"""TTS en español: Kokoro-82M (kokoro-onnx, voz ef_dora) o Piper (fallback ligero).

synthesize() devuelve (pcm_int16_bytes, sample_rate). El servidor lo envía por
WebSocket por oraciones para minimizar la latencia percibida.
"""
from __future__ import annotations

import re
import threading

import numpy as np

from app import config

_lock = threading.Lock()
_engine = None


class _KokoroEngine:
    sample_rate = 24000

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro
        self.k = Kokoro(str(config.KOKORO_MODEL), str(config.KOKORO_VOICES))

    def synth(self, text: str) -> tuple[bytes, int]:
        samples, sr = self.k.create(
            text, voice=config.KOKORO_VOICE, speed=config.TTS_SPEED, lang="es"
        )
        pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        return pcm.tobytes(), sr


class _PiperEngine:
    def __init__(self) -> None:
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(config.PIPER_VOICE))
        self.sample_rate = self.voice.config.sample_rate

    def synth(self, text: str) -> tuple[bytes, int]:
        chunks = []
        for ch in self.voice.synthesize(text):
            # piper>=1.3 entrega AudioChunk; versiones previas, bytes
            data = getattr(ch, "audio_int16_bytes", None) or (
                ch if isinstance(ch, (bytes, bytearray)) else b"")
            chunks.append(bytes(data))
        return b"".join(chunks), self.sample_rate


def get_engine():
    global _engine
    if _engine is None:
        if config.TTS_ENGINE == "kokoro":
            try:
                _engine = _KokoroEngine()
            except Exception:
                _engine = _PiperEngine()
        else:
            _engine = _PiperEngine()
    return _engine


_U = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho",
      "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis",
      "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós",
      "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete",
      "veintiocho", "veintinueve"]
_D = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta",
      8: "ochenta", 9: "noventa"}


def _n2w(n: int) -> str:
    """Número entero (0-999) a palabras en español."""
    if n < 30:
        return _U[n]
    if n < 100:
        d, u = divmod(n, 10)
        return _D[d] + (f" y {_U[u]}" if u else "")
    if n < 1000:
        c, r = divmod(n, 100)
        pre = {1: "ciento", 5: "quinientos", 7: "setecientos", 9: "novecientos"}.get(
            c, _U[c] + "cientos")
        if c == 1 and r == 0:
            return "cien"
        return pre + (f" {_n2w(r)}" if r else "")
    return str(n)


def _num_word(m: re.Match) -> str:
    """Expande '38.9' / '38,9' / '7' a palabras (la voz lee dígitos robótico)."""
    ent = int(m.group(1))
    dec = m.group(2)
    out = _n2w(ent) if ent < 1000 else m.group(1)
    if dec:
        out += " punto " + " ".join(_U[int(d)] for d in dec)
    return out


def _normalize(text: str) -> str:
    """Limpieza para voz: citas [n], símbolos, números a palabras y prosodia."""
    t = re.sub(r"\[(\d+)\]", "", text)              # las citas no se leen
    t = t.replace("°C", " grados").replace("º", " grados").replace("%", " por ciento")
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"(\d+)\s*/\s*10", lambda m: f"{_n2w(int(m.group(1)))} sobre diez", t)
    t = re.sub(r"\b(\d{1,3})[.,](\d{1,2})\b", _num_word, t)   # decimales
    t = re.sub(r"\b(\d{1,3})\b()", _num_word, t)              # enteros
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def synthesize(text: str) -> tuple[bytes, int]:
    t = _normalize(text)
    if not t:
        return b"", get_engine().sample_rate
    with _lock:
        return get_engine().synth(t)


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


class SentenceStreamer:
    """Acumula tokens del LLM y suelta oraciones completas para el TTS.

    Emite la PRIMERA oración en cuanto existe (aunque lleguen varias de golpe):
    la latencia percibida la define el primer audio, no el último.
    """

    def __init__(self, min_chars: int = 25) -> None:
        self.buf = ""
        self.min_chars = min_chars

    def push(self, token: str) -> list[str]:
        self.buf += token
        parts = _SENT_SPLIT.split(self.buf)
        out: list[str] = []
        while len(parts) > 1 and len(parts[0]) >= self.min_chars:
            out.append(parts.pop(0).strip())
        self.buf = " ".join(parts) if len(parts) > 1 else parts[0] if parts else ""
        return out

    def flush(self) -> str | None:
        out, self.buf = self.buf.strip(), ""
        return out or None


_phrase_cache: dict[str, tuple[bytes, int]] = {}


def synthesize_cached(text: str) -> tuple[bytes, int]:
    """Como synthesize(), con caché para frases fijas (saludo, escalamiento)."""
    key = text.strip()
    if key not in _phrase_cache:
        _phrase_cache[key] = synthesize(key)
    return _phrase_cache[key]

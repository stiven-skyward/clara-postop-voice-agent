"""Acondicionamiento de audio de entrada, calidad profesional.

El navegador entrega audio a la frecuencia nativa de la tarjeta (típicamente
44,1 o 48 kHz). Bajarlo a los 16 kHz que necesita el reconocedor **quitando
muestras** produce *aliasing*: las frecuencias por encima de 8 kHz se pliegan
sobre la banda de la voz y destruyen justamente las consonantes fricativas
(f, s, ch) que distinguen «fiebre» de «nombre».

Aquí se hace bien: filtro paso-bajo Butterworth con estado (funciona sobre un
flujo continuo, sin discontinuidades entre bloques) seguido de interpolación
con fase acumulada.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

DESTINO = 16000


class StreamingResampler:
    """Remuestrea un flujo continuo a 16 kHz conservando el estado del filtro."""

    def __init__(self, origen: int) -> None:
        self.origen = int(origen)
        self.ratio = self.origen / DESTINO
        self.pos = 0.0
        self.cola = np.zeros(0, dtype=np.float32)
        if self.origen > DESTINO:
            # corte al 90 % de Nyquist del destino (7,2 kHz): conserva toda la
            # banda útil de la voz y elimina lo que provocaría aliasing
            self.sos = signal.butter(8, 7200, btype="low", fs=self.origen,
                                     output="sos").astype(np.float64)
            self.zi = signal.sosfilt_zi(self.sos) * 0.0
        else:
            self.sos = None
            self.zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        if self.origen == DESTINO:
            return x
        if self.sos is not None:
            y, self.zi = signal.sosfilt(self.sos, x.astype(np.float64), zi=self.zi)
            x = y.astype(np.float32)
        # interpolación lineal con fase continua entre bloques
        buf = np.concatenate([self.cola, x])
        if buf.size < 2:
            self.cola = buf
            return np.zeros(0, dtype=np.float32)
        idx = np.arange(self.pos, buf.size - 1, self.ratio, dtype=np.float64)
        if idx.size == 0:
            self.cola = buf
            return np.zeros(0, dtype=np.float32)
        out = np.interp(idx, np.arange(buf.size), buf).astype(np.float32)
        consumidas = int(np.floor(idx[-1] + self.ratio))
        consumidas = min(consumidas, buf.size)
        self.pos = (idx[-1] + self.ratio) - consumidas
        # se conserva 1 muestra de solape para que la interpolación no salte
        self.cola = buf[max(0, consumidas - 1):]
        self.pos += min(1, consumidas)  # compensa la muestra conservada
        self.pos = max(0.0, self.pos)
        return out


def calidad(audio16k: np.ndarray) -> dict:
    """Métricas de calidad del audio recibido, para diagnóstico en los logs."""
    if audio16k.size < 800:
        return {"duracion_s": round(audio16k.size / DESTINO, 2)}
    a = audio16k - float(np.mean(audio16k))
    S = np.abs(np.fft.rfft(a * np.hanning(a.size))) ** 2
    fr = np.fft.rfftfreq(a.size, 1 / DESTINO)

    def banda(f0: float, f1: float) -> float:
        m = (fr >= f0) & (fr < f1)
        return float(np.sum(S[m]))

    total = banda(0, DESTINO / 2) or 1.0
    return {
        "duracion_s": round(a.size / DESTINO, 2),
        "rms": round(float(np.sqrt(np.mean(a ** 2))), 4),
        "pico": round(float(np.max(np.abs(a))), 3),
        "pct_voz_300_3k": round(100 * banda(300, 3000) / total, 1),
        "pct_agudos_3k_7k": round(100 * banda(3000, 7000) / total, 1),
        "pct_retumbe_0_300": round(100 * banda(0, 300) / total, 1),
    }

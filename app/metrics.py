"""Métricas obligatorias (rúbrica §5), loggeadas por turno a JSONL verificable.

Latencia = desde que el paciente termina de hablar (cierre del VAD) hasta que
empieza a sonar el audio del agente (primer chunk TTS enviado al navegador).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median

from app import config


class TurnMetrics:
    def __init__(self, call_id: str, turno: int) -> None:
        self.d: dict = {"call_id": call_id, "turno": turno, "ts": time.time()}
        self._t0: float | None = None

    def mark(self, name: str) -> None:
        """Marca temporal (speech_end, stt_done, llm_first_token, tts_first_audio, turn_done)."""
        now = time.perf_counter()
        if name == "speech_end":
            self._t0 = now
            self.d["t_speech_end"] = 0.0
        elif self._t0 is not None:
            self.d[f"t_{name}"] = round(now - self._t0, 3)

    def set(self, **kw) -> None:
        self.d.update(kw)

    def save(self) -> None:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.LOGS_DIR / "turnos.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(self.d, ensure_ascii=False) + "\n")


def log_call_summary(call_id: str, data: dict) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LOGS_DIR / "llamadas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"call_id": call_id, "ts": time.time(), **data},
                           ensure_ascii=False) + "\n")


def log_alert(call_id: str, data: dict) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LOGS_DIR / "alertas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"call_id": call_id, "ts": time.time(), **data},
                           ensure_ascii=False) + "\n")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    idx = min(len(vs) - 1, int(round(p / 100 * (len(vs) - 1))))
    return vs[idx]


def aggregate() -> dict:
    """Resumen P50/P95 y consumos a partir de los logs (para README y /metrics)."""
    path = Path(config.LOGS_DIR / "turnos.jsonl")
    if not path.exists():
        return {"turnos": 0}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    lat = [r["t_tts_first_audio"] for r in rows if "t_tts_first_audio" in r]
    tin = [r.get("tokens_in", 0) for r in rows]
    tout = [r.get("tokens_out", 0) for r in rows]
    calls = {}
    for r in rows:
        calls.setdefault(r["call_id"], []).append(r)
    return {
        "turnos": len(rows),
        "llamadas": len(calls),
        "latencia_p50_s": round(median(lat), 2) if lat else None,
        "latencia_p95_s": round(_percentile(lat, 95), 2) if lat else None,
        "tokens_entrada_por_turno_prom": round(sum(tin) / len(rows), 1) if rows else 0,
        "tokens_salida_por_turno_prom": round(sum(tout) / len(rows), 1) if rows else 0,
        "invocaciones_llm_por_turno_prom": round(
            sum(r.get("llm_calls", 0) for r in rows) / len(rows), 2) if rows else 0,
        "consultas_rag_por_llamada_prom": round(
            sum(sum(t.get("rag_queries", 0) for t in ts) for ts in calls.values())
            / max(1, len(calls)), 2),
        "tokens_por_llamada_prom": round(
            sum(sum(t.get("tokens_in", 0) + t.get("tokens_out", 0) for t in ts)
                for ts in calls.values()) / max(1, len(calls)), 1),
    }

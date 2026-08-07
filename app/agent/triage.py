"""Triaje determinista sobre síntomas estructurados.

El LLM NO decide el triaje: extrae síntomas a JSON (tarea en la que el 3B es
fiable) y este motor de reglas —derivado de los signos de alarma de las guías
del corpus— clasifica. Es determinista, auditable y sesgado a la seguridad:
ante la duda sube de nivel, y el nivel solo puede subir durante la llamada
(nunca bajar). Diseño elegido tras comprobar empíricamente que el 3B con
salida restringida subestima casos rojos (ver docs/DECISIONES-ARQUITECTURA.md).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

LEVELS = ["verde", "amarillo", "rojo"]

_RED_TEXT = [
    "dolor en el pecho", "dolor toracico", "no puedo respirar", "confusion",
    "confundid", "desmay", "convulsion", "sangrado abundante", "mucha sangre",
    "labios morados", "pantorrilla hinchada", "pierna hinchada y caliente",
    "pus", "se abrio la herida", "herida abierta", "vomito con sangre",
    "no orino", "sin orinar",
]
_YELLOW_TEXT = [
    "escalofrio", "pantorrilla", "mareo", "vomito", "diarrea", "estreni",
    "no he podido comer", "sin apetito", "no duermo", "hormigueo",
    "enrojecimiento", "se extiende", "hinchado",
]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


@dataclass
class SymptomState:
    """Estado acumulado de síntomas durante la llamada (merge por turno)."""
    dolor_nrs: float | None = None
    fiebre_c: float | None = None
    fiebre_subjetiva: bool | None = None
    herida: str | None = None
    dolor_empeora: bool | None = None
    sangrado: bool | None = None
    disnea: bool | None = None
    vomito_persistente: bool | None = None
    otros: list[str] = field(default_factory=list)

    def merge(self, ext: dict) -> None:
        # plausibilidad clínica: una "temperatura" fuera de 34-43 °C es casi
        # seguro un error de transcripción → se descarta el número y se marca
        # fiebre subjetiva para que el agente la confirme preguntando
        fc = ext.get("fiebre_c")
        if fc is not None and not (34.0 <= float(fc) <= 43.0):
            ext = {**ext, "fiebre_c": None, "fiebre_subjetiva": True}
        for k in ("dolor_nrs", "fiebre_c", "fiebre_subjetiva", "herida",
                  "dolor_empeora", "sangrado", "disnea", "vomito_persistente"):
            v = ext.get(k)
            if v is not None:
                # la herida solo empeora en gravedad dentro de la llamada
                if k == "herida" and self.herida in ("secrecion_purulenta", "abierta"):
                    order = ["normal", "enrojecida", "secrecion_clara",
                             "secrecion_purulenta", "abierta"]
                    if order.index(v) < order.index(self.herida):
                        continue
                setattr(self, k, v)
        for o in ext.get("otros", []) or []:
            if o and o not in self.otros:
                self.otros.append(o)

    def as_dict(self) -> dict:
        return {
            "dolor_nrs": self.dolor_nrs, "fiebre_c": self.fiebre_c,
            "fiebre_subjetiva": self.fiebre_subjetiva, "herida": self.herida,
            "dolor_empeora": self.dolor_empeora, "sangrado": self.sangrado,
            "disnea": self.disnea, "vomito_persistente": self.vomito_persistente,
            "otros": self.otros,
        }


@dataclass
class TriageResult:
    nivel: str
    razones: list[str]
    faltantes: list[str]   # qué indagar antes de decidir (ambigüedad → preguntar)


def evaluate(s: SymptomState, procedimiento: str = "", dia_postop: int = 1) -> TriageResult:
    razones_rojo: list[str] = []
    razones_amarillo: list[str] = []
    faltantes: list[str] = []

    # --- Fiebre ---
    if s.fiebre_c is not None:
        if s.fiebre_c >= 38.5:
            razones_rojo.append(f"Fiebre de {s.fiebre_c:g} °C (umbral de alarma ≥38.5)")
        elif s.fiebre_c >= 38.0:
            razones_amarillo.append(f"Fiebre de {s.fiebre_c:g} °C (38.0–38.4: vigilancia)")
    elif s.fiebre_subjetiva:
        razones_amarillo.append("Refiere sensación febril sin medición")
        faltantes.append("temperatura medida con termómetro")

    # --- Dolor ---
    if s.dolor_nrs is not None:
        if s.dolor_nrs >= 8:
            razones_rojo.append(f"Dolor {s.dolor_nrs:g}/10 (severo)")
        elif s.dolor_nrs >= 5:
            if s.dolor_empeora:
                razones_rojo.append(f"Dolor {s.dolor_nrs:g}/10 en aumento")
            else:
                razones_amarillo.append(f"Dolor {s.dolor_nrs:g}/10 (moderado)")
        elif s.dolor_empeora and dia_postop >= 3:
            razones_amarillo.append("Dolor que empeora pasado el día 3")
    elif s.dolor_empeora:
        razones_amarillo.append("Dolor en aumento sin cuantificar")
        faltantes.append("intensidad del dolor de 0 a 10")

    # --- Herida ---
    if s.herida == "abierta":
        razones_rojo.append("Herida abierta (dehiscencia)")
    elif s.herida == "secrecion_purulenta":
        razones_rojo.append("Secreción purulenta o de mal olor en la herida")
    elif s.herida == "secrecion_clara":
        razones_amarillo.append("Secreción clara en la herida")
    elif s.herida == "enrojecida":
        razones_amarillo.append("Enrojecimiento de la herida")

    # --- Signos sistémicos ---
    if s.sangrado:
        razones_rojo.append("Sangrado activo")
    if s.disnea:
        razones_rojo.append("Dificultad para respirar")
    if s.vomito_persistente:
        razones_rojo.append("Vómito persistente")

    # --- Texto libre (regionalismos y síntomas no estructurados) ---
    otros = _norm(" ".join(s.otros))
    for kw in _RED_TEXT:
        if kw in otros:
            razones_rojo.append(f"Signo de alarma referido: «{kw}»")
    for kw in _YELLOW_TEXT:
        if kw in otros and not any(kw in r for r in razones_rojo):
            razones_amarillo.append(f"Síntoma a vigilar: «{kw}»")

    # --- Combinaciones (fiebre + herida = infección probable) ---
    if (s.fiebre_c and s.fiebre_c >= 38.0 or s.fiebre_subjetiva) and \
            s.herida in ("enrojecida", "secrecion_clara", "secrecion_purulenta"):
        razones_rojo.append("Fiebre + alteración de la herida: posible infección del sitio quirúrgico")

    # DVT tras reemplazo articular: pantorrilla es rojo, no amarillo
    if "articular" in _norm(procedimiento) or "cadera" in _norm(procedimiento) \
            or "rodilla" in _norm(procedimiento):
        if "pantorrilla" in otros or "pierna hinchada" in otros:
            razones_rojo.append("Dolor/hinchazón de pantorrilla tras reemplazo articular: descartar trombosis")

    if razones_rojo:
        return TriageResult("rojo", razones_rojo, [])
    if razones_amarillo:
        return TriageResult("amarillo", razones_amarillo, faltantes)
    return TriageResult("verde", ["Sin signos de alarma en lo reportado"], faltantes)


_QUICK_RED = _RED_TEXT + [
    "huele feo", "huele mal", "mal olor", "liquido amarillo", "liquidito amarillo",
    "liquido verde", "no aguanto el dolor", "dolor insoportable",
]


def quick_red_scan(text: str) -> str | None:
    """Barrido léxico instantáneo (sin LLM) de señales rojas en el turno crudo.
    Permite responder el escalamiento en <1 s; la extracción formal corre después
    para el registro. Devuelve la señal detectada o None."""
    tn = _norm(text)
    for kw in _QUICK_RED:
        if kw in tn:
            return kw
    return None


def combine(previous: str, new: str) -> str:
    """El nivel de la llamada solo puede subir (asimetría clínica)."""
    return LEVELS[max(LEVELS.index(previous), LEVELS.index(new))]

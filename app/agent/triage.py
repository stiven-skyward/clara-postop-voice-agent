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
# Solo señales con valor clínico real; las molestias normales de recuperación
# (sueño irregular, poco apetito, mareo leve) NO suben el nivel: se registran
# en el resumen pero no disparan vigilancia. Calibrado contra el dataset del
# reto (la v1 convertía todos los casos verdes en amarillo/rojo).
_YELLOW_TEXT = [
    "escalofrio", "pantorrilla", "se extiende", "se esta extendiendo",
    "no he podido comer nada", "vomite varias veces", "muchos vomitos",
]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Anclaje léxico: una bandera booleana de la extracción solo se acepta si el
# texto del paciente contiene un lexema compatible. Neutraliza la tendencia del
# LLM pequeño a inferir síntomas no dichos ("no baja" → dolor_empeora, etc.).
_GROUNDING = {
    "dolor_empeora": r"empeor|aument|subiendo|sube|mas fuerte|más fuerte|peor|creciendo",
    "sangrado": r"sangr",
    "disnea": r"respir|ahog|aire|asfixi",
    "vomito_persistente": r"vomit|devolv|trasboc",
    "fiebre_subjetiva": r"fiebre|calentur|temperatura|destempl|febril|caliente|acalorad",
}

# Anclaje para campos enum: el valor solo se acepta si el texto contiene un
# lexema del síntoma DEGRADADO (evita que "dormí regular pero bien" → alterado)
_GROUNDING_ENUM = {
    "apetito": r"apetito|hambre|desgano|ganas|no he comido|como poc|comer poc",
    "sueno": r"no duermo|duermo poc|duermo mal|despiert|desvel|insomn|no (?:he )?podido dormir|casi no duermo|trasnoch|dormido mal",
}


def sanitize_extraction(ext: dict, user_text: str) -> dict:
    tn = _norm(user_text)
    out = dict(ext)
    for field_name, pattern in _GROUNDING.items():
        if out.get(field_name) is True and not re.search(pattern, tn):
            out.pop(field_name)
    for field_name, pattern in _GROUNDING_ENUM.items():
        if out.get(field_name) and not re.search(pattern, tn):
            out.pop(field_name)
    # el paciente habla del dolor pero no lo cuantifica → señal blanda de evasión
    if out.get("dolor_nrs") is None and re.search(r"dolor|duele|molest|adolori", tn):
        out["dolor_mencionado_sin_numero"] = True
    return out


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
    apetito: str | None = None
    sueno: str | None = None
    dolor_mencionado_sin_numero: bool = False
    otros: list[str] = field(default_factory=list)

    def merge(self, ext: dict) -> None:
        # plausibilidad clínica: una "temperatura" fuera de 34-43 °C es casi
        # seguro un error de transcripción → se descarta el número y se marca
        # fiebre subjetiva para que el agente la confirme preguntando
        fc = ext.get("fiebre_c")
        if fc is not None and not (34.0 <= float(fc) <= 43.0):
            ext = {**ext, "fiebre_c": None, "fiebre_subjetiva": True}
        for k in ("dolor_nrs", "fiebre_c", "fiebre_subjetiva", "herida",
                  "dolor_empeora", "sangrado", "disnea", "vomito_persistente",
                  "apetito", "sueno"):
            v = ext.get(k)
            if v is not None:
                # la herida solo empeora en gravedad dentro de la llamada
                if k == "herida" and self.herida in ("secrecion_purulenta", "abierta"):
                    order = ["normal", "enrojecida_leve", "enrojecida",
                             "secrecion_clara", "secrecion_purulenta", "abierta"]
                    if v in order and order.index(v) < order.index(self.herida):
                        continue
                setattr(self, k, v)
        if ext.get("dolor_mencionado_sin_numero"):
            self.dolor_mencionado_sin_numero = True
        if ext.get("dolor_nrs") is not None:
            self.dolor_mencionado_sin_numero = False  # ya lo cuantificó
        for o in ext.get("otros", []) or []:
            if o and o not in self.otros:
                self.otros.append(o)

    def as_dict(self) -> dict:
        return {
            "dolor_nrs": self.dolor_nrs, "fiebre_c": self.fiebre_c,
            "fiebre_subjetiva": self.fiebre_subjetiva, "herida": self.herida,
            "dolor_empeora": self.dolor_empeora, "sangrado": self.sangrado,
            "disnea": self.disnea, "vomito_persistente": self.vomito_persistente,
            "apetito": self.apetito, "sueno": self.sueno,
            "dolor_mencionado_sin_numero": self.dolor_mencionado_sin_numero,
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
        # sensación febril sin medición: por sí sola NO sube el nivel; el
        # agente debe indagar (pedir que se mida) antes de decidir
        faltantes.append("temperatura medida con termómetro")

    # --- Dolor ---
    if s.dolor_nrs is not None:
        if s.dolor_nrs >= 8:
            razones_rojo.append(f"Dolor {s.dolor_nrs:g}/10 (severo)")
        elif s.dolor_nrs >= 5:
            if s.dolor_empeora:
                razones_amarillo.append(f"Dolor {s.dolor_nrs:g}/10 en aumento")
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
        razones_amarillo.append("Enrojecimiento notorio de la herida")
    # enrojecida_leve NO sube el nivel por sí sola: cuenta como señal blanda

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
    # fiebre ≥38.5 o secreción purulenta ya son rojo por sí solas; la
    # combinación febrícula (38.0-38.4) + herida alterada leve refuerza el
    # amarillo (revisión el mismo día), no salta a rojo.
    fiebre_alguna = (s.fiebre_c is not None and s.fiebre_c >= 38.0) or s.fiebre_subjetiva
    if fiebre_alguna and s.herida in ("enrojecida", "secrecion_clara"):
        razones_amarillo.append("Fiebre/febrícula + herida alterada: vigilar posible infección del sitio quirúrgico")

    # DVT tras reemplazo articular: pantorrilla es rojo, no amarillo
    if "articular" in _norm(procedimiento) or "cadera" in _norm(procedimiento) \
            or "rodilla" in _norm(procedimiento):
        if "pantorrilla" in otros or "pierna hinchada" in otros:
            razones_rojo.append("Dolor/hinchazón de pantorrilla tras reemplazo articular: descartar trombosis")

    # --- Conteo de señales blandas: detector de minimización/incongruencia ---
    # Un paciente que resta importancia a todo ("un poquito", "uno aguanta")
    # no dispara ninguna regla dura, pero la degradación SIMULTÁNEA de varios
    # dominios es en sí misma una señal clínica (arquetipo minimizador del
    # dataset: trayectoria dolor 9 relatada como "un poquito molesto").
    blandas: list[str] = []
    if s.herida in ("enrojecida_leve", "enrojecida", "secrecion_clara"):
        blandas.append("herida alterada (aunque la describe leve)")
    if s.fiebre_subjetiva or (s.fiebre_c is not None and 37.8 <= s.fiebre_c < 38.0):
        blandas.append("sensación febril o febrícula")
    if s.apetito in ("reducido", "nulo"):
        blandas.append("apetito reducido")
    if s.sueno == "alterado":
        blandas.append("sueño alterado")
    if s.dolor_mencionado_sin_numero and s.dolor_nrs is None:
        blandas.append("dolor referido sin cuantificar")
        faltantes.append("intensidad del dolor de 0 a 10")

    if len(blandas) >= 4:
        razones_rojo.append(
            "Múltiples dominios afectados a la vez (" + "; ".join(blandas) +
            "): cuadro incongruente con la minimización del paciente")
    elif len(blandas) >= 3:
        razones_amarillo.append(
            "Varias molestias simultáneas (" + "; ".join(blandas) + ")")

    if razones_rojo:
        return TriageResult("rojo", razones_rojo, [])
    if razones_amarillo:
        return TriageResult("amarillo", razones_amarillo, faltantes)
    return TriageResult("verde", ["Sin signos de alarma en lo reportado"], faltantes)


_QUICK_RED = _RED_TEXT + [
    "huele feo", "huele mal", "mal olor", "liquido amarillo", "liquidito amarillo",
    "liquido verde", "amarillit", "verdos", "no aguanto el dolor",
    "dolor insoportable",
]


_NEGATION = r"(?:nada de|sin|no hay|no tengo|no me sale|ni|no veo|tampoco)\s+(?:\w+\s+){0,3}$"


def quick_red_scan(text: str) -> str | None:
    """Barrido léxico instantáneo (sin LLM) de señales rojas en el turno crudo.
    Permite responder el escalamiento en <1 s; la extracción formal corre después
    para el registro. Ignora menciones negadas ("nada de pus"). Devuelve la
    señal detectada o None."""
    tn = _norm(text)
    for kw in _QUICK_RED:
        i = tn.find(kw)
        if i >= 0 and not re.search(_NEGATION, tn[max(0, i - 30):i]):
            return kw
    return None


def combine(previous: str, new: str) -> str:
    """El nivel de la llamada solo puede subir (asimetría clínica)."""
    return LEVELS[max(LEVELS.index(previous), LEVELS.index(new))]

"""Detección de si el paciente ya respondió lo que se le preguntó.

Cuando el dato pedido (dolor, temperatura, herida…) llega completo, no hace
falta pagar extracción+conversación al LLM: se registra y se pasa al siguiente
punto del guion. También repara alucinaciones típicas de Whisper según el slot.
"""
from __future__ import annotations

import re
import unicodedata

from app.agent.triage import _palabras_a_numero, fallback_extract, quick_red_scan
from app.rag.lexicon import interpretar_numero_hablado

_REINTRO = re.compile(
    r"(?:^|(?<=\. ))(?:Hola\s+)?[A-ZÁÉÍÓÚÑÜa-záéíóúñü]+,?\s+le habla Clara\.\s*",
    re.IGNORECASE,
)

# Pregunta de Clara → slot del guion
_SLOT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("confirmacion", re.compile(r"qued[oó]\s+claro|me confirma", re.I)),
    ("dolor", re.compile(r"dolor|cero al diez|0 a 10|escala", re.I)),
    ("fiebre", re.compile(r"temperatura|fiebre|grados", re.I)),
    ("herida", re.compile(r"herida|enrojecimiento|secreci[oó]n|hinchaz", re.I)),
    ("movilidad", re.compile(r"moverse|caminar|movilidad", re.I)),
    ("apetito_sueno", re.compile(r"apetito|sue[nñ]o|dormir", re.I)),
    ("medicacion", re.compile(r"medicamento|pastilla|indicaciones del m[eé]dico", re.I)),
    ("cita", re.compile(r"volver a la cl[ií]nica|pr[oó]xima (?:cita|revisi[oó]n)|programado", re.I)),
    ("animo", re.compile(r"[aá]nimo|c[oó]mo se siente", re.I)),
]

SHORT_SLOTS = {"dolor", "confirmacion"}
FAST_SLOTS = {"dolor", "fiebre", "herida", "movilidad", "apetito_sueno"}

STT_PROMPTS = {
    "dolor": (
        "El dolor está en cero. El dolor está en uno. El dolor está en dos. "
        "El dolor está en tres. El dolor está en cuatro. El dolor está en cinco. "
        "El dolor está en seis. El dolor está en siete. El dolor está en ocho. "
        "El dolor está en nueve. El dolor está en diez."
    ),
    "fiebre": (
        "Tengo treinta y seis grados. Tengo treinta y siete grados. "
        "Tengo treinta y ocho grados. Tengo treinta y nueve grados. "
        "No he tenido fiebre. Temperatura normal. Treinta y siete."
    ),
    "herida": (
        "La herida se ve normal. Hay hinchazón. Hay enrojecimiento. "
        "Hay secreción. Un liquidito amarillo. Sin secreción."
    ),
    "movilidad": "Camino bien. Me duele un poco al caminar. No puedo caminar.",
    "apetito_sueno": "No tengo apetito. Como bien. Duermo bien. Duermo mal. He podido comer.",
    "medicacion": "Sí, me los sigo tomando. No, no me los estoy tomando.",
    "cita": "El próximo sábado. El lunes. La próxima semana. En tres días.",
    "animo": "Bien, animado. Regular. Mal, desanimado.",
    "cierre": "Gracias, adiós. Hasta luego. No, no tengo más preguntas. Chao. Eso es todo.",
}

PREGUNTAS = {
    "dolor": (
        "¿Cómo ha estado el dolor? Dígame un número del cero al diez, "
        "por ejemplo: el dolor está en cinco."
    ),
    "fiebre": (
        "¿Cuál es su temperatura actual? Dígame la cifra en una frase, "
        "por ejemplo: tengo treinta y siete grados."
    ),
    "herida": (
        "¿Cómo está la herida: normal, con enrojecimiento, secreción o hinchazón?"
    ),
    "movilidad": "¿Cómo le va al moverse o caminar?",
    "apetito_sueno": "¿Cómo le va al comer, ha tenido apetito desde la operación?",
}

_GARBAGE = re.compile(
    r"^(?:nda|n da|mm+|eh+|ah+|um+|este|pues|a ver)\.?$",
    re.IGNORECASE,
)
_SIN_FIEBRE = re.compile(
    r"(?:no|nada de|sin)\s+(?:\w+\s+){0,3}(?:fiebre|calentura)|"
    r"no he tenido fiebre|temperatura normal|sin fiebre",
    re.IGNORECASE,
)
_HERIDA = [
    (r"pus|huele feo|liquidito amarill|l[ií]quido amarill|secreci[oó]n purulent",
     "secrecion_purulenta"),
    (r"secreci[oó]n|supura|l[ií]quidito|l[ií]quido", "secrecion_clara"),
    (r"abierta|se abri[oó]|puntos sueltos", "abierta"),
    (r"hinch|hincha", "hinchada"),
    (r"enrojec|rojita|roja|rosadita", "enrojecida"),
    (r"normal|seca|bien|sin secreci", "normal"),
]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def slot_from_question(question: str | None) -> str | None:
    if not question:
        return None
    for slot, pat in _SLOT_PATTERNS:
        if pat.search(question):
            return slot
    return None


def stt_prompt_for_slot(slot: str | None) -> str | None:
    return STT_PROMPTS.get(slot or "")


def strip_reintro(text: str) -> str:
    """Quita «Mauricio, le habla Clara» si el 3B se vuelve a presentar."""
    prev = None
    out = text or ""
    while prev != out:
        prev = out
        out = _REINTRO.sub("", out, count=1).lstrip(" ,.")
    return out.strip()


def forzar_usted(text: str) -> str:
    reps = (
        (r"\bsigas\b", "siga"), (r"\btienes\b", "tiene"),
        (r"\bpuedes\b", "puede"), (r"\bdebes\b", "debe"),
        (r"\bsigues\b", "sigue"), (r"\bestás\b", "está"),
        (r"\btú\b", "usted"), (r"\bTu\b", "Su"), (r"\btu\b", "su"),
    )
    out = text
    for pat, repl in reps:
        out = re.sub(pat, repl, out)
    return out


def repair_for_slot(slot: str | None, text: str) -> str:
    """Corrige confusiones fonéticas típicas según lo que Clara acaba de pedir."""
    t = (text or "").strip()
    if not slot or not t:
        return t
    n = _norm(t)
    if slot == "cita":
        t = re.sub(r"\bsaludo\b", "sábado", t, flags=re.I)
        t = re.sub(r"^es el pr[oó]ximo", "el próximo", t, flags=re.I)
    elif slot == "medicacion":
        if re.search(r"cielo|cielo sigo|el cielo", n) and re.search(r"tom", n):
            return "Sí, me los sigo tomando."
        if re.search(r"\bs[ií]\b", n) and re.search(r"tom", n):
            return "Sí, me los sigo tomando."
    elif slot == "animo":
        if re.search(r"de la vida|animad|bien", n) and len(t.split()) <= 4:
            if "mal" in n:
                return "Mal, desanimado."
            return "Bien, animado."
    elif slot == "apetito_sueno":
        t = t.strip("¿?¡! ").strip()
        n = _norm(t)
        if re.fullmatch(r"c[oó]mo?\s+bien\.?", n) or n in {"bien", "bien.", "como bien"}:
            return "Como bien."
        if re.fullmatch(r"c[oó]mo?\s+mal\.?", n):
            return "Como mal."
    elif slot == "cierre":
        t = re.sub(r"\ba\s+dios\b", "adiós", t, flags=re.I)
        t = re.sub(r"\bhay dios\b", "adiós", t, flags=re.I)
        t = re.sub(r"\badios\b", "adiós", t, flags=re.I)
    elif slot == "herida":
        t = re.sub(r"\bhincha\s+(?:son|zon|cion|ción)\b", "hinchazón", t, flags=re.I)
        t = re.sub(r"\b(?:enchaz[oó]n|hinchazon)\b", "hinchazón", t, flags=re.I)
    elif slot == "fiebre":
        if _GARBAGE.match(t) or n in {"nda", "anda"}:
            return t  # el parser lo marcará unclear; no inventar 37
        if n in {"nada", "nada."}:
            return "No he tenido fiebre."
        temp = temperatura_de(t)
        if temp is not None and "grado" not in n and "fiebre" not in n:
            cifra = int(temp) if float(temp) == int(temp) else temp
            return f"Tengo {cifra} grados."
    return t


def temperatura_de(text: str) -> float | None:
    t = _palabras_a_numero(_norm(text))
    m = re.search(r"\b(3[5-9]|4[0-3])(?:[.,](\d))?\b", t)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2) or 0}")


def dolor_de(text: str) -> float | None:
    num = interpretar_numero_hablado(text)
    if num and re.fullmatch(r"\d{1,2}", num) and 0 <= int(num) <= 10:
        return float(int(num))
    if num:
        from app.agent.triage import _PALABRA_NUM
        n = _norm(num)
        if n in _PALABRA_NUM and _PALABRA_NUM[n] <= 10:
            return float(_PALABRA_NUM[n])
    ext = fallback_extract(text)
    if ext.get("dolor_nrs") is not None:
        return float(ext["dolor_nrs"])
    # «el dolor está en seis» / una sola palabra-número en contexto de escala
    t = _palabras_a_numero(_norm(text))
    m = re.search(r"\b(10|[0-9])\b", t)
    if m and not re.search(r"pastilla|hora|d[ií]a|grado", t):
        return float(m.group(1))
    return None


_PREGUNTA_REAL = re.compile(
    r"una pregunta|quiero saber|quisiera saber|c[oó]digo|"
    r"cu[aá]l es|cu[aá]ndo me|me puedo ba[nñ]ar|es normal que",
    re.IGNORECASE,
)
_HERIDA_NEG = re.compile(
    r"(?:nada de|sin|no hay|no tiene|no veo|ni|ningun[ao]s?)\s+(?:\w+\s+){0,3}$",
)


def herida_de(text: str) -> str | None:
    """Clasifica la herida ignorando síntomas negados («sin secreción»)."""
    tn = _norm(text)
    orden = ["normal", "enrojecida_leve", "enrojecida", "hinchada",
             "secrecion_clara", "secrecion_purulenta", "abierta"]
    worst = None
    for pat, val in _HERIDA:
        if val == "normal":
            continue
        for m in re.finditer(pat, tn):
            if _HERIDA_NEG.search(tn[max(0, m.start() - 28): m.start()]):
                continue
            if worst is None or orden.index(val) > orden.index(worst):
                worst = val
            break
    if worst:
        return worst
    if re.search(r"normal|seca|se ve bien|esta bien|sin (?:enrojec|secreci|hinch)", tn):
        return "normal"
    return None


def looks_unclear(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return True
    return bool(_GARBAGE.match(t))


_DESPEDIDA = re.compile(
    r"\b(?:adi[oó]s|chao|chau|hasta luego|nos vemos|eso es todo|"
    r"nada m[aá]s|ya est[aá]|puede colgar|cerremos|"
    r"no tengo (?:m[aá]s )?preguntas)\b"
    r"|gracias[,.]?\s*(?:adi[oó]s|adios|chao|hasta)",
    re.IGNORECASE,
)


def es_despedida(text: str, expected_slot: str | None, alerted: bool) -> bool:
    """True si el paciente está cerrando la llamada, no respondiendo un dato."""
    t = (text or "").strip()
    if not t:
        return False
    if _DESPEDIDA.search(t):
        return True
    n = _norm(t)
    if expected_slot is None and re.fullmatch(
            r"(?:no|no gracias|nada|eso|gracias|listo|ok|okay)\.?", n):
        return True
    if alerted and expected_slot is None and re.fullmatch(r"gracias\.?", n):
        return True
    return False


def parse_slot(slot: str | None, text: str) -> tuple[str, dict]:
    """Devuelve (estado, extracción).

    filled  — el dato pedido está en la frase; se puede responder sin LLM.
    unclear — audio basura; hay que pedir que repita.
    open    — conviene el pipeline completo (pregunta del paciente, rojo, etc.).
    """
    t = (text or "").strip()
    if not t:
        return "unclear", {}
    if quick_red_scan(t):
        return "open", {}
    # «¿Cómo bien?» no es una duda: Whisper marca «Como bien» como pregunta.
    # Solo el pipeline largo si el paciente formula una pregunta de verdad.
    if _PREGUNTA_REAL.search(t):
        return "open", {}
    if slot not in FAST_SLOTS:
        return "open", {}
    if looks_unclear(t) and slot in {"dolor", "fiebre"}:
        return "unclear", {}

    ext = fallback_extract(t)
    if slot == "dolor":
        n = dolor_de(t)
        if n is None:
            return "unclear" if looks_unclear(t) or len(t.split()) <= 2 else "open", ext
        ext["dolor_nrs"] = n
        if _SIN_FIEBRE.search(t):
            ext["_sin_fiebre"] = True
        return "filled", ext
    if slot == "fiebre":
        temp = temperatura_de(t)
        if temp is not None:
            ext["fiebre_c"] = temp
            return "filled", ext
        if _SIN_FIEBRE.search(t) or re.search(r"\bnada\b", _norm(t)):
            ext["_sin_fiebre"] = True
            return "filled", ext
        return "unclear", ext
    if slot == "herida":
        h = herida_de(t)
        if h:
            ext["herida"] = h
            return "filled", ext
        return "open", ext
    if slot == "movilidad":
        if looks_unclear(t):
            return "unclear", ext
        return "filled", ext
    if slot == "apetito_sueno":
        if looks_unclear(t):
            return "unclear", ext
        if re.search(r"no (?:tengo |tengo )?apetit|sin apetit|no com|nada", _norm(t)):
            ext["apetito"] = "nulo"
        return "filled", ext
    return "open", ext


def ack_for(slot: str, ext: dict) -> str:
    if slot == "dolor" and ext.get("dolor_nrs") is not None:
        n = ext["dolor_nrs"]
        return f"De acuerdo, el dolor está en {int(n) if float(n) == int(n) else n}."
    if slot == "fiebre":
        if ext.get("fiebre_c") is not None:
            t = ext["fiebre_c"]
            cifra = int(t) if float(t) == int(t) else t
            return f"De acuerdo, {cifra} grados."
        return "De acuerdo, sin fiebre."
    return "De acuerdo."

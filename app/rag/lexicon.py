"""Expansión bilingüe ES↔EN de términos médicos del dominio para la rama léxica (BM25).

Ataca el gap cross-lingual sin traducir el corpus: una consulta en español
recupera también documentos en inglés por coincidencia exacta de términos.
"""
from __future__ import annotations

import re
import unicodedata

# es -> [en...]; la expansión se aplica en ambos sentidos.
_LEX = {
    "apendicitis": ["appendicitis"],
    "apendicectomia": ["appendectomy", "appendicectomy"],
    "apendice": ["appendix"],
    "colecistectomia": ["cholecystectomy"],
    "colecistitis": ["cholecystitis"],
    "vesicula": ["gallbladder"],
    "biliar": ["biliary", "bile"],
    "colectomia": ["colectomy"],
    "colorrectal": ["colorectal"],
    "colon": ["colon", "bowel"],
    "mastectomia": ["mastectomy"],
    "mama": ["breast"],
    "seno": ["breast"],
    "cuello uterino": ["cervical", "cervix"],
    "reemplazo articular": ["joint replacement", "arthroplasty"],
    "cadera": ["hip"],
    "rodilla": ["knee"],
    "artroplastia": ["arthroplasty"],
    "herida": ["wound", "incision"],
    "incision": ["incision"],
    "infeccion": ["infection"],
    "fiebre": ["fever", "temperature"],
    "dolor": ["pain"],
    "sangrado": ["bleeding", "hemorrhage"],
    "hemorragia": ["hemorrhage", "bleeding"],
    "secrecion": ["discharge", "drainage", "exudate"],
    "pus": ["pus", "purulent"],
    "enrojecimiento": ["redness", "erythema"],
    "hinchazon": ["swelling", "edema"],
    "inflamacion": ["inflammation"],
    "nausea": ["nausea"],
    "nauseas": ["nausea", "vomiting"],
    "vomito": ["vomiting", "emesis"],
    "estrenimiento": ["constipation"],
    "diarrea": ["diarrhea"],
    "apetito": ["appetite"],
    "medicamento": ["medication", "drug"],
    "antibiotico": ["antibiotic"],
    "analgesico": ["analgesic", "painkiller"],
    "acetaminofen": ["acetaminophen", "paracetamol"],
    "ibuprofeno": ["ibuprofen"],
    "anticoagulante": ["anticoagulant", "blood thinner"],
    "trombosis": ["thrombosis", "dvt"],
    "embolia": ["embolism"],
    "dificultad para respirar": ["shortness of breath", "dyspnea"],
    "respirar": ["breathing", "breath"],
    "mareo": ["dizziness"],
    "desmayo": ["fainting", "syncope"],
    "sutura": ["suture", "stitches"],
    "puntos": ["stitches", "staples"],
    "vendaje": ["dressing", "bandage"],
    "curacion": ["wound care", "dressing change"],
    "banarse": ["shower", "bathing"],
    "ducha": ["shower"],
    "caminar": ["walking", "ambulation"],
    "movilidad": ["mobility", "range of motion"],
    "ejercicio": ["exercise", "physical therapy"],
    "fisioterapia": ["physical therapy", "physiotherapy"],
    "dieta": ["diet", "nutrition"],
    "alimentacion": ["diet", "feeding"],
    "cirugia": ["surgery", "operation"],
    "postoperatorio": ["postoperative", "post-op", "recovery"],
    "recuperacion": ["recovery"],
    "complicacion": ["complication"],
    "alarma": ["warning", "red flag", "alarm"],
    "urgencias": ["emergency", "emergency department"],
    "drenaje": ["drain", "drainage"],
    "seroma": ["seroma"],
    "hematoma": ["hematoma", "bruising"],
    "absceso": ["abscess"],
    "dehiscencia": ["dehiscence", "wound opening"],
    "ileo": ["ileus"],
    "fuga anastomotica": ["anastomotic leak"],
    "estoma": ["stoma", "ostomy"],
    "colostomia": ["colostomy"],
    "linfedema": ["lymphedema"],
    "protesis": ["prosthesis", "implant"],
    "luxacion": ["dislocation"],
    "temperatura": ["temperature", "fever"],
    "escalofrios": ["chills"],
    "orinar": ["urination", "urinary"],
    "estomago": ["stomach", "abdominal"],
    "abdomen": ["abdomen", "abdominal"],
    "barriga": ["abdomen", "belly"],
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# --- Reparación fonética de errores de ASR en respuestas cortas ---
# whisper con audio telefónico imperfecto produce "fierebres", "sin chasón"…
# Se repara por similitud contra el vocabulario del dominio, probando también
# la unión de palabras partidas ("sin chasón" → "hinchazón").
from difflib import SequenceMatcher

_ASR_TARGETS = [
    "fiebre", "hinchazón", "enrojecimiento", "secreción", "dolor", "herida",
    "mareo", "vómito", "náuseas", "sangrado", "calentura", "escalofríos",
    "pus", "apetito", "punzada", "supuración", "inflamación", "temperatura",
    "diarrea", "estreñimiento", "cansancio", "hormigueo", "ardor", "molestia",
    "pantorrilla", "drenaje", "sutura", "puntos", "vendaje", "medicamento",
    "acetaminofén", "ibuprofeno", "desmayo", "mocos", "tos", "hinchado",
    "enrojecida", "caliente", "morado", "amarillento", "cabeza", "espalda",
    "barriga", "estómago", "pierna", "brazo", "pecho", "cicatriz",
]
_ASR_NORM = {_norm(w): w for w in _ASR_TARGETS}
_CRITICAL_TARGETS = {"desmayo", "sangrado", "pus", "vomito"}


def _thresh(target_norm: str, base: float) -> float:
    return max(base, 0.86) if target_norm in _CRITICAL_TARGETS else base
# palabras comunes válidas que jamás deben "repararse" ni absorberse en uniones
_KEEP = set("""si no un una la el de mi me al ya muy asi mas nada todo bien mal
poco mucho mucha muchas muchos algo tengo tiene esta estoy fue por que como pero
con sin les los las doctor doctora senora senor gracias hola dias dia hace noche
anoche amarilla amarillo roja rojo rojiza verde blanca blanco duele siento sale
poquito normal ayer hoy grados casi cuando desde en y ni o u se lo te le ha he
va son esta estan para del""".split())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# Vocabulario español real (dataset del reto + corpus clínico es): la reparación
# SOLO puede tocar palabras fuera de este vocabulario. Sin esta compuerta, el
# 15% de frases válidas resultaban dañadas ("sanando"→"sangrado", "marcó"→"mareo").
from pathlib import Path as _Path

_VOCAB: set[str] = set()
try:
    _VOCAB = set((_Path(__file__).parent / "vocab_es.txt").read_text(
        encoding="utf-8").splitlines())
except OSError:
    pass


def _in_vocab(w: str) -> bool:
    if not _VOCAB:
        return False  # sin vocabulario: se confía en umbrales (modo degradado)
    if w in _VOCAB or w in _KEEP or w in _ASR_NORM:
        return True
    # tolerancia de plural: "punzadas" válida si "punzada" existe
    base = w.rstrip("s")
    if base in _VOCAB or base in _ASR_NORM:
        return True
    return w[:-2] in _VOCAB if w.endswith("es") else False


def _same_lemma(w: str, target: str) -> bool:
    """True si w es una flexión del target ('amarillenta' vs 'amarillento'):
    comparten raíz larga. Reparar ahí solo cambiaría género/número — cosmética
    que corrompe la transcripción sin aportar nada a la extracción."""
    i = 0
    while i < min(len(w), len(target)) and w[i] == target[i]:
        i += 1
    return i >= max(4, int(0.7 * len(target)))


# Números hablados: el paciente responde "seis" a "¿de 0 a 10?" y whisper
# devuelve "Sais", "¡Achoo!", "Ciete"… Con audio de medio segundo el modelo
# pequeño es inestable, así que se corrige fonéticamente contra los 11 números.
_NUMEROS = {
    "cero": "cero", "uno": "uno", "una": "uno", "dos": "dos", "tres": "tres",
    "cuatro": "cuatro", "cinco": "cinco", "seis": "seis", "siete": "siete",
    "ocho": "ocho", "nueve": "nueve", "diez": "diez",
}
# variantes fonéticas observadas en transcripción real
_NUM_VARIANTES = {
    "sais": "seis", "seys": "seis", "sei": "seis", "says": "seis", "sais": "seis",
    "achoo": "ocho", "acho": "ocho", "osho": "ocho", "ochio": "ocho", "hocho": "ocho",
    "ciete": "siete", "siet": "siete", "syete": "siete", "shiete": "siete",
    "nuebe": "nueve", "nuve": "nueve", "nweve": "nueve",
    "quatro": "cuatro", "cuatr": "cuatro", "kuatro": "cuatro",
    "sinco": "cinco", "zinco": "cinco", "cincoo": "cinco",
    "diescy": "diez", "dies": "diez", "diaz": "diez",
    "choco": "ocho", "chocho": "ocho", "nocho": "ocho", "ochop": "ocho",
    "noche": "ocho", "bocho": "ocho", "otro": "ocho",
    "baby": "nueve", "nube": "nueve", "llueve": "nueve", "nuece": "nueve",
    "sais": "seis", "vez": "diez", "die": "diez", "cinc": "cinco",
    "trez": "tres", "tress": "tres", "dose": "doce",
}


def interpretar_numero_hablado(text: str) -> str | None:
    """Si la respuesta es una o dos palabras y suena a un número del 0 al 10,
    devuelve ese número en letras. Devuelve None si no aplica."""
    limpio = re.sub(r"[^\wáéíóúñü ]", " ", text.lower()).strip()
    palabras = [_norm(p) for p in limpio.split() if p]
    if not palabras or len(palabras) > 2:
        return None
    for p in palabras:
        if re.fullmatch(r"\d{1,2}", p) and 0 <= int(p) <= 10:
            return p                       # whisper ya devolvió el dígito
        if p in _NUMEROS:
            return _NUMEROS[p]
        if p in _NUM_VARIANTES:
            return _NUM_VARIANTES[p]
    # Similitud fonética SOLO con alta confianza: devolver un número equivocado
    # («cero» oído como «seis») es peor que no devolver ninguno, porque el
    # agente puede simplemente volver a preguntar.
    for p in palabras:
        if len(p) < 3:
            continue
        mejor = max(_NUMEROS, key=lambda n: _ratio(p, n))
        if _ratio(p, mejor) >= 0.8:
            return _NUMEROS[mejor]
    return None


_CONFIRMACION_PREGUNTA = re.compile(
    r"qued[oó] claro|me confirma|entendi[oó]|comprendi[oó]|de acuerdo",
    re.IGNORECASE)
_CONFIRMACION_SI = re.compile(
    r"\b(?:s[ií]|claro|entend[ií]|entendido|comprend[ií]|de acuerdo|correcto)\b"
    r"|(?:y\s+)?se?\s+qued[ea]\s+(?:a\s+un\s+lado|claro)",
    re.IGNORECASE)
_CONFIRMACION_NO = re.compile(
    r"\bno\b.{0,30}\b(?:qued[oó]\s+claro|entend[ií]|comprend[ií]|de acuerdo)\b"
    r"|\b(?:repita|rep[ií]tame|otra vez|no entend[ií])\b",
    re.IGNORECASE)


def interpretar_confirmacion(text: str, pregunta_agente: str | None) -> str | None:
    """Normaliza respuestas sí/no SOLO cuando Clara acaba de pedir confirmación.

    Incluye variantes observadas del ASR («y se quede a un lado» por
    «sí, quedó claro»). Fuera de ese contexto no modifica el habla.
    """
    if not pregunta_agente or not _CONFIRMACION_PREGUNTA.search(pregunta_agente):
        return None
    limpio = re.sub(r"\s+", " ", text.strip())
    if not limpio or len(limpio.split()) > 10:
        return None
    if _CONFIRMACION_NO.search(limpio):
        return "No, no me quedó claro."
    if _CONFIRMACION_SI.search(limpio):
        return "Sí, quedó claro."
    return None


def repair_asr(text: str) -> tuple[str, list[str]]:
    """Repara palabras irreconocibles acercándolas al léxico clínico.
    Devuelve (texto_reparado, lista_de_reparaciones)."""
    tokens = text.split()
    out: list[str] = []
    repairs: list[str] = []
    i = 0
    while i < len(tokens):
        raw = tokens[i]
        w = _norm(re.sub(r"[^\wáéíóúñü]", "", raw.lower()))
        # 1) unión con la palabra siguiente ("sin"+"chason" → "hinchazon").
        #    Solo si la siguiente NO es ya una palabra válida (nunca absorber
        #    "sin fiebre" → "fiebre": eliminaría una negación) y el propio
        #    token actual tampoco es un término del dominio.
        if i + 1 < len(tokens):
            nxt = _norm(re.sub(r"[^\wáéíóúñü]", "", tokens[i + 1].lower()))
            joined = w + nxt
            # si la primera palabra es válida y la segunda se puede reparar
            # sola, NO unir (conserva artículos: "la erida" → "la herida")
            nxt_single_ok = (len(nxt) >= 5 and not _in_vocab(nxt)
                             and max((_ratio(nxt, t) for t in _ASR_NORM), default=0) >= 0.78)
            if (len(joined) >= 5 and w not in _ASR_NORM
                    and len(nxt) >= 3 and not _in_vocab(nxt)
                    and not (w in _KEEP and nxt_single_ok)):
                best = max(_ASR_NORM, key=lambda t: _ratio(joined, t), default=None)
                if best and _ratio(joined, best) >= _thresh(best, 0.75):
                    fixed = _ASR_NORM[best]
                    trailing = re.sub(r"[\wáéíóúñü]", "", tokens[i + 1])
                    out.append(fixed + trailing)
                    repairs.append(f"{raw} {tokens[i+1]}→{fixed}")
                    i += 2
                    continue
        # 2) palabra suelta ("fierebres" → "fiebre") — solo si NO es española válida
        if len(w) >= 5 and not _in_vocab(w):
            best = max(_ASR_NORM, key=lambda t: _ratio(w, t), default=None)
            if best and _ratio(w, best) >= _thresh(best, 0.74) and not _same_lemma(w, best):
                fixed = _ASR_NORM[best]
                trailing = re.sub(r"[\wáéíóúñü]", "", raw)
                out.append(fixed + trailing)
                repairs.append(f"{raw}→{fixed}")
                i += 1
                continue
        out.append(raw)
        i += 1
    return " ".join(out), repairs


def expand_query(query: str) -> str:
    """Añade equivalentes EN (y ES) de los términos detectados a la consulta."""
    qn = _norm(query)
    extra: list[str] = []
    for es, ens in _LEX.items():
        if es in qn:
            extra += [e for e in ens if _norm(e) not in qn]
        else:
            for e in ens:
                if re.search(rf"\b{re.escape(_norm(e))}\b", qn):
                    extra.append(es)
                    break
    if extra:
        return query + " " + " ".join(dict.fromkeys(extra))
    return query

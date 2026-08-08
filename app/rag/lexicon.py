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
# palabras comunes válidas que jamás deben "repararse" ni absorberse en uniones
_KEEP = set("""si no un una la el de mi me al ya muy asi mas nada todo bien mal
poco mucho mucha muchas muchos algo tengo tiene esta estoy fue por que como pero
con sin les los las doctor doctora senora senor gracias hola dias dia hace noche
anoche amarilla amarillo roja rojo rojiza verde blanca blanco duele siento sale
poquito normal ayer hoy grados casi cuando desde en y ni o u se lo te le ha he
va son esta estan para del""".split())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


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
            nxt_single_ok = (len(nxt) >= 5 and nxt not in _KEEP and nxt not in _ASR_NORM
                             and max((_ratio(nxt, t) for t in _ASR_NORM), default=0) >= 0.78)
            if (len(joined) >= 5 and w not in _ASR_NORM
                    and nxt not in _ASR_NORM and nxt not in _KEEP and nxt
                    and not (w in _KEEP and nxt_single_ok)):
                best = max(_ASR_NORM, key=lambda t: _ratio(joined, t), default=None)
                if best and _ratio(joined, best) >= 0.75:
                    fixed = _ASR_NORM[best]
                    trailing = re.sub(r"[\wáéíóúñü]", "", tokens[i + 1])
                    out.append(fixed + trailing)
                    repairs.append(f"{raw} {tokens[i+1]}→{fixed}")
                    i += 2
                    continue
        # 2) palabra suelta ("fierebres" → "fiebre")
        if len(w) >= 5 and w not in _KEEP and w not in _ASR_NORM:
            best = max(_ASR_NORM, key=lambda t: _ratio(w, t), default=None)
            if best and _ratio(w, best) >= 0.78:
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

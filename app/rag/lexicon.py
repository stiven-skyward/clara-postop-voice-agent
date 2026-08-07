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

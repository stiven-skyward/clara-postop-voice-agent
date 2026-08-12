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
    # formas pronominales colombianas de dehiscencia (antes solo casaban los
    # literales impersonales y el rojo dependía por completo del LLM)
    "se me abrio", "abrio la herida", "se abrio la incision", "reventaron los puntos",
    "se me reventaron", "soltaron los puntos", "se abrieron los puntos",
    "se me solto la herida", "se me salieron los puntos",
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
    "disnea": r"respir|ahog|aire|asfixi|alient|fatig|resuell|agitad|jadea|"
              r"cansancio al camin|me canso|sofoc",
    "vomito_persistente": r"vomit|devolv|trasboc|arroj|guacar|basque",
    "fiebre_subjetiva": r"fiebre|calentur|temperatura|destempl|febril|caliente|acalorad",
}

# Anclaje para campos enum: el valor solo se acepta si el texto contiene un
# lexema del síntoma DEGRADADO (evita que "dormí regular pero bien" → alterado)
_GROUNDING_ENUM = {
    "apetito": r"apetito|hambre|desgano|ganas|no he comido|como poc|comer poc|provoca|casi no com|no com[eo] (?:casi )?nada",
    "sueno": r"no duermo|duermo poc|duermo mal|despiert|desvel|insomn|no (?:he )?podido dormir|casi no duermo|trasnoch|dormido mal|no logro dormir|pego el ojo|dando vueltas|no puedo dormir|no poder dormir|no me deja dormir|duermo casi nada",
}

# Marcadores de minimización ASERTIVA ("uno aguanta", "nada grave") — el
# minimizador afirma que todo está bien; el ansioso PREGUNTA si es normal.
# Distinguirlos evita que el detector de incongruencia escale a los ansiosos.
_MINIMIZATION = re.compile(
    r"uno aguanta|aguanta uno|yo aguanto|no se preocupe|nada grave|no es nada"
    r"|ya se (?:me )?pasara|no es para tanto|no quiero molestar"
    r"|creo que es normal|capaz es normal|es normal (?:de la|del|no mas|con)"
    r"|uno ya sabe|no me preocupo|no estoy tan mal|tan mal no")


def sanitize_extraction(ext: dict, user_text: str) -> dict:
    tn = _norm(user_text)
    out = dict(ext)
    grounded = fallback_extract(user_text)
    for field_name, pattern in _GROUNDING.items():
        if out.get(field_name) is True and not re.search(pattern, tn):
            out.pop(field_name)
    for field_name, pattern in _GROUNDING_ENUM.items():
        if out.get(field_name) and not re.search(pattern, tn):
            out.pop(field_name)
    # anclaje de los campos NUMÉRICOS: el modelo pequeño llega a inventar
    # "dolor_nrs: 10" en textos que no hablan de dolor (se observó con
    # inyecciones de prompt). Exigimos palabra del dominio + un número.
    tnum = _palabras_a_numero(tn)
    # Asimetría clínica en el dolor: una puntuación BAJA inferida por el modelo
    # sin que el paciente diera un número es exactamente lo que produce el
    # minimizador ("un poquito molesto, uno aguanta" → el LLM emitía 2 y
    # enmascaraba un dolor real de 9/10). Solo se acepta si hay respaldo
    # numérico explícito; una puntuación alta sí se acepta (va del lado seguro).
    dolor = out.get("dolor_nrs")
    if dolor is not None:
        sin_dominio = not re.search(r"dolor|duele|molest|adolori|escala|punzad|arde", tn)
        dolor_grounded = grounded.get("dolor_nrs")
        respaldo = dolor_grounded is not None
        if sin_dominio or (float(dolor) < 5 and not respaldo):
            out.pop("dolor_nrs")
        elif respaldo:
            # El número literal manda sobre una cifra distinta inventada por
            # el modelo pequeño (p. ej. «siete» no puede convertirse en nueve).
            out["dolor_nrs"] = dolor_grounded
    if out.get("fiebre_c") is not None:
        if not (re.search(r"fiebre|temperatur|calentur|grados|termometr|febril", tn)
                and re.search(r"\d", tnum)):
            out.pop("fiebre_c")
        elif grounded.get("fiebre_c") is not None:
            # Ancla la temperatura a la cifra realmente pronunciada.
            out["fiebre_c"] = grounded["fiebre_c"]
    # el paciente habla del dolor pero no lo cuantifica → señal blanda de evasión
    if out.get("dolor_nrs") is None and re.search(r"dolor|duele|molest|adolori", tn):
        out["dolor_mencionado_sin_numero"] = True
    if _MINIMIZATION.search(tn):
        out["minimizacion"] = True
    # anclaje inverso: secreción purulenta descrita con eufemismos ("liquidito
    # amarillito") que el LLM a veces clasifica benigna — el regex la fuerza,
    # respetando negaciones ("nada de pus")
    for m in re.finditer(
            r"(?:liquid\w*|secre\w*|supura\w*|sale|solt\w*|salien\w*)[^.]{0,25}(?:amarill\w*|verd\w*)"
            r"|pus|huele (?:feo|mal|maluco)|mal olor|olor feo", tn):
        if re.search(_NEGATION, tn[max(0, m.start() - 30):m.start()]):
            continue          # esta ocurrencia está negada; seguir buscando
        if out.get("herida") not in ("abierta",):
            out["herida"] = "secrecion_purulenta"
        break
    return out


# --- Extractor determinista de respaldo (NO depende del LLM) --------------
# Red de seguridad: si llama-server está caído o devuelve basura, los síntomas
# numéricos de alarma (fiebre y dolor) se siguen detectando. Sin esto, una
# llamada con el LLM caído clasificaba "39 de fiebre y dolor 9" como VERDE.
_PALABRA_NUM = {
    "cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "treinta": 30, "cuarenta": 40,
}


def _palabras_a_numero(texto: str) -> str:
    """'treinta y nueve' → '39'; 'nueve' → '9' (para que los regex numéricos
    funcionen aunque el ASR escriba los números en letras)."""
    t = texto
    for dec in ("cuarenta", "treinta", "veinte"):
        for uni, v in _PALABRA_NUM.items():
            if v < 10 and v > 0:
                t = re.sub(rf"\b{dec}\s+y\s+{uni}\b",
                           str(_PALABRA_NUM[dec] + v), t)
        t = re.sub(rf"\b{dec}\b", str(_PALABRA_NUM[dec]), t)
    for pal, v in _PALABRA_NUM.items():
        t = re.sub(rf"\b{pal}\b", str(v), t)
    t = re.sub(r"\b(\d+)\s+(?:punto|coma|con)\s+(\d+)\b", r"\1.\2", t)
    return t


# unidades temporales/posológicas: un número seguido de estas NO es un síntoma
_NO_ES_MEDIDA = r"(?:hora|dia|día|minuto|semana|mes|año|ano|punto de sutura|puntos|" \
                r"pastilla|tableta|capsula|cápsula|gota|veces|vez|cucharada|mg|ml|" \
                r"miligramo|centimetro|centímetro|cm|noche|mañana|tarde)"


# Síntomas atribuidos a OTRA persona: en las llamadas interviene un familiar
# (el dataset del reto inserta turnos de terceros). "Mi hijo tiene 39 de fiebre"
# no debe escalar la llamada del paciente.
_TERCERO = re.compile(
    r"\b(?:mi|el|la|su)\s+(?:hij[oa]|mam[aá]|pap[aá]|espos[oa]|marido|mujer|"
    r"herman[oa]|niet[oa]|abuel[oa]|vecin[oa]|amig[oa]|suegr[oa]|yern[oa]|"
    r"nuer[a]|primo|prima|ti[oa]|sobrin[oa])\b")


# marcas de que el síntoma SÍ es del paciente aunque lo narre un familiar
# ("mi hija dice que TENGO 39 de fiebre")
_PRIMERA_PERSONA = re.compile(
    r"\b(?:tengo|siento|estoy|ando|amaneci|amanecí|me\s+\w+|mi\s+(?:herida|"
    r"cirugia|cirugía|operacion|operación|dolor|puntos|pierna|barriga)|"
    r"mio|mía|mia|yo|conmigo)\b")


def _sin_terceros(tn: str) -> str:
    """Elimina las oraciones cuyo sujeto es un tercero, PERO conserva las que
    llevan marca de primera persona: un cuidador narrando los síntomas del
    paciente es lo habitual en postoperatorio."""
    frases = re.split(r"(?<=[.!?,;])\s+|\s+(?:pero|aunque|y)\s+", tn)
    quedan = [f for f in frases
              if not _TERCERO.search(f) or _PRIMERA_PERSONA.search(f)]
    return " ".join(quedan) if quedan else ""


def fallback_extract(text: str) -> dict:
    """Extracción por reglas de los síntomas de mayor señal. Se aplica SIEMPRE
    como red de seguridad y se fusiona con la del LLM tomando lo más grave."""
    tn = _sin_terceros(_palabras_a_numero(_norm(text)))
    out: dict = {}

    # Fiebre: número plausible (35-43) que sea REALMENTE una temperatura.
    # Sin la exclusión temporal, "fiebre desde hace treinta y seis horas"
    # producía fiebre_c=36 y encima enmascaraba la fiebre subjetiva → verde.
    for m in re.finditer(r"\b(3[5-9]|4[0-3])(?:[.,](\d))?\b", tn):
        val = float(f"{m.group(1)}.{m.group(2) or 0}")
        despues = tn[m.end():m.end() + 22]
        if re.match(rf"\s*{_NO_ES_MEDIDA}", despues):
            continue                      # "36 horas", "38 días"
        antes = tn[max(0, m.start() - 28):m.start()]
        if re.search(r"fiebre|temperatura|calentur|termometr|febril|marc[oó]|tengo|"
                     r"estaba|esta en|subi[oó]|paso de|más de|mas de", antes) or \
                re.search(r"^\s*(?:grados|°|de fiebre|de temperatura)", despues):
            out["fiebre_c"] = max(out.get("fiebre_c", 0), val)
    hay_palabra_fiebre = re.search(r"fiebre|calentur|destempl|febril|acalorad", tn)
    negada = re.search(
        r"(?:sin|no|nada de|ni)\s+(?:\w+\s+){0,2}(?:fiebre|calentur|temperatura)", tn)
    if hay_palabra_fiebre and not negada and "fiebre_c" not in out:
        out["fiebre_subjetiva"] = True

    # Dolor 0-10: exige patrón de escala explícito, no mera cercanía. Antes,
    # "dos pastillas cada ocho horas para el dolor" daba dolor_nrs=8 → rojo.
    for pat in (r"(?:dolor|duele|molestia|adolorid\w*)\D{0,18}?\b(10|[0-9])\b",
                r"\b(10|[0-9])\s*(?:de|sobre|/)\s*(?:10|diez)",
                r"\b(10|[0-9])\b\D{0,12}?(?:de dolor|en la escala)"):
        for m in re.finditer(pat, tn):
            val = int(m.group(1))
            if re.match(rf"\s*{_NO_ES_MEDIDA}", tn[m.end(1):m.end(1) + 22]):
                continue
            out["dolor_nrs"] = max(out.get("dolor_nrs", 0), val)

    if re.search(r"empeor|aument|va subiendo|mas fuerte|cada vez peor", tn):
        out["dolor_empeora"] = True
    if re.search(r"no puedo respirar|me falta el (?:aire|aliento)|me ahogo|"
                 r"dificultad para respirar|me sofoco|me fatigo|sin resuello|"
                 r"me agito|me canso (?:mucho|al camin)", tn):
        out["disnea"] = True
    if re.search(r"\bsangr\w*|\bsangre\b|\bhemorragi\w*", tn) and not re.search(
            r"(?:sin|no|nada de|ni)\s+(?:\w+\s+){0,2}(?:sangr|sangre|hemorragi)", tn):
        out["sangrado"] = True
    return out


def merge_worst(base: dict, extra: dict) -> dict:
    """Funde dos extracciones quedándose con el valor MÁS GRAVE de cada campo."""
    out = dict(base)
    for k, v in extra.items():
        if v is None:
            continue
        cur = out.get(k)
        if cur is None:
            out[k] = v
        elif k in ("fiebre_c", "dolor_nrs"):
            out[k] = max(float(cur), float(v))
        elif isinstance(v, bool):
            out[k] = bool(cur) or v
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
    minimizacion: int = 0   # nº de turnos con minimización asertiva
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
            if v is None:
                continue
            cur = getattr(self, k)
            # MONOTONÍA CLÍNICA: dentro de una llamada el cuadro solo se agrava.
            # Sin esto, un "no, la herida ya está normal" del turno 4 borraba la
            # secreción purulenta del turno 2 y el caso dejaba de escalar.
            if cur is not None:
                if k == "herida":
                    orden = ["normal", "enrojecida_leve", "enrojecida", "hinchada",
                             "secrecion_clara", "secrecion_purulenta", "abierta"]
                    if v in orden and cur in orden and orden.index(v) < orden.index(cur):
                        continue
                elif k == "apetito":
                    if cur == "nulo" and v == "reducido":
                        continue
                elif k in ("dolor_nrs", "fiebre_c"):
                    if float(v) < float(cur):
                        continue
                elif isinstance(cur, bool) and cur and not v:
                    continue          # una bandera de alarma no se apaga sola
            setattr(self, k, v)
        if ext.get("minimizacion"):
            self.minimizacion += 1
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
            "minimizacion": self.minimizacion,
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
    elif s.herida == "hinchada":
        razones_amarillo.append("Hinchazón alrededor de la herida")
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
    if fiebre_alguna and s.herida in ("enrojecida", "hinchada", "secrecion_clara"):
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
    if s.herida in ("enrojecida_leve", "enrojecida", "hinchada", "secrecion_clara"):
        blandas.append("herida alterada (aunque la describe leve)")
    # el termómetro manda: sensación febril solo cuenta si NO hay medición normal
    if (s.fiebre_subjetiva and s.fiebre_c is None) or \
            (s.fiebre_c is not None and 37.8 <= s.fiebre_c < 38.0):
        blandas.append("sensación febril o febrícula")
    if s.apetito in ("reducido", "nulo"):
        blandas.append("apetito reducido")
    if s.sueno == "alterado":
        blandas.append("sueño alterado")
    if s.dolor_mencionado_sin_numero and s.dolor_nrs is None:
        # Siempre se INDAGA (pedir el número de 0 a 10), pero solo cuenta como
        # señal blanda si además minimiza activamente: un paciente que sencilla-
        # mente no dio un número no es sospechoso; uno que evade el número
        # mientras repite "uno aguanta, no es nada" sí lo es. Medido contra el
        # dataset: sin esta condición, 7 de 20 casos verdes escalaban a rojo.
        faltantes.append("intensidad del dolor de 0 a 10")
        if s.minimizacion >= 1:
            blandas.append("dolor evadido sin cuantificar (paciente que minimiza)")

    # Calibración final (barrido contra el dataset: 0 falsos negativos,
    # recall rojo 12/12): ≥3 dominios blandos degradados a la vez → rojo,
    # ≥2 → amarillo. Sesgo deliberado a la seguridad: el sobre-escalamiento
    # es el costo aceptado de no subestimar nunca a un minimizador.
    if len(blandas) >= 3:
        sufijo = (" en paciente que minimiza activamente ('uno aguanta')"
                  if s.minimizacion >= 1 else "")
        razones_rojo.append(
            "Múltiples dominios afectados a la vez (" + "; ".join(blandas) +
            ")" + sufijo + ": cuadro incongruente con evolución normal")
    elif len(blandas) >= 2:
        razones_amarillo.append(
            "Varias molestias simultáneas (" + "; ".join(blandas) + ")")

    if razones_rojo:
        return TriageResult("rojo", razones_rojo, [])
    if razones_amarillo:
        return TriageResult("amarillo", razones_amarillo, faltantes)
    return TriageResult("verde", ["Sin signos de alarma en lo reportado"], faltantes)


_QUICK_RED = _RED_TEXT + [
    "huele feo", "huele mal", "mal olor", "liquido amarillo", "liquidito amarillo",
    "liquido verde", "amarillit", "amarillent", "verdos", "no aguanto el dolor",
    "dolor insoportable",
]


_NEGATION = r"(?:nada de|sin|no hay|no tengo|no me sale|ni|no veo|tampoco)\s+(?:\w+\s+){0,3}$"


def quick_red_scan(text: str) -> str | None:
    """Barrido léxico instantáneo (sin LLM) de señales rojas en el turno crudo.
    Permite responder el escalamiento en <1 s; la extracción formal corre después
    para el registro. Ignora menciones negadas ("nada de pus"). Devuelve la
    señal detectada o None."""
    tn = _sin_terceros(_norm(text))   # "mi hijo bota pus" no escala esta llamada
    for kw in _QUICK_RED:
        # TODAS las ocurrencias: "no me sale pus… bueno, ahorita sí me salió
        # pus amarillo" se perdía por mirar solo la primera (negada)
        for m in re.finditer(re.escape(kw), tn):
            if not re.search(_NEGATION, tn[max(0, m.start() - 30):m.start()]):
                return kw
    return None


def combine(previous: str, new: str) -> str:
    """El nivel de la llamada solo puede subir (asimetría clínica)."""
    return LEVELS[max(LEVELS.index(previous), LEVELS.index(new))]

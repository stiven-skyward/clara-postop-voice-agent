"""Orquestador de la llamada: máquina de estados conversacional.

Por cada turno del paciente:
  1. Extracción estructurada de síntomas y dudas (LLM, JSON forzado).
  2. Triaje determinista (reglas) — el nivel solo sube.
  3. Si hay duda del paciente → RAG (híbrido) → respuesta fundamentada con citas.
  4. Si triaje rojo → protocolo de escalamiento (alerta persistente + aviso al paciente).
  5. Avance del guion clínico (dolor→fiebre→herida→movilidad→apetito/sueño) con
     indagación ante ambigüedad (faltantes del motor de triaje).
Al cerrar: resumen estructurado persistente de la llamada.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from app import config, llm, metrics
from app.agent import prompts
from app.agent.triage import (SymptomState, TriageResult, combine, evaluate,
                              fallback_extract, merge_worst, quick_red_scan,
                              sanitize_extraction)
from app.rag.search import Source, get_searcher


@dataclass
class Patient:
    nombre: str = "Paciente"
    edad: int = 50
    procedimiento: str = "cirugía"
    dia_postop: int = 3
    escenario: str | None = None
    paciente_id: str = ""


@dataclass
class CallState:
    patient: Patient
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    history: list[dict] = field(default_factory=list)     # mensajes chat completos
    symptoms: SymptomState = field(default_factory=SymptomState)
    nivel: str = "verde"
    razones: list[str] = field(default_factory=list)
    checklist_done: set = field(default_factory=set)
    turno: int = 0
    sources_used: list[dict] = field(default_factory=list)
    alerted: bool = False
    closed: bool = False
    rag_queries: int = 0
    texto_final: str | None = None  # transcripción corregida del último turno
    transcript: list = field(default_factory=list)  # íntegro, sin poda (resumen)
    checklist_pendiente: str | None = None   # dominio entregado, aún sin confirmar


def _system_prompt(p: Patient) -> str:
    return prompts.SYSTEM_CONVERSACION.format(
        nombre=p.nombre, edad=p.edad,
        procedimiento=p.procedimiento, dia_postop=p.dia_postop,
    )


def greeting(state: CallState) -> str:
    p = state.patient
    # "Le llamo": trato neutro válido para cualquier género del paciente
    text = (
        f"Hola {p.nombre.split()[0]}, le habla Clara, del programa de seguimiento "
        f"de su clínica. Le llamo para saber cómo sigue después de su "
        f"{p.procedimiento.lower()}, que fue hace {p.dia_postop} "
        f"día{'s' if p.dia_postop != 1 else ''}. "
        f"Para empezar, ¿cómo ha estado el dolor? De 0 a 10, ¿en cuánto lo siente?"
    )
    state.history.append({"role": "system", "content": _system_prompt(p)})
    state.history.append({"role": "assistant", "content": text})
    state.transcript.append(f"AGENTE: {text}")
    state.checklist_done.add("dolor")
    return text


def _next_checklist_hint(state: CallState) -> str | None:
    # lo que el paciente ya contó espontáneamente no se vuelve a preguntar
    s = state.symptoms
    if s.dolor_nrs is not None:
        state.checklist_done.add("dolor")
    if s.fiebre_c is not None or s.fiebre_subjetiva is not None:
        state.checklist_done.add("fiebre")
    if s.herida is not None:
        state.checklist_done.add("herida")
    if s.apetito is not None or s.sueno is not None:
        state.checklist_done.add("apetito_sueno")
    for key, desc in prompts.GUION_CHECKLIST:
        if key not in state.checklist_done:
            # se marca de forma PROVISIONAL: si el turno se interrumpe antes de
            # que Clara llegue a preguntarlo, se revierte (ver process_turn)
            state.checklist_done.add(key)
            state.checklist_pendiente = key
            return desc
    return None


def _format_sources(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        body = s.ventana(1200)  # ventana centrada en lo que casó; prefill acotado
        blocks.append(f"[{s.n}] «{s.titulo}» — sección: {s.seccion} "
                      f"(págs. {s.pagina_ini}-{s.pagina_fin})\n{body}")
    return "\n\n".join(blocks)


# La primera oración es estática a propósito: está pre-sintetizada en caché de
# TTS, así el escalamiento rojo empieza a sonar de inmediato.
_ESCALATION_MSG = (
    "Por lo que me cuenta, esto lo debe revisar el equipo médico hoy mismo. "
    "{nombre}, voy a generar una alerta ahora para que le devuelvan la llamada lo antes posible. "
    "Si presenta más síntomas o se siente peor, no espere: acuda a urgencias de una vez. "
    "¿Me confirma que quedó claro?"
)


def _persistir_alerta(state: CallState, disparador: str) -> None:
    p = state.patient
    metrics.log_alert(state.call_id, {
        "paciente": p.nombre, "paciente_id": p.paciente_id,
        "procedimiento": p.procedimiento, "dia_postop": p.dia_postop,
        "nivel": "rojo", "razones": state.razones,
        "sintomas": state.symptoms.as_dict(),
        "transcripcion_disparadora": disparador,
    })


_CLINICAL_HINTS = re.compile(
    r"dolor|duele|fiebre|calentura|temperatura|herida|sangr|mareo|vomit|nause|"
    r"respir|hinch|roj|secre|liquid|pus|comer|apetito|dorm|sueño|camin|mover|"
    r"pierna|brazo|pecho|cabeza|barriga|estomago|estómago|punto|grado|\?|\d",
    re.IGNORECASE)


def _worth_extracting(text: str) -> bool:
    return len(text) > 80 or bool(_CLINICAL_HINTS.search(text))


def _extraction_messages(state: CallState, user_text: str) -> list[dict]:
    """La última pregunta del agente da el contexto para interpretar
    respuestas cortas o mal transcritas ("sin chasón" tras preguntar por
    hinchazón = "sí, hinchazón")."""
    last_q = ""
    for m in reversed(state.history):
        if m["role"] == "assistant":
            last_q = m["content"][:220]
            break
    content = (f"El AGENTE preguntó: «{last_q}»\n"
               f"El PACIENTE respondió (transcripción telefónica): «{user_text}»")
    return [{"role": "system", "content": prompts.SYSTEM_EXTRACCION},
            {"role": "user", "content": content}]


def process_turn(state: CallState, user_text: str,
                 tm: metrics.TurnMetrics) -> Iterator[str]:
    """Procesa un turno del paciente. Genera el texto de respuesta en streaming."""
    state.turno += 1
    stats = llm.LLMStats()
    p = state.patient

    # 0) Atajo de seguridad: señal roja léxica → escalar YA (<1 s), sin esperar
    #    al LLM. La extracción corre después solo para completar el registro.
    state.texto_final = None
    state.checklist_pendiente = None
    state.transcript.append(f"PACIENTE: {user_text}")
    señal = quick_red_scan(user_text)
    if señal and not state.alerted:
        state.nivel = "rojo"
        state.alerted = True
        state.symptoms.merge(fallback_extract(user_text))
        state.symptoms.otros.append(user_text[:160])
        tri = evaluate(state.symptoms, p.procedimiento, p.dia_postop)
        state.razones = list(tri.razones)
        marca = f"Señal de alarma textual: «{señal}»"
        if marca not in state.razones:
            state.razones.append(marca)
        # La alerta se PERSISTE ANTES de hablar: si el paciente interrumpe o se
        # cae la conexión durante el yield, el generador se abandona y la alerta
        # se habría perdido con state.alerted ya en True (nunca se reintentaría).
        _persistir_alerta(state, user_text)
        text = _ESCALATION_MSG.format(nombre=p.nombre.split()[0])
        tm.set(triaje="rojo", escalado=True, atajo_lexico=señal)
        state.history.append({"role": "user", "content": user_text})
        state.history.append({"role": "assistant", "content": text})
        state.transcript.append(f"AGENTE: {text}")
        yield text
        try:   # la extracción fina solo enriquece el registro ya guardado
            ext = llm.structured(_extraction_messages(state, user_text),
                                 prompts.SCHEMA_EXTRACCION, stats=stats, max_tokens=300)
            ground = user_text + " " + (ext.get("texto_corregido") or "")
            state.symptoms.merge(merge_worst(sanitize_extraction(ext, ground),
                                             fallback_extract(ground)))
        except Exception:
            pass
        tm.set(tokens_in=stats.prompt_tokens, tokens_out=stats.completion_tokens,
               llm_calls=stats.calls, rag_queries=0)
        return

    # 1) Extracción estructurada (solo si el turno puede contener información
    #    clínica o una pregunta; un "sí señora" corto no paga LLM)
    ext = {}
    state.texto_final = None
    if _worth_extracting(user_text):
        try:
            ext = llm.structured(_extraction_messages(state, user_text),
                                 prompts.SCHEMA_EXTRACCION, stats=stats, max_tokens=300)
        except Exception:
            # conservar el habla cruda: las reglas de texto libre (TVP,
            # red flags) leen state.symptoms.otros
            ext = {"otros": [user_text[:200]], "pregunta": None}
    corregido = (ext.get("texto_corregido") or "").strip()
    if corregido and corregido.lower() != user_text.strip().lower():
        state.texto_final = corregido
    # el anclaje valida contra el texto original MÁS el corregido: una señal
    # legítima recuperada por la corrección no debe descartarse
    ext = sanitize_extraction(ext, user_text + " " + corregido)
    # red de seguridad determinista: si el LLM falló, omitió o subestimó un
    # síntoma numérico de alarma, las reglas lo recuperan (falso negativo = falla
    # catastrófica). Se funde quedándose siempre con lo más grave.
    ext = merge_worst(ext, fallback_extract(user_text + " " + corregido))
    state.symptoms.merge(ext)
    user_text_final = corregido or user_text

    # 2) Triaje determinista (solo sube)
    tri: TriageResult = evaluate(state.symptoms, p.procedimiento, p.dia_postop)
    prev = state.nivel
    state.nivel = combine(state.nivel, tri.nivel)
    nuevas = [r for r in tri.razones if r not in state.razones]
    if state.nivel != prev or not state.razones:
        state.razones = list(tri.razones)
    elif nuevas and state.nivel == "rojo":
        # un SEGUNDO motivo rojo en la misma llamada debe llegar a quien
        # devuelve la llamada; antes solo quedaba registrado el primero
        state.razones.extend(nuevas)
        _persistir_alerta(state, user_text)
    tm.set(triaje=state.nivel, extraccion=ext)

    # 3) Escalamiento inmediato si pasó a rojo
    if state.nivel == "rojo" and not state.alerted:
        state.alerted = True
        metrics.log_alert(state.call_id, {
            "paciente": p.nombre, "paciente_id": p.paciente_id,
            "procedimiento": p.procedimiento, "dia_postop": p.dia_postop,
            "nivel": "rojo", "razones": tri.razones,
            "sintomas": state.symptoms.as_dict(),
            "transcripcion_disparadora": user_text,
        })
        text = _ESCALATION_MSG.format(nombre=p.nombre.split()[0])
        state.history.append({"role": "user", "content": user_text})
        state.history.append({"role": "assistant", "content": text})
        tm.set(tokens_in=stats.prompt_tokens, tokens_out=stats.completion_tokens,
               llm_calls=stats.calls, rag_queries=0, escalado=True)
        yield text
        return

    # 4) RAG si hay duda explícita del paciente
    sources: list[Source] = []
    pregunta = ext.get("pregunta")
    if pregunta:
        sources = get_searcher().search(pregunta, escenario=p.escenario)
        state.rag_queries += 1
        for s in sources:
            state.sources_used.append(
                {"n": s.n, "doc_id": s.doc_id, "titulo": s.titulo,
                 "seccion": s.seccion, "paginas": [s.pagina_ini, s.pagina_fin],
                 "pregunta": pregunta, "turno": state.turno})

    # 5) Respuesta conversacional
    guidance = []
    if tri.faltantes:
        guidance.append("Antes de nada, indaga con tacto: " + "; ".join(tri.faltantes) + ".")
    if state.nivel == "amarillo" and prev == "verde":
        guidance.append("Hay síntomas que vigilar: reconócelo con calma. PROHIBIDO decir que es normal o quitarle importancia. Di que es algo que conviene revisar y que el equipo de enfermería lo contactará hoy mismo.")
    if pregunta and sources:
        guidance.append("Responde la duda usando SOLO las fuentes, citando [n].")
    elif pregunta and not sources:
        guidance.append("No hay fuentes para su duda: reconócelo y ofrece remitirla al equipo de salud.")
    nxt = None
    if not pregunta or len(guidance) == 0:
        nxt = _next_checklist_hint(state)
        if nxt:
            guidance.append(f"Luego pregunta por {nxt}.")
        elif not pregunta:
            guidance.append("El chequeo está completo: ofrece resolver dudas o despedirte con los próximos pasos.")

    if sources:
        # Respuesta factual con fuentes: llamada LIMPIA sin historial — el
        # prefill de la conversación entera + fuentes costaría 20-30 s en un
        # CPU de 4 núcleos. El contexto clínico va comprimido en una línea.
        ctx = (f"Paciente: {p.nombre.split()[0]}, {p.procedimiento}, día "
               f"postoperatorio {p.dia_postop}, triaje {state.nivel}.")
        msgs = [
            {"role": "system", "content":
             "Eres «Clara», asistente de voz de seguimiento postoperatorio en Colombia. "
             "Responde la duda del paciente en máximo 2-3 frases habladas, en español, "
             "trato de usted, usando SOLO las fuentes y citando [n]. Si las fuentes no "
             "responden la duda, dilo honestamente y ofrece remitirla al equipo de salud.\n\n"
             + prompts.PLANTILLA_FUENTES.format(fuentes=_format_sources(sources))},
            {"role": "user", "content": f"{ctx}\nDuda: {pregunta}\n({user_text})"},
        ]
        if guidance and not pregunta:
            msgs.append({"role": "system", "content": " ".join(guidance)})
    else:
        msgs = list(state.history)
        msgs.append({"role": "user", "content": user_text})
        if guidance:
            msgs.append({"role": "system", "content":
                         "INSTRUCCIÓN PARA ESTE TURNO (no la menciones): " + " ".join(guidance)})

    parts: list[str] = []
    try:
        for tok in llm.chat_stream(msgs, stats=stats):
            parts.append(tok)
            yield tok
    finally:
        # el barge-in abandona el generador: el historial DEBE quedar
        # consolidado igual (con lo parcial), o el siguiente turno no sabría
        # qué dijo el paciente ni qué alcanzó a responder el agente
        reply = "".join(parts)
        # si el turno se cortó sin respuesta, el dominio del guion vuelve a la
        # cola: de lo contrario nunca se volvería a preguntar
        if state.checklist_pendiente and len(reply) < 15:
            state.checklist_done.discard(state.checklist_pendiente)
        state.checklist_pendiente = None
        state.history.append({"role": "user", "content": user_text_final})
        state.history.append({"role": "assistant", "content": reply or "(interrumpido)"})
        if user_text_final != user_text and state.transcript:
            state.transcript[-1] = f"PACIENTE: {user_text_final}"
        state.transcript.append(f"AGENTE: {reply or '(interrumpido)'}")
    # poda de historial: en voz el contexto clínico vive en SymptomState, no
    # hace falta arrastrar toda la conversación (prefill caro en CPU)
    if len(state.history) > 15:
        state.history = state.history[:1] + state.history[-12:]

    tm.set(tokens_in=stats.prompt_tokens, tokens_out=stats.completion_tokens,
           llm_calls=stats.calls, rag_queries=1 if pregunta else 0,
           fuentes=[s.n for s in sources])


def close_call(state: CallState) -> dict:
    """Genera y persiste el resumen estructurado de la llamada."""
    if state.closed:
        return {}
    stats = llm.LLMStats()
    transcript = "\n".join(state.transcript) or "\n".join(
        f"{'AGENTE' if m['role'] == 'assistant' else 'PACIENTE'}: {m['content']}"
        for m in state.history if m["role"] in ("user", "assistant"))
    p = state.patient
    try:
        res = llm.structured(
            [{"role": "system", "content": prompts.SYSTEM_RESUMEN},
             {"role": "user",
              "content": f"Paciente: {p.nombre} ({p.edad} años). Procedimiento: "
                         f"{p.procedimiento}, día postoperatorio {p.dia_postop}.\n"
                         f"Triaje final: {state.nivel} ({'; '.join(state.razones)}).\n\n"
                         f"TRANSCRIPCIÓN:\n{transcript[-4000:]}"}],
            prompts.SCHEMA_RESUMEN, stats=stats, max_tokens=380,
        )
    except Exception as e:
        res = {"motivo_llamada": "seguimiento postoperatorio", "error_resumen": str(e)}
    summary = {
        "paciente": p.nombre, "paciente_id": p.paciente_id, "edad": p.edad,
        "procedimiento": p.procedimiento, "dia_postop": p.dia_postop,
        "triaje_final": state.nivel, "razones_triaje": state.razones,
        "sintomas_estructurados": state.symptoms.as_dict(),
        "alerta_generada": state.alerted,
        "referencias_usadas": state.sources_used,
        "turnos": state.turno,
        **res,
    }
    metrics.log_call_summary(state.call_id, summary)
    state.closed = True   # solo tras persistir: si falla, se puede reintentar
    return summary

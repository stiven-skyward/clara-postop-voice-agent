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
                              quick_red_scan, sanitize_extraction)
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


def _system_prompt(p: Patient) -> str:
    return prompts.SYSTEM_CONVERSACION.format(
        nombre=p.nombre, edad=p.edad,
        procedimiento=p.procedimiento, dia_postop=p.dia_postop,
    )


def greeting(state: CallState) -> str:
    p = state.patient
    text = (
        f"Hola {p.nombre.split()[0]}, le habla Clara, del programa de seguimiento "
        f"de su clínica. La llamo para saber cómo sigue después de su "
        f"{p.procedimiento.lower()}, que fue hace {p.dia_postop} "
        f"día{'s' if p.dia_postop != 1 else ''}. "
        f"Para empezar, ¿cómo ha estado el dolor? De 0 a 10, ¿en cuánto lo siente?"
    )
    state.history.append({"role": "system", "content": _system_prompt(p)})
    state.history.append({"role": "assistant", "content": text})
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
            state.checklist_done.add(key)
            return desc
    return None


def _format_sources(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        body = s.texto[:800]  # prefill acotado: latencia de voz en CPU
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


_CLINICAL_HINTS = re.compile(
    r"dolor|duele|fiebre|calentura|temperatura|herida|sangr|mareo|vomit|nause|"
    r"respir|hinch|roj|secre|liquid|pus|comer|apetito|dorm|sueño|camin|mover|"
    r"pierna|brazo|pecho|cabeza|barriga|estomago|estómago|punto|grado|\?|\d",
    re.IGNORECASE)


def _worth_extracting(text: str) -> bool:
    return len(text) > 80 or bool(_CLINICAL_HINTS.search(text))


def process_turn(state: CallState, user_text: str,
                 tm: metrics.TurnMetrics) -> Iterator[str]:
    """Procesa un turno del paciente. Genera el texto de respuesta en streaming."""
    state.turno += 1
    stats = llm.LLMStats()
    p = state.patient

    # 0) Atajo de seguridad: señal roja léxica → escalar YA (<1 s), sin esperar
    #    al LLM. La extracción corre después solo para completar el registro.
    señal = quick_red_scan(user_text)
    if señal and not state.alerted:
        state.nivel = "rojo"
        state.alerted = True
        text = _ESCALATION_MSG.format(nombre=p.nombre.split()[0])
        tm.set(triaje="rojo", escalado=True, atajo_lexico=señal)
        yield text
        state.history.append({"role": "user", "content": user_text})
        state.history.append({"role": "assistant", "content": text})
        try:
            ext = llm.structured(
                [{"role": "system", "content": prompts.SYSTEM_EXTRACCION},
                 {"role": "user", "content": user_text}],
                prompts.SCHEMA_EXTRACCION, stats=stats, max_tokens=260)
            state.symptoms.merge(sanitize_extraction(ext, user_text))
        except Exception:
            state.symptoms.otros.append(user_text[:120])
        tri = evaluate(state.symptoms, p.procedimiento, p.dia_postop)
        state.razones = tri.razones or [f"Señal de alarma textual: «{señal}»"]
        if f"Señal de alarma textual: «{señal}»" not in state.razones:
            state.razones.append(f"Señal de alarma textual: «{señal}»")
        metrics.log_alert(state.call_id, {
            "paciente": p.nombre, "paciente_id": p.paciente_id,
            "procedimiento": p.procedimiento, "dia_postop": p.dia_postop,
            "nivel": "rojo", "razones": state.razones,
            "sintomas": state.symptoms.as_dict(),
            "transcripcion_disparadora": user_text,
        })
        tm.set(tokens_in=stats.prompt_tokens, tokens_out=stats.completion_tokens,
               llm_calls=stats.calls, rag_queries=0)
        return

    # 1) Extracción estructurada (solo si el turno puede contener información
    #    clínica o una pregunta; un "sí señora" corto no paga LLM)
    ext = {}
    if _worth_extracting(user_text):
        try:
            ext = llm.structured(
                [{"role": "system", "content": prompts.SYSTEM_EXTRACCION},
                 {"role": "user", "content": user_text}],
                prompts.SCHEMA_EXTRACCION, stats=stats, max_tokens=260,
            )
        except Exception:
            ext = {"otros": [], "pregunta": None}
    ext = sanitize_extraction(ext, user_text)
    state.symptoms.merge(ext)

    # 2) Triaje determinista (solo sube)
    tri: TriageResult = evaluate(state.symptoms, p.procedimiento, p.dia_postop)
    prev = state.nivel
    state.nivel = combine(state.nivel, tri.nivel)
    if state.nivel != prev or not state.razones:
        state.razones = tri.razones
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
        guidance.append("Hay síntomas que vigilar: dilo con calma, sin alarmar ni restar importancia, y anuncia que el equipo de enfermería lo contactará hoy.")
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
    for tok in llm.chat_stream(msgs, stats=stats):
        parts.append(tok)
        yield tok
    reply = "".join(parts)

    state.history.append({"role": "user", "content": user_text})
    state.history.append({"role": "assistant", "content": reply})
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
    state.closed = True
    stats = llm.LLMStats()
    transcript = "\n".join(
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
    return summary

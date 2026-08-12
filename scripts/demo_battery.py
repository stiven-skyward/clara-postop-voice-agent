"""Batería E2E: una llamada de voz real por cada paciente del demo.

Simula al usuario hablando (TTS → PCM → WebSocket), captura STT + respuestas
de Clara + triaje + resumen, y escribe un informe JSON + Markdown.

Requiere el servidor en marcha (scripts/run.sh).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import numpy as np
from websockets.asyncio.client import connect

from app import config, tts

OUT_DIR = config.LOGS_DIR
REPORT_JSON = OUT_DIR / os.getenv("DEMO_REPORT_JSON", "demo_battery_report.json")
REPORT_MD = OUT_DIR / os.getenv("DEMO_REPORT_MD", "demo_battery_report.md")

# Señales de que Clara entró a RAG / consulta de fuentes.
_RAG_FILLER = re.compile(
    r"Perm[ií]tame revisar|no tengo esa informaci[oó]n en las fuentes|"
    r"no est[aá] incluida en las fuentes|fuentes disponibles",
    re.I)
# Turnos del paciente que son datos clínicos (NO deben disparar RAG).
_DECLARATIVE = re.compile(
    r"\b(?:dolor|fiebre|grados|temperatura|herida|secreci[oó]n|enrojec|"
    r"hinchaz|muy bien|normal|rojita)\b",
    re.I)
_REAL_QUESTION = re.compile(
    r"¿|\b(?:qu[eé]|c[oó]mo|cu[aá]ndo|d[oó]nde|por qu[eé]|puedo|debo|"
    r"una pregunta|quisiera|quiero saber)\b",
    re.I)

# Casos rotativos: cubren verde, amarillo, rojo, confirmación y pregunta RAG.
# Cada paciente recibe uno según índice.
CASOS = [
    {
        "id": "verde_bien",
        "esperado_triaje": {"verde"},
        "frases": [
            "La verdad muy bien, el dolor está como en dos y no he tenido fiebre.",
            "La herida se ve normal, sin enrojecimiento ni secreción.",
            "Muchas gracias, eso es todo.",
        ],
    },
    {
        "id": "amarillo_dolor_fiebre",
        "esperado_triaje": {"amarillo", "rojo"},
        "frases": [
            "El dolor está en siete.",
            "Actual, unos treinta y ocho grados.",
            "La herida se ve un poquito rojita, nada más.",
        ],
    },
    {
        "id": "rojo_alarma",
        "esperado_triaje": {"rojo"},
        "frases": [
            "El dolor está como en ocho y va subiendo, y anoche me midieron treinta y ocho nueve de fiebre.",
            "Sí señora, y la herida me está soltando un liquidito amarillo que huele feo.",
            "Sí, quedó claro.",
        ],
    },
    {
        "id": "verde_pregunta_bano",
        "esperado_triaje": {"verde", "amarillo"},
        "frases": [
            "El dolor está en tres y no tengo fiebre.",
            "Una pregunta, ¿cuándo me puedo bañar normal sin cuidar la herida del agua?",
            "Listo, muchas gracias.",
        ],
    },
    {
        "id": "confirmacion_corta",
        "esperado_triaje": {"rojo"},
        "frases": [
            "El dolor está en nueve y tengo fiebre de treinta y nueve.",
            "Sí, quedó claro.",
        ],
    },
]

_BAD_PATTERNS = [
    re.compile(r"gracias por compartir", re.I),
    re.compile(r"puedo ofrecerte|puedo ofrecerle la posibilidad", re.I),
    re.compile(r"y se quede a un lado", re.I),
    re.compile(r"no parece ser un problema grave", re.I),
]

# Tuteo: solo marcar si hay formas claras (excluye "usted", "actual", etc.)
_TUTEO = re.compile(
    r"\b(?:tú|tu temperatura|te recomiendo|puedes|debes|tienes)\b", re.I
)


def patient_audio(text: str) -> np.ndarray:
    pcm, sr = tts.synthesize(text)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    x = np.arange(0, len(audio), sr / 16000.0)
    a16 = np.interp(x, np.arange(len(audio)), audio).astype(np.float32)
    sil_s = config.VAD_SILENCE_MS / 1000.0 + 0.8
    sil = np.zeros(int(16000 * sil_s), dtype=np.float32)
    return np.concatenate([np.zeros(1600, np.float32), a16, sil])


def _issues_from_turn(said: str, stt: str | None, agent: str | None,
                      triaje: str | None) -> list[str]:
    issues = []
    if not stt:
        issues.append("STT vacío o ausente")
    elif said and stt:
        # similitud grosera: si el STT no conserva ninguna palabra clave del dicho
        said_w = {w for w in re.findall(r"[a-záéíóúñü]{3,}", said.lower())}
        stt_w = {w for w in re.findall(r"[a-záéíóúñü]{3,}", stt.lower())}
        if said_w and len(said_w & stt_w) / len(said_w) < 0.25:
            issues.append(f"STT muy distinto de lo dicho: «{stt}»")
    if agent:
        if _TUTEO.search(agent):
            issues.append("Posible tuteo en respuesta")
        for pat in _BAD_PATTERNS:
            if pat.search(agent):
                issues.append(f"Fórmula poco natural: {pat.pattern}")
        if agent.count("¿") > 1:
            issues.append("Más de una pregunta en el turno")
        # duplicación del protocolo de alerta
        if agent.count("Por lo que me cuenta") > 1:
            issues.append("Protocolo de alerta duplicado en el mismo texto")
    return issues


async def call_one(patient: dict, caso: dict, uri: str) -> dict:
    nombre = patient["nombre"]
    primer = nombre.split()[0]
    result = {
        "paciente_id": patient.get("paciente_id"),
        "nombre": nombre,
        "procedimiento": patient.get("procedimiento"),
        "escenario": patient.get("escenario"),
        "caso": caso["id"],
        "ok": True,
        "error": None,
        "saludo": None,
        "turnos": [],
        "summary": None,
        "triaje_final": None,
        "issues": [],
        "duracion_s": 0.0,
    }
    t0 = time.time()
    try:
        async with connect(uri, max_size=None, open_timeout=30) as ws:
            payload = {
                "type": "start",
                "sample_rate": 16000,
                "patient": {
                    "paciente_id": patient.get("paciente_id", ""),
                    "nombre": nombre,
                    "edad": patient.get("edad", 50),
                    "procedimiento": patient.get("procedimiento", "cirugía"),
                    "escenario": patient.get("escenario"),
                    "dia_postop": 3,
                },
            }
            await ws.send(json.dumps(payload))

            async def wait_turn(tag: str, timeout: float = 180.0) -> dict:
                out = {
                    "tag": tag, "stt": None, "agent": None, "triaje": None,
                    "alerta": False, "audio_chunks": 0, "s": 0.0, "error": None,
                }
                t1 = time.time()
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    if isinstance(msg, bytes):
                        out["audio_chunks"] += 1
                        continue
                    m = json.loads(msg)
                    typ = m.get("type")
                    if typ == "transcript":
                        out["stt"] = m.get("text")
                    elif typ == "transcript_fix":
                        out["stt"] = m.get("text")
                    elif typ == "agent_text":
                        out["agent"] = m.get("text")
                        out["triaje"] = m.get("triaje")
                        out["alerta"] = bool(m.get("alerta"))
                    elif typ == "error":
                        out["error"] = m.get("detail")
                    elif typ == "turn_end":
                        out["s"] = round(time.time() - t1, 1)
                        return out
                    elif typ == "call_summary":
                        # puede llegar si el servidor cierra; no es turn_end
                        result["summary"] = m.get("summary")
                        out["s"] = round(time.time() - t1, 1)
                        return out

            saludo = await wait_turn("saludo")
            result["saludo"] = saludo.get("agent")
            if saludo.get("agent") and primer.lower() not in saludo["agent"].lower():
                result["issues"].append(
                    f"Saludo no menciona el nombre «{primer}»")
            if not saludo.get("agent"):
                result["issues"].append("Sin saludo del agente")
                result["ok"] = False

            for i, frase in enumerate(caso["frases"], 1):
                audio = patient_audio(frase)
                for j in range(0, len(audio), 1024):
                    await ws.send(audio[j:j + 1024].tobytes())
                    await asyncio.sleep(0.008)
                turn = await wait_turn(f"turno{i}")
                turn["dijo"] = frase
                turn["issues"] = _issues_from_turn(
                    frase, turn.get("stt"), turn.get("agent"), turn.get("triaje"))
                # Dato clínico declarado (aunque el STT cuelgue un "?") no
                # debe activar la frase de espera RAG ni la respuesta de
                # «sin fuentes». Las preguntas reales sí pueden.
                agent = turn.get("agent") or ""
                said = frase
                if (_DECLARATIVE.search(said) and not _REAL_QUESTION.search(said)
                        and _RAG_FILLER.search(agent)):
                    msg = ("RAG espurio ante dato clínico (posible '?' "
                           f"de ASR): «{(turn.get('stt') or '')[:80]}»")
                    turn["issues"].append(msg)
                    result["ok"] = False
                turn["rag_filler"] = bool(_RAG_FILLER.search(agent))
                result["turnos"].append(turn)
                result["issues"].extend(
                    [f"T{i}: {x}" for x in turn["issues"]])
                if turn.get("triaje"):
                    result["triaje_final"] = turn["triaje"]
                if turn.get("error"):
                    result["ok"] = False
                    result["error"] = turn["error"]

            await ws.send(json.dumps({"type": "end"}))
            # esperar resumen
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=90)
                    if isinstance(msg, str):
                        m = json.loads(msg)
                        if m.get("type") == "call_summary":
                            result["summary"] = m.get("summary")
                            if m.get("summary", {}).get("triaje_final"):
                                result["triaje_final"] = m["summary"]["triaje_final"]
                            break
            except asyncio.TimeoutError:
                result["issues"].append("Timeout esperando resumen")

            esperado = caso["esperado_triaje"]
            if result["triaje_final"] and result["triaje_final"] not in esperado:
                result["issues"].append(
                    f"Triaje final «{result['triaje_final']}» fuera de {sorted(esperado)}")
                # rojo esperado y salió verde/amarillo es fallo duro
                if "rojo" in esperado and result["triaje_final"] != "rojo":
                    result["ok"] = False

            if result["issues"] and any(
                    x.startswith("STT") or "duplicado" in x or "Timeout" in x
                    for x in result["issues"]):
                result["ok"] = False

    except Exception as e:
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        result["issues"].append(result["error"])

    result["duracion_s"] = round(time.time() - t0, 1)
    return result


def _to_md(report: dict) -> str:
    lines = [
        f"# Batería demo — {report['ts']}",
        "",
        f"- Pacientes: **{report['total']}**",
        f"- OK: **{report['ok']}** · Fallos: **{report['fail']}**",
        f"- Duración total: **{report['duracion_total_s']} s**",
        "",
        "## Resumen por paciente",
        "",
        "| # | Paciente | Caso | Triaje | OK | Issues | s |",
        "|---|----------|------|--------|----|--------|---|",
    ]
    for i, r in enumerate(report["resultados"], 1):
        iss = "; ".join(r["issues"][:3]) if r["issues"] else "—"
        iss = iss.replace("|", "/")
        ok = "sí" if r["ok"] else "NO"
        lines.append(
            f"| {i} | {r['nombre'].split()[0]}… | {r['caso']} | "
            f"{r.get('triaje_final') or '—'} | {ok} | {iss} | {r['duracion_s']} |"
        )
    lines += ["", "## Detalle conversacional", ""]
    for r in report["resultados"]:
        lines.append(f"### {r['nombre']} — {r['caso']}")
        lines.append(f"- Procedimiento: {r['procedimiento']} · Triaje: {r.get('triaje_final')}")
        if r.get("saludo"):
            lines.append(f"- **Clara (saludo):** {r['saludo'][:280]}")
        for t in r.get("turnos", []):
            lines.append(f"- **Usted ({t['tag']}):** {t.get('dijo')}")
            lines.append(f"  - STT: «{t.get('stt') or '—'}»")
            ag = t.get("agent") or "—"
            lines.append(
                f"  - Clara ({t.get('triaje') or '—'}{' · ALERTA' if t.get('alerta') else ''}): {ag[:400]}")
        if r.get("issues"):
            lines.append("- Issues: " + "; ".join(r["issues"]))
        lines.append("")
    return "\n".join(lines)


async def main() -> int:
    host = "127.0.0.1"
    port = config.WEB_PORT
    base = f"http://{host}:{port}"
    uri = f"ws://{host}:{port}/ws/call"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{base}/api/patients")
        r.raise_for_status()
        patients = r.json()

    allow = os.getenv("DEMO_CASES", "").strip()
    allow_ids = {x.strip() for x in allow.split(",") if x.strip()} if allow else None
    plan: list[tuple[dict, dict]] = []
    for i, p in enumerate(patients):
        caso = CASOS[i % len(CASOS)]
        if allow_ids and caso["id"] not in allow_ids:
            continue
        plan.append((p, caso))

    print(f"Pacientes en demo: {len(patients)}")
    print(f"Llamadas a ejecutar: {len(plan)}"
          + (f" (filtro={sorted(allow_ids)})" if allow_ids else ""))
    print(f"WebSocket: {uri}")
    resultados = []
    t0 = time.time()
    for n, (p, caso) in enumerate(plan, 1):
        print(f"\n[{n}/{len(plan)}] {p['nombre']} · caso={caso['id']}")
        res = await call_one(p, caso, uri)
        resultados.append(res)
        flag = "OK" if res["ok"] else "FAIL"
        rag_turns = sum(1 for t in res["turnos"] if t.get("rag_filler"))
        print(f"  → {flag} triaje={res.get('triaje_final')} "
              f"rag_filler_turns={rag_turns} "
              f"issues={len(res['issues'])} ({res['duracion_s']}s)")
        for iss in res["issues"][:5]:
            print(f"     · {iss}")
        # breve pausa para no saturar llama-server / whisper
        await asyncio.sleep(1.0)

    false_rag = sum(
        1 for r in resultados
        for iss in r["issues"] if "RAG espurio" in iss)
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "filtro_casos": sorted(allow_ids) if allow_ids else None,
        "total": len(resultados),
        "ok": sum(1 for x in resultados if x["ok"]),
        "fail": sum(1 for x in resultados if not x["ok"]),
        "rag_espurio": false_rag,
        "duracion_total_s": round(time.time() - t0, 1),
        "resultados": resultados,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    REPORT_MD.write_text(_to_md(report), encoding="utf-8")
    print(f"\nInforme: {REPORT_MD}")
    print(f"JSON:    {REPORT_JSON}")
    print(f"OK={report['ok']} FAIL={report['fail']} "
          f"rag_espurio={report['rag_espurio']} "
          f"total={report['total']} en {report['duracion_total_s']}s")
    return 0 if report["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

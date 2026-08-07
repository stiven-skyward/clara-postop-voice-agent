"""Servidor: consola de administración + interfaz de llamada + WebSocket de voz.

Superficies exigidas por el reto (contrato funcional):
  /            → interfaz de llamada (micrófono ↔ voz del agente)
  /admin       → consola de conocimiento (subir / listar / eliminar, con estado)
  /ws/call     → audio PCM float32 @16 kHz del navegador; el servidor devuelve
                 JSON de control (frames de texto) y PCM int16 del TTS (binarios).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from app import config, llm, metrics, stt, tts
from app.agent.orchestrator import CallState, Patient, close_call, greeting, process_turn
from app.rag import ingest
from app.rag.store import get_store
from app.vad import StreamingVAD

app = FastAPI(title="Agente de seguimiento postoperatorio")
WEB = Path(__file__).resolve().parent.parent / "web"


@app.on_event("startup")
def _startup() -> None:
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    llm.ensure_server()
    # precalienta modelos para que la primera llamada no pague la carga
    threading.Thread(target=stt.get_model, daemon=True).start()
    threading.Thread(target=_warm_tts_phrases, daemon=True).start()
    threading.Thread(target=_warm_llm, daemon=True).start()


def _warm_tts_phrases() -> None:
    """Pre-sintetiza las frases fijas críticas (1ª oración del escalamiento)."""
    try:
        tts.synthesize_cached("Por lo que me cuenta, esto lo debe revisar el equipo médico hoy mismo.")
    except Exception:
        pass


def _warm_llm() -> None:
    """Prima la caché de prompt de ambos slots (extracción y conversación):
    el primer turno real solo prefill-ea los tokens nuevos."""
    from app.agent import prompts as pr
    try:
        llm.structured(
            [{"role": "system", "content": pr.SYSTEM_EXTRACCION},
             {"role": "user", "content": "hola"}],
            pr.SCHEMA_EXTRACCION, max_tokens=8)
        sys_conv = pr.SYSTEM_CONVERSACION.format(
            nombre="Paciente", edad=50, procedimiento="cirugía", dia_postop=1)
        list(llm.chat_stream(
            [{"role": "system", "content": sys_conv},
             {"role": "user", "content": "hola"}], max_tokens=4))
    except Exception:
        pass


@app.get("/")
def index():
    return FileResponse(WEB / "call.html")


@app.get("/admin")
def admin():
    return FileResponse(WEB / "admin.html")


# ---------- Consola de administración (G5: conocimiento vivo) ----------

@app.get("/api/docs")
def list_docs():
    return get_store().list_documents()


@app.post("/api/docs")
async def upload_doc(file: UploadFile):
    dest = config.UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    t0 = time.perf_counter()
    try:
        info = await asyncio.to_thread(ingest.ingest_file, dest, file.filename)
    except Exception as e:
        dest.unlink(missing_ok=True)
        return JSONResponse({"error": str(e)}, status_code=400)
    info["segundos_ingesta"] = round(time.perf_counter() - t0, 1)
    return info


@app.delete("/api/docs/{doc_id}")
async def delete_doc(doc_id: str):
    ok = await asyncio.to_thread(ingest.delete_document, doc_id)
    return {"doc_id": doc_id, "eliminado": ok}


@app.get("/api/metrics")
def get_metrics():
    return metrics.aggregate()


# ---------- Pacientes para el demo ----------

def _load_patients() -> list[dict]:
    try:
        import pandas as pd
        clin = pd.read_excel(
            config.DATASET_DIR / "perfiles_clinicos_pacientes_silver_contest.xlsx",
            sheet_name="result")
        demo = pd.read_excel(config.DATASET_DIR / "perfiles_pacientes_co.xlsx",
                             sheet_name="result")
        m = clin.merge(demo, on="paciente_id")
        esc_map = {"Apendicectomía": "apendicectomia", "Colecistectomía": "colecistectomia",
                   "Colectomía": "colectomia", "Mastectomía": "mastectomia",
                   "Reemplazo de cadera/rodilla": "reemplazo_articular"}
        out = []
        for _, r in m.iterrows():
            out.append({
                "paciente_id": r["paciente_id"], "nombre": r["nombre_completo"],
                "edad": int(r["edad"]), "genero": r.get("genero", ""),
                "procedimiento": r["procedimiento"], "ciudad": r.get("ciudad", ""),
                "escenario": esc_map.get(r["procedimiento"], None),
            })
        return out
    except Exception:
        return []


_PATIENTS: list[dict] | None = None


@app.get("/api/patients")
def patients():
    global _PATIENTS
    if _PATIENTS is None:
        _PATIENTS = _load_patients()
    if not _PATIENTS:
        return [{"paciente_id": "demo", "nombre": "Paciente de Prueba", "edad": 52,
                 "procedimiento": "Apendicectomía", "escenario": "apendicectomia"}]
    return _PATIENTS


# ---------- Llamada de voz ----------

@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    out_q: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()
    vad = StreamingVAD()
    state: CallState | None = None
    worker: threading.Thread | None = None

    def send(msg) -> None:  # llamable desde hilos
        loop.call_soon_threadsafe(out_q.put_nowait, msg)

    async def sender():
        while True:
            msg = await out_q.get()
            if msg is None:
                break
            try:
                if isinstance(msg, (bytes, bytearray)):
                    await ws.send_bytes(msg)
                else:
                    await ws.send_text(json.dumps(msg, ensure_ascii=False))
            except Exception:
                break

    send_task = asyncio.create_task(sender())
    last_activity = {"t": time.monotonic(), "nudges": 0}

    async def silence_watchdog():
        """Si el paciente calla >15 s sin turno en curso, Clara retoma."""
        while True:
            await asyncio.sleep(5)
            idle = time.monotonic() - last_activity["t"]
            busy = worker is not None and worker.is_alive()
            if state is not None and not busy and idle > 15 and last_activity["nudges"] < 3:
                last_activity["t"] = time.monotonic()
                last_activity["nudges"] += 1
                frase = ("¿Sigue ahí? Tómese su tiempo, no hay afán."
                         if last_activity["nudges"] < 3 else
                         "Parece que se cortó la comunicación. Si me escucha, "
                         "recuerde que puede volver a llamar cuando quiera. "
                         "Que siga muy bien.")
                def _nudge(f=frase):
                    pcm, sr = tts.synthesize_cached(f)
                    if pcm:
                        send({"type": "audio_start", "sr": sr, "text": f})
                        send(bytes(pcm))
                threading.Thread(target=_nudge, daemon=True).start()

    watchdog_task = asyncio.create_task(silence_watchdog())

    def speak(text: str, tm: metrics.TurnMetrics | None, cached: bool = False) -> None:
        """Sintetiza y envía una oración; marca primer audio si aplica."""
        pcm, sr = (tts.synthesize_cached if cached else tts.synthesize)(text)
        if not pcm:
            return
        if tm is not None and "t_tts_first_audio" not in tm.d:
            tm.mark("tts_first_audio")
        send({"type": "audio_start", "sr": sr, "text": text})
        send(bytes(pcm))

    def run_turn(audio: np.ndarray) -> None:
        try:
            _run_turn(audio)
        except Exception as e:
            send({"type": "error", "detail": str(e)})
            send({"type": "turn_end"})

    def _run_turn(audio: np.ndarray) -> None:
        assert state is not None
        tm = metrics.TurnMetrics(state.call_id, state.turno + 1)
        tm.mark("speech_end")
        text = stt.transcribe(audio)
        tm.mark("stt_done")
        if not text:
            send({"type": "no_speech"})
            return
        send({"type": "transcript", "text": text})
        streamer = tts.SentenceStreamer()
        first_tok = False
        reply_parts = []
        for tok in process_turn(state, text, tm):
            if cancel.is_set():
                break
            if not first_tok:
                tm.mark("llm_first_token")
                first_tok = True
            reply_parts.append(tok)
            for sent in streamer.push(tok):
                # la 1ª oración del escalamiento rojo está pre-sintetizada
                speak(sent, tm, cached=sent.startswith("Por lo que me cuenta"))
        rest = streamer.flush()
        if rest and not cancel.is_set():
            speak(rest, tm)
        tm.mark("turn_done")
        send({"type": "agent_text", "text": "".join(reply_parts),
              "triaje": state.nivel, "alerta": state.alerted})
        tm.save()
        send({"type": "turn_end"})
        last_activity["t"] = time.monotonic()  # el silencio cuenta desde aquí

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "start":
                    p = data.get("patient", {})
                    state = CallState(Patient(
                        nombre=p.get("nombre", "Paciente"),
                        edad=int(p.get("edad", 50)),
                        procedimiento=p.get("procedimiento", "cirugía"),
                        dia_postop=int(p.get("dia_postop", 3)),
                        escenario=p.get("escenario"),
                        paciente_id=p.get("paciente_id", ""),
                    ))
                    text = greeting(state)

                    def _greet(t=text):
                        st = tts.SentenceStreamer()
                        for sent in st.push(t):
                            speak(sent, None)
                        rest = st.flush()
                        if rest:
                            speak(rest, None)
                        send({"type": "agent_text", "text": t,
                              "triaje": "verde", "alerta": False})
                        send({"type": "turn_end"})
                    threading.Thread(target=_greet, daemon=True).start()
                elif data.get("type") == "end":
                    break
            elif msg.get("bytes") and state is not None:
                samples = np.frombuffer(msg["bytes"], dtype=np.float32)
                for ev, audio in vad.feed(samples):
                    if ev == "speech_start":
                        cancel.set()  # barge-in: corta la síntesis en curso
                        last_activity["t"] = time.monotonic()
                        last_activity["nudges"] = 0
                        send({"type": "user_speech_start"})
                    elif ev == "speech_end" and audio is not None:
                        last_activity["t"] = time.monotonic()
                        if worker is not None and worker.is_alive():
                            continue  # aún procesando el turno anterior
                        cancel.clear()
                        worker = threading.Thread(
                            target=run_turn, args=(audio,), daemon=True)
                        worker.start()
    except WebSocketDisconnect:
        pass
    finally:
        cancel.set()
        watchdog_task.cancel()
        if state is not None:
            summary = await asyncio.to_thread(close_call, state)
            try:
                await ws.send_text(json.dumps(
                    {"type": "call_summary", "summary": summary}, ensure_ascii=False))
            except Exception:
                pass
        out_q.put_nowait(None)
        await send_task


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT)

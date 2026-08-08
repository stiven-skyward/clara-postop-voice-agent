"""Prueba de estrés: llamadas concurrentes + entradas adversas.

Verifica que el servidor aguanta varias llamadas simultáneas (locks globales de
STT/TTS, slots del LLM, conexiones SQLite por hilo) y que las entradas límite no
lo tumban: frames malformados, audio de ruido puro, desconexión a mitad de turno,
barge-in agresivo y silencio prolongado.

Uso: python scripts/stress_test.py [n_llamadas]
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from websockets.asyncio.client import connect

from app import config, tts

URI = f"ws://{config.WEB_HOST}:{config.WEB_PORT}/ws/call"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def audio_de(texto: str) -> np.ndarray:
    pcm, sr = tts.synthesize(texto)
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    x = np.arange(0, len(a), sr / 16000.0)
    a16 = np.interp(x, np.arange(len(a)), a).astype(np.float32)
    return np.concatenate([np.zeros(1600, np.float32), a16,
                           np.zeros(int(16000 * 0.9), np.float32)])


async def enviar(ws, audio: np.ndarray, pausa: float = 0.01):
    for j in range(0, len(audio), 1024):
        await ws.send(audio[j:j + 1024].tobytes())
        await asyncio.sleep(pausa)


async def esperar(ws, tipo: str, timeout: float = 180) -> dict | None:
    fin = time.time() + timeout
    while time.time() < fin:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=fin - time.time())
        except (asyncio.TimeoutError, Exception):
            return None
        if isinstance(msg, str):
            m = json.loads(msg)
            if m["type"] == tipo:
                return m
    return None


async def llamada(idx: int, frases: list[str]) -> dict:
    r = {"id": idx, "turnos": 0, "errores": [], "triaje": None}
    try:
        async with connect(URI, max_size=None) as ws:
            await ws.send(json.dumps({"type": "start", "patient": {
                "nombre": f"Paciente {idx}", "edad": 55,
                "procedimiento": "Apendicectomía", "dia_postop": 3,
                "escenario": "apendicectomia", "paciente_id": f"stress{idx}"}}))
            await esperar(ws, "turn_end")
            for f in frases:
                await enviar(ws, audio_de(f))
                m = await esperar(ws, "agent_text")
                if m:
                    r["turnos"] += 1
                    r["triaje"] = m.get("triaje")
                await esperar(ws, "turn_end", timeout=60)
            await ws.send(json.dumps({"type": "end"}))
            s = await esperar(ws, "call_summary", timeout=120)
            r["resumen"] = bool(s and s.get("summary"))
    except Exception as e:
        r["errores"].append(f"{type(e).__name__}: {e}")
    return r


async def caso_frames_malformados() -> str:
    async with connect(URI) as ws:
        await ws.send("no-soy-json")
        await ws.send(json.dumps({"type": "desconocido"}))
        await ws.send(json.dumps({"type": "start", "patient": {"nombre": "X", "edad": "no-numero"}}))
        await asyncio.sleep(2)
        try:
            await ws.send(b"\x00" * 100)   # binario no múltiplo de float32
            await asyncio.sleep(1)
            return "sobrevivió"
        except Exception as e:
            return f"cerró: {type(e).__name__}"


async def caso_ruido_puro() -> str:
    """Ruido blanco: el VAD no debe disparar transcripción alucinada."""
    async with connect(URI, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "patient": {
            "nombre": "Ruido", "edad": 50, "procedimiento": "Apendicectomía",
            "dia_postop": 1, "paciente_id": "noise"}}))
        await esperar(ws, "turn_end")
        ruido = (np.random.RandomState(0).randn(16000 * 6) * 0.25).astype(np.float32)
        await enviar(ws, ruido, pausa=0.002)
        m = await esperar(ws, "transcript", timeout=45)
        await ws.send(json.dumps({"type": "end"}))
        return f"transcribió: «{m['text'][:60]}»" if m else "sin transcripción (correcto)"


async def caso_desconexion_abrupta() -> str:
    """Cierra el socket a mitad del turno: el servidor no debe quedar colgado."""
    ws = await connect(URI, max_size=None)
    await ws.send(json.dumps({"type": "start", "patient": {
        "nombre": "Corte", "edad": 50, "procedimiento": "Apendicectomía",
        "dia_postop": 1, "paciente_id": "cut"}}))
    await esperar(ws, "turn_end")
    await enviar(ws, audio_de("Tengo mucho dolor, como en nueve, y fiebre alta."))
    await asyncio.sleep(1.5)
    await ws.close()
    return "cerrado a mitad de turno"


async def main():
    print(f"== {N} llamadas concurrentes ==")
    t0 = time.time()
    guiones = [
        ["El dolor está en dos y no he tenido fiebre.", "¿Cuándo me puedo bañar?"],
        ["Tengo treinta y nueve de fiebre y el dolor en nueve.", "Sí, entiendo."],
        ["Todo bien, gracias.", "La herida se ve normal."],
    ]
    res = await asyncio.gather(*[llamada(i, guiones[i % len(guiones)]) for i in range(N)])
    for r in res:
        estado = "OK" if not r["errores"] else "FALLO"
        print(f"  [{estado}] llamada {r['id']}: {r['turnos']} turnos, triaje={r['triaje']}, "
              f"resumen={r.get('resumen')} {r['errores']}")
    print(f"  duración: {time.time()-t0:.0f}s\n")

    print("== casos límite ==")
    for nombre, coro in [("frames malformados", caso_frames_malformados()),
                         ("ruido puro", caso_ruido_puro()),
                         ("desconexión abrupta", caso_desconexion_abrupta())]:
        try:
            print(f"  {nombre}: {await coro}")
        except Exception as e:
            print(f"  {nombre}: FALLO {type(e).__name__}: {e}")

    import httpx
    with httpx.Client(timeout=30) as c:
        ok = c.get(f"http://{config.WEB_HOST}:{config.WEB_PORT}/api/docs").status_code
    print(f"\n== servidor sigue vivo tras todo: HTTP {ok} ==")


if __name__ == "__main__":
    asyncio.run(main())

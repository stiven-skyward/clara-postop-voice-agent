"""Prueba E2E de la llamada de voz: cliente WebSocket sintético.

Sintetiza la voz del "paciente" con el TTS, la envía como micrófono al
/ws/call y verifica: transcripción → respuesta del agente → audio de vuelta →
triaje → resumen al colgar. Requiere el servidor corriendo (scripts/run.sh).
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

FRASES = [
    "El dolor está como en ocho y va subiendo, y anoche me midieron treinta y ocho nueve de fiebre.",
    "Sí señora, y la herida me está soltando un liquidito amarillo que huele feo.",
]


def patient_audio(text: str) -> np.ndarray:
    pcm, sr = tts.synthesize(text)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    x = np.arange(0, len(audio), sr / 16000.0)
    a16 = np.interp(x, np.arange(len(audio)), audio).astype(np.float32)
    sil = np.zeros(int(16000 * 0.8), dtype=np.float32)
    return np.concatenate([np.zeros(1600, np.float32), a16, sil])


async def main():
    uri = f"ws://{config.WEB_HOST}:{config.WEB_PORT}/ws/call"
    async with connect(uri, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "patient": {
            "nombre": "Carlos Prueba", "edad": 58,
            "procedimiento": "Apendicectomía", "dia_postop": 3,
            "escenario": "apendicectomia", "paciente_id": "e2e"}}))

        async def wait_turn_end(tag: str):
            t0 = time.time()
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=180)
                if isinstance(msg, bytes):
                    print(f"  [{tag}] audio: {len(msg)//2} muestras")
                    continue
                m = json.loads(msg)
                if m["type"] == "transcript":
                    print(f"  [{tag}] STT: «{m['text']}»")
                elif m["type"] == "agent_text":
                    print(f"  [{tag}] AGENTE ({m['triaje']}{', ALERTA' if m.get('alerta') else ''}): {m['text'][:200]}")
                elif m["type"] == "turn_end":
                    print(f"  [{tag}] fin de turno en {time.time()-t0:.1f}s")
                    return

        await wait_turn_end("saludo")
        for i, frase in enumerate(FRASES, 1):
            print(f"\nPACIENTE dice: «{frase}»")
            audio = patient_audio(frase)
            for j in range(0, len(audio), 1024):
                await ws.send(audio[j:j + 1024].tobytes())
                await asyncio.sleep(0.01)
            await wait_turn_end(f"turno{i}")

        await ws.send(json.dumps({"type": "end"}))
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=120)
            if isinstance(msg, str):
                m = json.loads(msg)
                if m["type"] == "call_summary":
                    print("\nRESUMEN:", json.dumps(m["summary"], ensure_ascii=False, indent=1)[:1200])
                    return


if __name__ == "__main__":
    asyncio.run(main())

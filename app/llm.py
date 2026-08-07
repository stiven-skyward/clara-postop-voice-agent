"""Cliente de llama-server (llama.cpp) — chat streaming y salida estructurada.

Dos modos sobre el MISMO servidor/modelo:
  - chat_stream(): respuesta conversacional en streaming (SSE), con cache_prompt
    para que cada turno solo procese los tokens nuevos (crítico en CPU).
  - structured(): decodificación restringida por JSON-schema (GBNF) para el
    triaje — la salida inválida es imposible por construcción.
"""
from __future__ import annotations

import json
import subprocess
import time
from typing import Iterator

import httpx

from app import config

_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)


def ensure_server() -> None:
    """Arranca llama-server si no está corriendo y espera a que cargue el modelo."""
    try:
        httpx.get(f"{config.LLM_URL}/health", timeout=2.0)
        return
    except Exception:
        pass
    subprocess.Popen(
        [
            str(config.LLAMA_SERVER_BIN), "-m", str(config.LLM_MODEL),
            "-c", str(config.LLM_CTX), "-t", str(config.LLM_THREADS),
            "--host", config.LLM_HOST, "--port", str(config.LLM_PORT),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        time.sleep(1)
        try:
            r = httpx.get(f"{config.LLM_URL}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            continue
    raise RuntimeError("llama-server no arrancó en 120 s")


class LLMStats:
    """Métricas del último turno (para logging obligatorio de la rúbrica)."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0


def chat_stream(messages: list[dict], stats: LLMStats | None = None,
                max_tokens: int = 220, temperature: float = 0.35) -> Iterator[str]:
    """Genera la respuesta token a token (str incrementales)."""
    payload = {
        "messages": messages,
        "stream": True,
        "cache_prompt": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream_options": {"include_usage": True},
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        with client.stream("POST", f"{config.LLM_URL}/v1/chat/completions", json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage")
                if usage and stats is not None:
                    stats.prompt_tokens += usage.get("prompt_tokens", 0)
                    stats.completion_tokens += usage.get("completion_tokens", 0)
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
    if stats is not None:
        stats.calls += 1


def structured(messages: list[dict], schema: dict, stats: LLMStats | None = None,
               max_tokens: int = 300, temperature: float = 0.0) -> dict:
    """Llamada con salida JSON forzada por schema (GBNF interno de llama.cpp)."""
    payload = {
        "messages": messages,
        "stream": False,
        "cache_prompt": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "out", "schema": schema, "strict": True},
        },
    }
    with httpx.Client(timeout=_TIMEOUT) as client:
        r = client.post(f"{config.LLM_URL}/v1/chat/completions", json=payload)
        r.raise_for_status()
        obj = r.json()
    if stats is not None:
        usage = obj.get("usage", {})
        stats.prompt_tokens += usage.get("prompt_tokens", 0)
        stats.completion_tokens += usage.get("completion_tokens", 0)
        stats.calls += 1
    return json.loads(obj["choices"][0]["message"]["content"])

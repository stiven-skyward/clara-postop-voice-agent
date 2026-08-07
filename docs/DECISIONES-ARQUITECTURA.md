# Decisiones de arquitectura (ADR)

Basadas en investigación web exhaustiva (2026-08-07) + `REGLAS-FIJAS.md`.
Restricción rectora: **CPU-only, objetivo 8 GB RAM totales, todo local y gratis.**

## ADR-1 · Modelo de lenguaje: Llama 3.2 3B Instruct (Q4_K_M) sobre llama-server

**Elegido:** `Llama-3.2-3B-Instruct-Q4_K_M.gguf` servido con `llama-server` (llama.cpp),
contexto fijo `-c 4096`, `--flash-attn`, KV cache q8_0 si hace falta margen (448→224 MB),
`cache_prompt` activado (crítico: sin él, el prefill del system prompt en CPU añade 8-20 s
al primer token; con él, cada turno solo procesa la frase nueva → <2 s).

**Por qué (vs. alternativas permitidas):**
- **Phi-3.5 Mini (3.8B) — eliminado por RAM:** no tiene GQA (32 KV heads, MHA pura):
  su KV cache es ~768 KB/token ≈ **3 GB a 4K de contexto**, vs 448 MB del Llama 3B.
  Pesos (2.4 GB) + KV (3 GB) no caben en 8 GB compartidos. Además: español conversacional
  más rígido ("acartonado", entrenamiento mayoritariamente inglés) y ~30% menos tok/s.
- **Llama 3.2 1B — eliminado como modelo principal:** español notablemente más pobre
  (errores gramaticales, pierde el hilo multi-turno); insuficiente para empatía clínica.
  Queda como fallback degradado opcional (0.81 GB) si hay presión de RAM.
- **Cloud (Gemini 1.5 Flash / Groq 70B) — descartados como vía primaria:** dependen de
  internet y cuotas (Groq free: ~1.000 req/día); además Gemini 1.5 Flash está retirado en
  2026 y Groq decomisionó Llama 3.1 70B (hoy `llama-3.3-70b-versatile`). Un agente médico
  debe funcionar siempre. Nota: quedan como referencia para extrapolar costo por llamada (R6).
- **Arquitectura de 2 modelos (1B+3B) — descartada:** compiten por los mismos núcleos,
  +1.2 GB de RAM, y el triaje no necesita otro modelo: GBNF sobre el 3B ya garantiza JSON.
- **llama-server vs Ollama:** Ollama añade ~10-27% de overhead y prompt processing hasta
  ~10× más lento en tests medidos; llama-server da control total (hilos, `-c`, KV quant,
  gramáticas GBNF de primera clase). En CPU el prefill es el cuello del primer token.

**Triaje estructurado:** segunda llamada al mismo servidor con JSON-schema/GBNF forzando
`{"razon": "...", "triaje": "verde"|"amarillo"|"rojo"}` — la razón ANTES de la etiqueta
para preservar razonamiento. Validez sintáctica 100% por construcción (los logits de
tokens inválidos se anulan); precisión semántica reportada ~94% (XGrammar/SLOT, arXiv:2505.04016).

**Cifras esperadas (CPU 4-8 núcleos):** RAM total LLM ~2.5-2.9 GB · generación 8-25 tok/s
(el habla equivale a ~3-4 tok/s: alcanza para TTS en streaming) · primer token 1-2 s con caché.

## ADR-2 · Pipeline de voz: Silero VAD + whisper.cpp + Kokoro-82M (fallback Piper)

```
Navegador (getUserMedia, PCM 16 kHz, WebSocket)
  → Silero VAD ONNX (~2 MB, <1 ms/chunk; fin de habla = 400-500 ms de silencio)
  → whisper.cpp small-q5 (es) con initial_prompt de léxico médico/colombiano
  → LLM (streaming) → segmentación por oraciones (. ? ! :)
  → TTS por oración → chunks de audio al navegador (empieza a sonar con la 1ª oración)
```

- **STT elegido: whisper.cpp** — perfil por hardware:
  - 8 GB RAM: `small-q5_1` (~500 MB disco, ~1-1.3 GB runtime, WER es ~7%).
  - Plan B (CPU débil o RAM justa): `base-q5_1` (~500 MB runtime, WER es ~12%), libera ~1 GB.
  - Descartados: faster-whisper (misma velocidad en CPU mono-stream pero 1.5-2.3 GB RAM),
    Vosk (WER duplica a whisper-small; queda como opción extrema de 300 MB), Moonshine es
    (licencia no comercial), distil-whisper (solo inglés), sherpa-onnx zipformer (no hay español).
- **VAD: Silero VAD ONNX** — única opción seria en CPU; RTF ~0.004.
- **TTS principal: Kokoro-82M vía kokoro-onnx** (fp32 + `voices-v1.0.bin`), voz **ef_dora**
  (las voces es de Kokoro tienen pronunciación LATINOAMERICANA — ventaja para Colombia).
  RAM ~0.8 GB, RTF 0.1-0.35 (3-11× tiempo real), primer audio ~0.5-1.5 s/oración.
  int8 no ahorra RAM apreciable y puede ser más lento → fp32.
- **TTS fallback: Piper** (`es_MX-claude-high` o `es_ES-davefx-medium`): <100 MB RAM,
  RTF 0.008-0.03, primer audio ~40 ms. Menos natural, pero imbatible en latencia.
  Configurable por flag para demostrar operación en hardware mínimo.

**Presupuesto de RAM (dispositivo de 8 GB):**

| Componente | Principal | Modo ligero |
|---|---|---|
| SO + servidor base | ~1.5-2.0 GB | ~1.5-2.0 GB |
| LLM Llama 3.2 3B Q4_K_M (ctx 4K) | 2.5-2.9 GB | 2.5 GB (KV q8_0) |
| whisper.cpp | small-q5: ~1.2 GB | base-q5: ~0.5 GB |
| Kokoro ONNX / Piper | ~0.8 GB | ~0.1 GB |
| VAD + embeddings + índice | ~0.6 GB | ~0.6 GB |
| **Total** | **~6.6-7.5 GB** | **~5.2-5.7 GB** |

Ambos perfiles se implementan; el modo ligero es un flag de configuración.

**Latencia esperada (fin de habla → primer audio):** ~2.5-6 s según perfil; se optimiza
transcribiendo al cierre del VAD, TTS de la 1ª oración mientras el LLM sigue generando,
y streaming de chunks al navegador. Se mide y loggea P50/P95 automáticamente (R6).

## ADR-3 · RAG mejorado: híbrido BM25S+denso con sqlite-vec y small-to-big

**Presupuesto RAM del subsistema: ~600-800 MB pico.**

- **Embeddings: `multilingual-e5-small` int8 ONNX** (118M params, 384 dims, ~150-300 MB
  RAM, ~920 tok/s CPU). BGE-M3 (sugerido por el reto) **descartado**: 568M params,
  ~1.2-2.3 GB RAM — consumiría solo él casi todo el presupuesto del RAG en un equipo de
  8 GB. e5-small es cross-lingual (ES↔EN en el mismo espacio vectorial) y exige prefijos
  `query:` / `passage:` (omitirlos degrada mucho).
- **Vector store: sqlite-vec** (extensión C sin dependencias, un solo fichero .db).
  ChromaDB (sugerido por el reto) **descartado**: su delete en HNSW es soft-delete con
  tombstones y requiere rebuilds — el "olvido garantizado" de G5 es más demostrable con
  un `DELETE` transaccional SQL verificable por `SELECT`. A 10-25k vectores de 384 dims
  la búsqueda exacta es de milisegundos; no se necesita ANN.
- **Retrieval híbrido:** BM25S (léxico, sub-ms, órdenes de magnitud más rápido que
  rank-bm25) top-30 + denso top-30 → **fusión RRF** (k=60) → top-10 → reranker opcional
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` int8 (multilingüe, Apache-2.0, ~0.3-0.8 s
  CPU sobre 10 candidatos) → top-4/5. BM25 captura términos exactos (fármacos, guías);
  el denso, semántica cross-lingual.
- **Chunking estructural + small-to-big:** padres = secciones de la guía (~800-1200 tok,
  vía TOC/tamaños de fuente de PyMuPDF), hijos = ~250 tok con 15% solape e indexados con
  **header contextual** `[doc | sección | escenario]`. Se recupera por hijos y se entrega
  al LLM la sección padre (dedupe por parent_id): +15-25% precisión reportada, sin LLM extra.
- **Metadata filtering:** cada documento se clasifica en ingesta por escenario quirúrgico
  (apendicectomía, colecistectomía, colectomía, mastectomía, reemplazo articular) e idioma;
  cuando la llamada tiene procedimiento conocido, se filtra pre-búsqueda (~5× menos espacio).
- **Bilingüe (R8): NO se traduce el corpus.** Traducir en ingesta (Argos/opus-mt) rompería
  el conocimiento vivo (decenas de minutos por PDF en CPU) y la trazabilidad (la cita
  dejaría de ser el texto original) con riesgo de error en terminología médica. En su lugar:
  embeddings cross-lingual + **expansión bilingüe por diccionario** ES↔EN de términos del
  dominio en la rama BM25 + system prompt que fuerza respuesta SIEMPRE en español.
- **Extracción PDF: PyMuPDF** (~0.01 s/pág; TOC + fuentes para seccionar). Docling/
  unstructured hi-res descartados (500 MB+, 30-60 s de arranque — rompen ingesta en vivo).
  PDF sin capa de texto → **OCR automático** (tesseract/OCRmyPDF `spa+eng`) como subproceso.
- **HyDE descartado** en runtime (llamada LLM extra = segundos en CPU, más alucinación en
  modelos pequeños).
- **Conocimiento vivo (G5):** alta = pipeline de ingesta completo en segundos + rebuild
  atómico del BM25 en memoria (hot-swap); baja = transacción única
  `DELETE FROM chunks/documents WHERE doc_id=?` + rebuild BM25 → olvido total, inmediato
  y verificable. El manifiesto `documents` alimenta la consola de administración.
- **Trazabilidad:** bloques `[FUENTE n: título, sección, página]` en el prompt; el agente
  cita `[n]` y la app resuelve a documento/página literal del original.

**Detalle completo y fuentes:** `docs/investigacion/rag.md` (también `llm.md` y `voz.md`).

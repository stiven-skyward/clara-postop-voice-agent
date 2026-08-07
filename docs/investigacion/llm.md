# Informe: Selección de LLM para agente de voz médico en español (CPU, 8 GB RAM)

**Fecha:** 2026-08-07. Investigación web sobre los 3 candidatos permitidos + plan B cloud.

---

## 1. Calidad en español

### Llama 3.2 1B / 3B Instruct
- **Español oficialmente soportado.** El model card de Meta lista 8 idiomas soportados: inglés, alemán, francés, italiano, portugués, hindi, **español** y tailandés ([Meta AI blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/), [Ollama library](https://ollama.com/library/llama3.2)).
- Meta reporta evaluación multilingüe (MMLU multilingüe y MGSM promediados sobre esos idiomas, español incluido) en [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783). No hay tabla per-idioma pública para 3.2, pero el 3B es consistentemente más fuerte que el 1B en seguimiento de instrucciones, razonamiento y tareas multilingües; el 1B es competitivo solo dentro de su clase.
- El 1B en español conversacional es notablemente más pobre: comete errores gramaticales ocasionales, se repite y pierde el hilo en conversación multi-turno más rápido. Para "conversación empática" el salto 1B→3B es el más importante de toda esta decisión.

### Phi-3.5 Mini Instruct (3.8B)
- Microsoft lo presenta como multilingüe con soporte mejorado en ~20+ idiomas de "alto recurso" (español incluido), con MMLU-multilingüe 55.4 y MGSM 47.9 (0-shot CoT) — bueno para su tamaño ([Microsoft Tech Community](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/discover-the-new-multi-lingual-high-quality-phi-3-5-slms/4225280), [model card HF](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)).
- **Pero**: el propio model card advierte que los datos de entrenamiento son mayoritariamente inglés y que el rendimiento en otros idiomas es inferior. El sentimiento histórico de la comunidad (r/LocalLLaMA) sobre los Phi: excelentes en benchmarks, **conversación real "acartonada"/menos natural**, y más aún fuera del inglés. Existencia de finetunes comunitarios específicos por idioma (p. ej. [Phi-3.5-mini-ITA para italiano](https://huggingface.co/QuantFactory/Phi-3.5-mini-ITA-GGUF)) es señal indirecta de que el multilingüe "de fábrica" se queda corto para chat natural.

**Veredicto sección 1:** para español conversacional empático, **Llama 3.2 3B > Phi-3.5 Mini > Llama 3.2 1B**. Phi-3.5 gana en razonamiento puro; Llama 3B suena más natural en español.

## 2. Instrucciones estrictas / JSON estructurado (triaje verde/amarillo/rojo)

- **Con prompting solo, ninguno es fiable.** Llama 3.2 produce JSON malformado (comas colgantes, preámbulos, fences markdown) en un porcentaje no trivial de llamadas; un estudio midió ~16% de salidas imparseables en Llama-3, reducible a ~2% con few-shot + fences ([StructuredRAG](https://arxiv.org/pdf/2408.11061), [guía](https://llmconfigurator.com/en/guides/llm-json-structured-output)).
- **Con decodificación restringida, el problema desaparece:** Llama 3.2 1B + XGrammar alcanzó **93.92% de precisión de esquema sin fine-tuning** (96.24% con FT ligero) ([SLOT, arXiv:2505.04016](https://arxiv.org/html/2505.04016v1)). Con GBNF/JSON-schema en llama.cpp la salida sintácticamente inválida es **imposible** por construcción (se anulan los logits de tokens que violan la gramática) ([grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)).
- Para un clasificador de 3 clases (`{"triaje": "verde"|"amarillo"|"rojo", ...}`) una gramática GBNF trivial da 100% de validez sintáctica en cualquiera de los 3 modelos; la **exactitud semántica** de la clasificación sí depende del modelo: 3B/3.8B >> 1B en seguir protocolos con criterios (Phi-3.5 y Llama 3B, IFEval ~llegando a niveles de modelos 7B de la generación anterior).
- Caveat documentado: restricción dura puede degradar levemente el razonamiento; el patrón recomendado es **dos pasadas** (razonar libre → extraer con gramática) o incluir campo `"razon"` antes del campo `"triaje"` en el esquema ([Markaicode](https://markaicode.com/ollama-structured-output-pipeline/)).

## 3. Rendimiento en CPU (llama.cpp/Ollama, Q4_K_M)

Referencia directa: [edge-cpu-inference](https://github.com/nandan2003/edge-cpu-inference) en Azure D4s_v5 (4 vCPU Xeon, AVX2, 16 GB), llama.cpp Q4_K_M:

| Modelo | Tamaño GGUF Q4_K_M | RAM proceso aprox.* | tok/s generación (4 núcleos) | tok/s (8 núcleos modernos) |
|---|---|---|---|---|
| Llama 3.2 1B | **0.81 GB** | ~1.0–1.3 GB | ~20–30 (TinyLlama 1.1B midió 24.9) | 40–60 |
| Llama 3.2 3B | **2.02 GB** ([bartowski GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)) | ~2.4–2.9 GB | **~8–12** | 15–25 |
| Phi-3.5 Mini | **2.39 GB** ([bartowski GGUF](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF)) | ~2.8 GB **+ KV enorme (ver §4)** | ~6–8 (Phi-3 midió 6.4) | 10–18 |

\* pesos + buffers + KV cache a 2–4K de contexto. Cifras de la orden confirmadas: 1B ≈ 0.8–1.2 GB ✔, 3B ≈ 2–2.5 GB ✔ (2.02 GB archivo), Phi-3.5 ≈ 2.4 GB archivo (no 2.7–3; eso es con KV incluido).

- Confirmación cruzada: 8B Q4_K_M rinde ~12.5 t/s en CPU de gama alta y ~6.8 t/s en CPUs modestas; el 3B es ~2.5× más ligero ([Markaicode CPU benchmark](https://markaicode.com/benchmarks/tool-cpu-benchmark/), [MyAIHardware](https://www.myaihardware.com/llama-cpp-benchmarks)).
- No usar más hilos que núcleos físicos; más allá de 8 hilos no hay ganancia ([ibíd.](https://markaicode.com/benchmarks/tool-cpu-benchmark/)).
- **Primer token (<2–3 s):** el prefill en CPU para 3B ronda 30–80 t/s en 4 núcleos. Con system prompt de 500+ tokens el primer token tardaría 8–20 s **salvo que uses caché de prompt** (`cache_prompt` en llama-server, activado por defecto): el system prompt y el historial se procesan una sola vez y cada turno solo prefill-ea la frase nueva del usuario (~20–50 tokens → <2 s). Esto es un requisito de arquitectura, no opcional.

## 4. Contexto práctico / KV cache (el factor eliminatorio)

KV cache FP16 por token = 2 × capas × (kv_heads × head_dim) × 2 bytes ([GitHub #9936](https://github.com/ggml-org/llama.cpp/discussions/9936), [cálculo GQA](https://medium.com/@liu.peng.uppsala/key-value-kv-cache-size-calculations-in-grouped-query-attention-gqa-e090d3037ab3)):

| Modelo | Atención | KV/token FP16 | KV a 4K ctx | KV a 8K ctx |
|---|---|---|---|---|
| Llama 3.2 1B (16 capas, 8 KV heads, GQA) | GQA | 32 KB | 128 MB | 256 MB |
| Llama 3.2 3B (28 capas, 8 KV heads, GQA) | GQA | 112 KB | **448 MB** | 896 MB |
| Phi-3.5 Mini (32 capas, **32 KV heads, MHA — sin GQA**) | MHA | **768 KB** | **~3 GB** | ~6 GB |

- **Phi-3.5 Mini no tiene GQA** ([config.json](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/config.json), num_key_value_heads=32): su KV cache es ~7× el del Llama 3B. A 4K de contexto necesitarías 2.4 GB (pesos) + 3 GB (KV) ≈ 5.4 GB — **inviable en 8 GB compartidos con OS+STT+TTS+embeddings**. Incluso con KV q8_0 (`--cache-type-k/v q8_0`, requiere flash-attn) sigue en ~1.5 GB a 4K.
- Presupuesto realista en tu equipo: OS ~1.5–2 GB, Whisper/STT ~0.5–1 GB, TTS ~0.2–0.5 GB, embeddings ~0.3 GB, app ~0.5 GB → **~3–3.5 GB para el LLM**.
- **Llama 3.2 3B Q4_K_M + 4K de contexto ≈ 2.5–2.9 GB → cabe.** llama.cpp pre-asigna todo el KV al arrancar, así que fija `-c 4096` explícitamente (el default puede intentar 128K y reventar la RAM) ([discusión #9784](https://github.com/ggml-org/llama.cpp/discussions/9784)). 4K tokens ≈ 12–15 turnos de conversación de voz — suficiente para triaje con resumen/truncado de historial.

## 5. Ollama vs llama.cpp server vs llama-cpp-python

| Criterio | llama-server (nativo) | Ollama | llama-cpp-python |
|---|---|---|---|
| Velocidad | Referencia (la más rápida) | +10–27% overhead (capa Go; prompt processing hasta ~10× más lento en un test medido) | +12–18% overhead |
| Streaming SSE | Sí | Sí | Sí |
| JSON schema / gramática | **GBNF completo + `response_format`/`--json` (json-schema→GBNF integrado)** | Structured outputs (JSON schema vía `format`; usa la misma maquinaria GBNF por debajo, menos expresivo) | GBNF completo (API Python) |
| Control (hilos, `-c`, KV quant, cache_prompt) | Total | Limitado | Total |
| Facilidad | Media | Máxima | Media |

Fuentes: [InventiveHQ benchmark](https://inventivehq.com/blog/ollama-vs-llama-cpp-vs-lm-studio-benchmark), [openxcell](https://www.openxcell.com/blog/llama-cpp-vs-ollama), [kunalganglani](https://www.kunalganglani.com/blog/ollama-vs-llama-cpp), [Ollama structured outputs](https://ollama.com/blog/structured-outputs), [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md), [issue llama-cpp-python #398](https://github.com/abetlen/llama-cpp-python/issues/398).

**Recomendación:** `llama-server` nativo. En CPU, donde cada punto porcentual de prefill cuenta para el primer token, el ~10× de ventaja en prompt processing medido frente a Ollama y el soporte GBNF de primera clase lo hacen la opción clara. Ollama vale para prototipar en 5 minutos.

## Plan B cloud (breve)

- **Ojo con las reglas del concurso:** ambas opciones cloud nombradas están desactualizadas en 2026. **Gemini 1.5 Flash está retirado** (la serie 1.5 se retiró en 2025; hoy el free tier de Gemini API cubre solo Flash/Flash-Lite de generación 3.x, con cuotas recortadas desde abril 2026 — [pricing oficial](https://ai.google.dev/gemini-api/docs/pricing)). **Llama 3.1 70B en Groq fue decomisionado** en favor de `llama-3.3-70b-versatile`: free tier ~30 RPM / 1,000 req/día / 6K TPM, sin tarjeta ([GroqDocs](https://console.groq.com/docs/model/llama-3.3-70b-versatile), [límites 2026](https://tokenmix.ai/blog/groq-free-tier-limits-2026)).
- Como **híbrido** tiene sentido: local 3B para todo el flujo + escalado opcional a Groq 70B solo para casos ambiguos de triaje (el 70B es muy superior en español y protocolos, y Groq da <1 s de latencia). Pero 1,000 req/día y dependencia de red lo descartan como vía primaria para un agente médico que debe funcionar siempre.

## 6. Recomendación final

**Modelo único: Llama 3.2 3B Instruct Q4_K_M sobre llama-server, contexto 4096.** Phi-3.5 Mini queda **eliminado** por su KV cache MHA (768 KB/token, ~3 GB a 4K) que no cabe en tu presupuesto de RAM, además de español conversacional más rígido y ~30% menos tok/s. La arquitectura de 2 modelos (1B chat + reglas) **no compensa**: en 4 núcleos ambos compitiendo por CPU degradan la latencia mutua, sumas ~1.2 GB de RAM extra, y el 1B es demasiado débil para empatía en español; la clasificación de triaje no necesita un segundo modelo porque GBNF sobre el mismo 3B ya garantiza el JSON.

Configuración concreta:
1. `llama-server -m Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 4096 -t <núcleos físicos> --flash-attn` (+ `-ctk q8_0 -ctv q8_0` si necesitas margen de RAM: KV 4K baja de 448 a 224 MB).
2. Chat empático: streaming SSE normal, system prompt corto (<300 tokens) y `cache_prompt: true` → primer token <2 s tras el primer turno.
3. Triaje: segunda llamada al mismo servidor con `response_format`/GBNF forzando `{"razon": "...", "triaje": "verde|amarillo|rojo"}` (razón antes de la etiqueta para preservar el razonamiento). Validez sintáctica 100% garantizada; fiabilidad semántica ~94%+ según literatura (XGrammar/SLOT).
4. Cifras esperadas en tu hardware: ~2.5–2.9 GB RAM totales del LLM, 8–12 tok/s de generación (suficiente para TTS en streaming: el habla humana son ~3–4 tok/s equivalentes), primer token 1–2 s con caché de prompt.
5. Fallback degradado opcional: tener el 1B Q4_K_M (0.81 GB) en disco por si la telemetría muestra presión de RAM/latencia, y Groq `llama-3.3-70b-versatile` como verificador cloud de casos amarillo/rojo cuando haya red.
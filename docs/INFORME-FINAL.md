# Informe final — Clara, agente de voz de seguimiento postoperatorio

> Entregable 03 del Tech Sphere Challenge 2026. Evidencia del proceso: decisiones,
> prompts, configuraciones y métricas. Complementa al [README](../README.md) y a los
> [ADRs](DECISIONES-ARQUITECTURA.md).

## 1. Declaración de modelo (compuerta G3)

**Modelo usado: Llama 3.2 3B Instruct, cuantizado Q4_K_M (GGUF), servido localmente
en CPU con `llama-server` (llama.cpp b10313), contexto 4096.**

### Por qué este y no los otros tres permitidos

| Candidato | Veredicto | Razón |
|---|---|---|
| Gemini 1.5 Flash (nube) | ✗ | Dependencia de internet y de cuotas para un agente médico que debe funcionar siempre; además la serie 1.5 está retirada del free tier en 2026. Nuestro requisito rector era operar en un equipo básico de 8 GB sin GPU y sin costo. |
| Llama 3.1 70B vía Groq (nube) | ✗ | Misma dependencia de red; free tier ~1.000 req/día no sostiene operación continua. Se conserva como referencia de costo. |
| Llama 3.2 1B (local) | ✗ como principal | Español conversacional pobre (errores gramaticales, pierde el hilo); insuficiente para empatía clínica. |
| **Llama 3.2 3B (local)** | **✓** | Español oficialmente soportado y natural; GQA → KV cache de solo 448 MB a 4K (vs ~3 GB de Phi-3.5, que es MHA pura); 2.0 GB de pesos; 8-25 tok/s en CPU modesta; JSON fiable con decodificación restringida. |
| Phi-3.5 Mini 3.8B (local) | ✗ | **Eliminatorio: sin GQA su KV cache es ~768 KB/token (~3 GB a 4K)** — no cabe en 8 GB junto a STT+TTS+RAG+SO. Además español más rígido y ~30 % menos tok/s. |

Investigación de soporte con fuentes: [investigacion/llm.md](investigacion/llm.md).

## 2. La decisión técnica más relevante: el LLM no decide el triaje

Durante las pruebas de integración el 3B, con salida JSON forzada, **clasificó
"verde" un caso rojo evidente** (fiebre 38.9 °C + dolor 8/10 en aumento + secreción
purulenta). Con prompt mejorado subió solo a "amarillo". En salud, ese falso negativo
es la falla catastrófica.

**Solución adoptada (arquitectura extractor + reglas):**

1. El LLM **extrae** los síntomas del habla libre a JSON con esquema forzado
   (decodificación restringida GBNF: JSON inválido imposible por construcción).
   En esta tarea el 3B es fiable incluso con jerga regional: "me está soltando un
   liquidito amarillo que huele feo" → `herida: secrecion_purulenta`.
2. Un **motor de reglas determinista** (`app/agent/triage.py`), derivado de los
   signos de alarma de las guías del corpus, decide verde/amarillo/rojo. Es
   auditable línea a línea, testeable contra el dataset, y sesgado a seguridad:
   el nivel **solo puede subir** durante la llamada, y ante ambigüedad (fiebre
   "subjetiva" sin termómetro, dolor sin cuantificar) genera indagaciones antes
   de decidir.

**Alternativas evaluadas:** triaje por LLM con few-shot (descartado: subestima),
triaje por LLM 70B en nube (descartado: dependencia de red; queda como verificador
opcional), clasificador ML entrenado con el dataset (descartado: 160 casos son pocos
y la explicabilidad clínica de las reglas es superior).

**Calibración medida contra el dataset (evidencia de iteración):**

- **v1 de las reglas:** 0 falsos negativos y recall rojo 12/12 sobre 57 casos
  (12 rojos + 25 amarillos + 20 verdes, capa limpia)… pero **45 falsos positivos**:
  ningún caso quedaba en verde. Dos causas: el LLM pequeño *infiere* síntomas no
  dichos ("el dolor no baja" → `dolor_empeora`; "un poquito de enrojecimiento,
  se ve normal" → `herida: enrojecida`), y las reglas contaban molestias normales
  de recuperación (sueño irregular, mareo leve) como señales amarillas.
- **v2 (calibrada):** (a) **anclaje léxico** de la extracción — una bandera booleana
  solo se acepta si el texto del paciente contiene un lexema compatible
  (`sanitize_extraction`), lo que neutraliza la sobre-inferencia del 3B de forma
  determinista; (b) poda de keywords benignos; (c) febrícula 38.0-38.4 + herida
  alterada leve refuerza el amarillo en vez de saltar a rojo; (d) la sensación
  febril sin termómetro ya no sube el nivel: genera una *indagación* ("¿puede
  medirse la temperatura?") antes de decidir — exactamente lo que la rúbrica pide
  ante la ambigüedad.
- **v2 destapó el reto real del dataset:** aparecieron falsos negativos en casos
  rojos de **pacientes minimizadores** — trayectoria real con dolor 9/10 relatada
  como "un poquito molesto, uno aguanta", o secreción purulenta descrita como
  "un liquidito amarillito, normal de la sanada".
- **v3 (final):** tres defensas deterministas contra la minimización:
  (a) la secreción amarilla/verde o con mal olor es purulenta **siempre**, por
  más que el paciente la reste importancia; (b) **detector de incongruencia por
  señales blandas**: la degradación *simultánea* de varios dominios (herida
  rojita + sensación febril + apetito caído + sueño alterado + dolor evadido
  sin cuantificar) es en sí una señal — ≥3 dominios → amarillo, ≥4 → rojo;
  (c) guardia de **negación** en el atajo léxico rojo ("nada de pus" ya no
  dispara). Además, el agente en vivo *indaga* ante señales evadidas (pide el
  número de dolor, pide medir la temperatura), cosa que la evaluación sobre
  diálogos congelados no puede capturar.
- **Calibración final por barrido:** en vez de ajustar umbrales a mano (cada
  evaluación completa toma ~20 min de LLM en CPU), los estados de síntomas por
  caso se persisten en JSON y las variantes de reglas se barren **offline en
  segundos** (16 combinaciones de umbral/compuertas). La configuración elegida
  —≥2 dominios blandos → amarillo, ≥3 → rojo, "el termómetro manda" sobre la
  sensación febril— logra **0 falsos negativos y recall rojo 12/12** sobre los
  57 casos, aceptando deliberadamente sobre-escalamiento (asimetría clínica del
  reto). Los 3 verdes→rojo restantes son pacientes ansiosos que *declararon*
  dolor 8/10 o secreción: escalarlos es lo correcto con la información disponible
  por teléfono.
- Resultado final: ver métricas del README (detalle por caso en
  `data/logs/eval_triage_capa1_limpia.json`).

**Riesgos identificados:** (1) reglas incompletas ante síntomas no previstos —
mitigado con listas de red flags textuales y con el merge de "otros" síntomas;
(2) error de extracción — mitigado con esquema cerrado y umbral conservador;
(3) el paciente no menciona el síntoma — mitigado con guion de chequeo sistemático
de 5 dominios.

**Con dos semanas más:** validaría las reglas con clínicos; añadiría verificación
de citas post-generación (entailment ligero); afinaría Whisper con acento colombiano
(fine-tune LoRA); y probaría el reranker mmarco-mMiniLMv2 int8 medido A/B.

## 3. RAG y conocimiento vivo

Pipeline y justificación completa en [ADR-3](DECISIONES-ARQUITECTURA.md) e
[investigacion/rag.md](investigacion/rag.md). Puntos clave frente al RAG convencional:

- **Híbrido** BM25S + denso (multilingual-e5-small int8 ONNX) con fusión RRF: la rama
  léxica ancla términos exactos (fármacos, nombres de guías) y la densa cruza idiomas.
- **Corpus bilingüe sin traducción**: la cita siempre es el texto original (integridad
  de trazabilidad); expansión bilingüe por diccionario médico ES↔EN en la rama léxica;
  el LLM lee inglés y responde siempre en español.
- **Small-to-big**: se busca por chunks hijos (~250 tok con header contextual
  documento|sección) y se entrega al LLM la sección padre completa → precisión de
  búsqueda + contexto suficiente para no alucinar.
- **Conocimiento vivo real**: alta = ingesta en segundos (extracción estructural
  PyMuPDF, OCR automático si el PDF es escaneado) + hot-swap del índice BM25;
  baja = `DELETE` transaccional en sqlite-vec — sin tombstones ni índices
  fantasma; el olvido es verificable con un SELECT.
- **Filtro por escenario quirúrgico** en ambas ramas cuando el procedimiento del
  paciente se conoce.
- **Trazabilidad**: el prompt entrega bloques `[FUENTE n: doc, sección, páginas]`;
  el agente cita `[n]`; la app registra en el resumen de llamada qué documento
  sustenta cada respuesta.

## 4. Prompts (evidencia)

Los prompts de producción viven en `app/agent/prompts.py` (versionados en git):
`SYSTEM_CONVERSACION` (reglas inquebrantables, anti-inyección, formato de voz),
`SYSTEM_EXTRACCION` + `SCHEMA_EXTRACCION` (síntomas a JSON), `SYSTEM_RESUMEN` +
`SCHEMA_RESUMEN` (resumen estructurado de llamada). Iteraciones relevantes:

- v1 triaje por LLM → descartado por subestimación (sección 2).
- v2 extracción: se añadió el campo `pregunta` (detector de dudas) para disparar el
  RAG solo cuando hay duda real → menos consultas, menos latencia.
- v3 conversación: "usted" obligatorio, validación emocional ante miedo, y la
  instrucción de turno se inyecta como mensaje `system` efímero (no contamina el
  historial).

## 5. Métricas y costo

Ver tabla del README (§Métricas) — se generan de `data/logs/*.jsonl`, con
`GET /api/metrics` como agregador. Metodología de costo: tokens medidos por llamada
× tarifas API públicas (Groq Llama-3.3-70B y Gemini Flash) como extrapolación,
$0 real en local.

## 6. Seguridad y comportamiento adverso

- Inyección de prompt: reglas inquebrantables en system prompt + prueba adversaria
  (`scripts/`): el agente rechaza "ignora tus instrucciones" y redirige a su misión.
- Medicación: prohibido recetar/ajustar dosis; probado con petición de doble dosis
  de tramadol → rechazo y remisión al médico.
- Honestidad: si el RAG no aporta fuentes, el agente lo dice y remite al equipo
  de salud (sin improvisar).
- Silencios/ruido: VAD con umbral y filtro de alucinaciones típicas de Whisper.
- Barge-in: si el paciente interrumpe, la síntesis en curso se cancela.

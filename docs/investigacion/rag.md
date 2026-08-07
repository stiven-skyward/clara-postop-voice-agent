He completado la investigación web. A continuación el informe estructurado.

---

# Informe: RAG "mejorado" para agente médico de voz (ES), 100% CPU, 8 GB RAM

**Contexto asumido de presupuesto RAM:** LLM ~2.5 GB + voz (STT/TTS) ~1–1.5 GB + SO ~1 GB → **quedan ~1–1.5 GB para todo el RAG**. Corpus: 107 PDFs (≈ 10–25k chunks estimados), 5 escenarios quirúrgicos, ES+EN.

---

## 1. Embeddings multilingües ligeros para CPU

| Modelo | Params | Dim | RAM aprox. | MIRACL nDCG@10 | Veredicto |
|---|---|---|---|---|---|
| **BGE-M3** | 568M | 1024 | ~1.2 GB (Q4) / 2.3 GB fp32 | **67.8–69.0** | Mejor calidad, pero **demasiado pesado**: solo el modelo consumiría casi todo el presupuesto RAG |
| **multilingual-e5-base** | 278M | 768 | ~1.1 GB fp32 | 62.5 | Compromiso viable si sobra RAM |
| **multilingual-e5-small** | 118M | 384 | ~470 MB fp32 / **~113 MB int8 ONNX** | 60.8 (ES: 50.8) | **Recomendado** |
| paraphrase-multilingual-MiniLM-L12-v2 | 118M | 384 | ~470 MB | claramente inferior en retrieval asimétrico | Descartado: entrenado para similitud simétrica, límite 128 tokens |
| snowflake-arctic-embed 2.0 (m) | 305M | 768 | ~1.2 GB | fuerte fuera de MIRACL (CLEF) | Alternativa media, más pesada que e5-small |

Datos clave:
- BGE-M3 gana ~7–10 puntos nDCG sobre e5-small en MIRACL, pero con 5× parámetros y 2.7× dimensionalidad (más RAM también en el índice) ([informe técnico mE5](https://arxiv.org/pdf/2402.05672), [BGE-M3 paper](https://arxiv.org/pdf/2402.03216), [discusión HF BGE-M3](https://huggingface.co/BAAI/bge-m3/discussions/23)).
- **Cross-lingual real (consulta ES → doc EN):** BGE-M3 fue entrenado con pares cross-lingual (MKQA 75.1 R@100); mE5 se entrenó mayormente monolingüe, lo que penaliza algo su CLIR — mitigable con retrieval híbrido + expansión bilingüe (sección 6). El paper de [Arctic-Embed 2.0](https://arxiv.org/pdf/2412.04506) advierte que los scores MIRACL (dominio Wikipedia) pueden sobreestimar el rendimiento en corpus clínicos: valida con un mini test set propio.
- **Cuantización int8 ONNX**: ~2.7–3.4× más rápido en CPU conservando 94–98% de calidad; multilingual-e5-small ONNX rinde ~920 tokens/s vs 470 tokens/s PyTorch en CPU; existen variantes listas: [deepfile/multilingual-e5-small-onnx-qint8](https://huggingface.co/deepfile/multilingual-e5-small-onnx-qint8) (112.8 MB) y [Xenova/multilingual-e5-small](https://huggingface.co/Xenova/multilingual-e5-small/tree/main/onnx) ([benchmark ONNX](https://github.com/tkys/multilingual-e5_onnx), [Vespa: tradeoffs cuantizados](https://blog.vespa.ai/embedding-tradeoffs-quantified/), [docs sentence-transformers: 3.08× speedup](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)).
- **Crítico con e5:** usar prefijos `query: ` / `passage: ` — omitirlos degrada mucho la calidad ([Pinecone guía E5](https://www.pinecone.io/learn/the-practitioners-guide-to-e5/), [HF e5-small](https://huggingface.co/intfloat/multilingual-e5-small)).
- "jina-embeddings-v2-small-es" **no existe** como tal; existe `jina-embeddings-v2-base-es` (bilingüe ES/EN, 161M) — opción decente pero sin ventaja clara sobre e5-small int8 en este presupuesto.

**Recomendación:** `multilingual-e5-small` int8 ONNX (onnxruntime u OpenVINO): ~150–300 MB RAM en runtime, embeddings de consulta en pocos ms, y vectores de 384 dims → índice de 20k chunks ≈ 30 MB fp32 (15 MB int8).

## 2. Retrieval híbrido: BM25 + denso + RRF (+ ¿reranker?)

- **BM25S** es la elección clara sobre rank-bm25: órdenes de magnitud más rápido (scoring sparse eager con NumPy/Scipy), comparable o superior a Elasticsearch en QPS single-thread, sin servidor ni Java ([bm25s GitHub](https://github.com/xhluca/bm25s), [paper](https://arxiv.org/pdf/2407.03618), [blog HF](https://huggingface.co/blog/xhluca/bm25s)). Para 20k chunks la consulta es sub-milisegundo. Tantivy es excelente pero añade complejidad innecesaria a esta escala. Limitación: bm25s no soporta borrado incremental → **reconstruir el índice BM25 al añadir/eliminar un doc** (segundos con este corpus; ver sección 7).
- **Fusión RRF**: no requiere normalizar scores de escalas distintas, ~7 líneas de Python, k=60; mejora nDCG sobre BM25 solo ([explicación RRF](https://blog.serghei.pl/posts/reciprocal-rank-fusion-explained/), [guía híbrida 2026](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)). El híbrido es especialmente valioso aquí: BM25 captura términos exactos (fármacos, dosis, "Tokyo Guidelines") y el denso captura semántica cross-lingual.
- **Reranker en CPU — sí, pero uno pequeño y sobre pocos candidatos:**
  - bge-reranker-v2-m3: descartado en CPU (~10.6× más lento que ms-marco-MiniLM; pensado para GPU) ([benchmark rerankers](https://aimultiple.com/rerankers)).
  - ms-marco-MiniLM-L6-v2: ~0.14 s/query en CPU con ~20 candidatos, pero **solo inglés**.
  - jina-reranker-v1-tiny: 0.13 GB, ~0.14 s/query, pero **solo inglés** y deprecado ([HF](https://huggingface.co/jinaai/jina-reranker-v1-tiny-en)); jina-reranker-v2-base-multilingual es 278M y licencia CC-BY-NC (no comercial) ([Jina](https://jina.ai/models/jina-reranker-v2-base-multilingual/)).
  - **Opción multilingüe viable:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (mMARCO, ~117M, Apache-2.0; catálogo en [sbert.net](https://sbert.net/docs/cross_encoder/pretrained_models.html)) cuantizado int8 → reranquear solo el **top-10/15** de la fusión: ~0.3–0.8 s en CPU. En un agente de voz eso es tolerable si se solapa con el streaming del TTS; si no, hazlo **opcional/configurable**.

**Recomendación:** BM25S + denso (top-30 cada uno) → RRF → top-10 → reranker mMiniLMv2 int8 opcional → top-4/5 al LLM.

## 3. Técnicas anti-limitaciones del RAG clásico (baratas en CPU)

Todas las siguientes operan en **indexación o lookup**, sin llamadas extra al LLM:

1. **Chunking por estructura del documento** (la técnica de mayor ROI aquí): las guías clínicas tienen estructura fuerte (Recomendaciones, Diagnóstico, Tratamiento, Grados de evidencia). Segmenta por títulos/secciones (TOC de PyMuPDF + heurística de tamaño de fuente) en vez de por ventana fija ([estrategias de chunking 2026](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)).
2. **Contextual chunk headers**: antepón a cada chunk `[Doc: Guía WSES apendicitis 2020 | Sección: Tratamiento antibiótico | Escenario: apendicitis]` antes de embedder. Da contexto global sin LLM; cuidado con que el header no domine al contenido ([Unstructured: contextual chunking](https://unstructured.io/blog/contextual-chunking-in-unstructured-platform-boost-your-rag-retrieval-accuracy)). (La "contextual retrieval" de Anthropic usa LLM en ingesta — demasiado cara con tu LLM de 2.5 GB.)
3. **Parent-document / small-to-big**: indexa chunks hijos pequeños (~200–300 tokens) y entrega al LLM el chunk padre (sección completa, ~800–1200 tokens). Ganancias reportadas de 15–25% en precisión de respuesta con overhead mínimo; solo requiere un mapa `child_id → parent_id` ([Small-to-Big, Sophia Yang](https://medium.com/data-science/advanced-rag-01-small-to-big-retrieval-172181b396d4), [ZeroEntropy PDR](https://zeroentropy.dev/concepts/parent-document-retrieval/), [LanceDB blog](https://www.lancedb.com/blog/modified-rag-parent-document-bigger-chunk-retriever-62b3d1e79bc6)). Sentence-window es la variante más fina, pero puede empeorar en texto repetitivo ([ARAGOG](https://arxiv.org/pdf/2404.01037)) — con secciones clínicas, parent-document es más robusto.
4. **Metadata filtering por escenario quirúrgico**: clasifica cada documento en ingesta (por keywords en título/primeras páginas: apendicitis, mama, colecistitis, colorrectal, articular) y filtra pre-búsqueda cuando la conversación tiene escenario activo. Reduce el espacio de búsqueda ~5× y elimina confusiones entre guías (p. ej. antibióticos de colecistitis vs apendicitis).
5. **Query expansion barata (sin LLM)**: diccionario estático ES↔EN de ~100–200 términos médicos del dominio (colecistitis→cholecystitis, reemplazo articular→joint replacement/arthroplasty, apendicectomía→appendectomy) aplicado a la rama BM25. Coste cero, ataca directamente el gap cross-lingual léxico.
6. **HyDE: descartado en runtime.** Requiere una llamada LLM extra por consulta; con un LLM ~2.5 GB en CPU son varios segundos añadidos y +43–60% de latencia con más alucinación en modelos pequeños ([HyDE paper](https://arxiv.org/pdf/2212.10496), [comparativa HyDE vs RAG](https://beyondscale.tech/blog/hyde-vs-rag-retrieval-augmented-generation)). Si quieres el efecto, usa **HyPE** (preguntas hipotéticas generadas offline en la ingesta y embebidas junto al chunk) — mueve el coste fuera del camino de la voz, aunque encarece la ingesta "en vivo"; yo lo omitiría en v1.

## 4. Vector store embebido con DELETE real y filtros

| Opción | Delete en caliente | Filtros metadata | RAM/despliegue | Notas |
|---|---|---|---|---|
| **sqlite-vec** | Sí, transaccional (ACID, journaled) | Sí (columnas de metadata + auxiliares en vec0, filtrado nativo) | Mínima: extensión C sin dependencias, un solo fichero | Pre-v1 (bug de DELETE corregido en [v0.1.9](https://github.com/asg017/sqlite-vec/releases/tag/v0.1.9)); búsqueda exacta brute-force — **perfecta a 20k vectores** |
| **LanceDB** | Sí (`delete(where)`) | Sí (SQL-like) | In-process, formato columnar en disco; ~559 MB–1.6 GB según carga | Buen segundo candidato; versionado de datos incluido |
| **ChromaDB** | Sí (`delete(ids/where)`) pero **soft-delete** en HNSW → fragmentación, requiere `hnsw rebuild` como mantenimiento ([Cookbook](https://cookbook.chromadb.dev/running/maintenance/)); bug histórico de carpetas huérfanas ([#1309](https://github.com/chroma-core/chroma/issues/1309)) | Sí | El más pesado en RAM a escala | API cómoda; aceptable a tu escala pero menos "olvido garantizado" |
| Qdrant embedded | — | Excelentes | **No es realmente embebido**: binario servidor ~900 MB; Qdrant Edge aún en beta privada | Descartado |
| FAISS | Problemático (IDMap/tombstones, sin filtros nativos) | No | Ligero | Descartado por el requisito de conocimiento vivo |

Fuentes: [comparativa 2026](https://4xxi.com/articles/vector-database-comparison/), [Firecrawl vector DBs](https://www.firecrawl.dev/blog/best-vector-databases), [sqlite-vec metadata filtering](https://github.com/asg017/sqlite-vec/issues/26), [sqlite-vec v0.1.0](https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html).

**Recomendación:** **sqlite-vec**. A 10–25k vectores de 384 dims la búsqueda exacta tarda milisegundos, el DELETE es una transacción SQL real (`DELETE FROM chunks WHERE doc_id = ?` → olvido inmediato y verificable, sin tombstones ni rebuilds), y chunks + metadatos + vectores + manifiesto de documentos viven en **un único fichero .db** — trazabilidad y backup triviales. Alternativa igual de válida si prefieres API más rica: LanceDB.

## 5. Extracción de PDFs y OCR

- **PyMuPDF (fitz)**: el más rápido con diferencia (~0.01 s/página, 10–50× más rápido que alternativas), extrae TOC, tamaños de fuente y bloques → suficiente para chunking por secciones de guías clínicas. Licencia AGPL (ojo si el producto es cerrado; pdfplumber/pypdfium2 como alternativa permisiva) ([benchmark 200 PDFs](https://pdfmux.com/blog/pdfmux-vs-pymupdf-vs-marker-vs-docling/), [comparativa 2026](https://link.sc/blog/best-pdf-parsers-2026)).
- **Docling**: la mejor estructura (97.9% en tablas complejas) pero **pesado para tu caso**: +500 MB de instalación, carga de modelos 30–60 s y marcadamente más lento en CPU-only ([benchmark Procycons](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)). Incompatible con ingesta "en vivo" rápida en 8 GB compartidos. Unstructured hi-res: mismo problema.
- **OCR para el PDF escaneado**: **OCRmyPDF con `-l spa+eng`** (wrapper de Tesseract con deskew/limpieza y capa de texto; ~45 s/24 páginas en CPU) ([GitHub](https://github.com/ocrmypdf/OCRmyPDF)); alternativa moderna y mínima: **RapidOCR** (modelos PaddleOCR en ONNX, el menor footprint CPU) — pero valida su precisión en español; Tesseract tiene cobertura de idiomas más probada ([comparativa OCR](https://modal.com/blog/8-top-open-source-ocr-models-compared)).

**Recomendación:** PyMuPDF como parser único + detección "¿tiene capa de texto?" (`page.get_text()` vacío) → si no, ruta OCRmyPDF/tesseract spa+eng automática.

## 6. Manejo bilingüe: ¿traducir en ingesta o cross-lingual?

**Opción A — Traducir EN→ES en ingesta (Argos Translate / opus-mt vía CTranslate2):**
- Pros: todo el índice queda en español (BM25 funciona directo), el LLM nunca lee inglés.
- Contras: modelos opus-mt en-es ≈ 300 MB disco, RAM adicional en ingesta; velocidad limitada en CPU (referencia LibreTranslate: ~20–25 frases/min saturando 2 cores) → traducir un PDF de 100 páginas subido "en vivo" tardaría **decenas de minutos**, rompiendo el requisito de conocimiento vivo; y sobre todo **riesgo de error de traducción en terminología médica** (dosis, contraindicaciones) sin supervisión — inaceptable en trazabilidad clínica: la cita ya no sería el texto original del documento ([Argos GitHub](https://github.com/argosopentech/argos-translate/), [comparativa offline MT](https://skeptric.com/python-offline-translation/), [OPUS-MT](https://aiwiki.ai/wiki/opus-mt)).

**Opción B — Embeddings cross-lingual + LLM lee EN y responde ES (recomendada):**
- Pros: ingesta instantánea, cero RAM extra, la cita es literal del documento original (trazabilidad intacta), e5-small ya alinea ES/EN en el mismo espacio; los LLM pequeños actuales leen inglés y responden en español con fiabilidad razonable.
- Contras: gap cross-lingual de mE5 (mitigado con el híbrido: expansión bilingüe de términos en BM25 + reranker multilingüe); hay que forzar en el system prompt "responde SIEMPRE en español".
- Híbrido pragmático opcional: traducir **solo los títulos de sección y de documento** (diccionario o Argos, es poco texto) para los contextual headers, dejando el cuerpo original.

## 7. Arquitectura final recomendada

### Presupuesto RAM del subsistema RAG (~600–800 MB pico)
- e5-small int8 ONNX: ~150–300 MB · sqlite-vec + SQLite: ~50–100 MB · índice BM25S en memoria: ~30–80 MB · reranker int8 (opcional, cargable bajo demanda): ~150 MB · PyMuPDF/proceso: ~100 MB. OCR/tesseract solo se lanza como subproceso puntual.

### Pipeline de ingesta (por documento, en vivo)
1. `doc_id = hash SHA-256 del fichero`; registrar en tabla `documents` (manifiesto: título, fichero, idioma detectado, escenario, fecha).
2. PyMuPDF → texto por página + TOC/fuentes → **secciones**. Sin capa de texto → OCRmyPDF (spa+eng) → reintentar.
3. Clasificar escenario quirúrgico (keywords) e idioma (langdetect/lingua) → metadatos.
4. Chunking jerárquico: **padres** = secciones (~800–1200 tokens), **hijos** = ~250 tokens con solape 15%. A cada hijo, header contextual `[doc | sección | escenario]`.
5. Embeddings de hijos (`passage: ...`) con e5-small int8 → `INSERT` en sqlite-vec con metadatos (doc_id, parent_id, página, sección, escenario, idioma) **en una transacción**.
6. Reconstruir índice BM25S sobre todos los chunks activos (segundos a esta escala) y hot-swap atómico en memoria.

### Consulta (turno de voz)
1. Transcripción → normalización + expansión bilingüe por diccionario ES↔EN.
2. Filtro de metadatos por escenario activo (si se conoce) → BM25S top-30 + denso (`query: ...`) top-30 → **RRF** → top-10 → (opcional) reranker mMiniLMv2 int8 → top-4.
3. **Small-to-big**: sustituir hijos por sus secciones padre (dedupe por parent_id).
4. Prompt al LLM con bloques etiquetados `[FUENTE n: título, sección, página]` + instrucción de citar `[n]` y responder en español; la app resuelve `[n]` → documento/página para la **cita verbal y visual** (trazabilidad literal, sin traducción intermedia).

### Conocimiento vivo (add/delete robusto)
- **Alta:** pipeline de ingesta completo; un PDF típico de guía tarda segundos–pocos minutos (OCR aparte); disponible en la siguiente consulta.
- **Baja:** transacción única: `DELETE FROM chunks WHERE doc_id=?; DELETE FROM documents WHERE doc_id=?;` + rebuild/hot-swap del BM25 → **olvido completo e inmediato, verificable con un `SELECT`**, sin tombstones HNSW ni índices desincronizados (la razón principal para elegir sqlite-vec sobre Chroma/FAISS).
- Un watcher de carpeta o endpoint de upload dispara ambos flujos; el manifiesto `documents` es la fuente de verdad para la UI ("documentos que el agente conoce ahora").

### Por qué esto es "mejor que RAG convencional"
Híbrido léxico+denso con RRF (robustez cross-lingual y a terminología exacta), chunking estructural con headers contextuales (chunks autoexplicativos), small-to-big (precisión de retrieval + contexto completo de la sección para el LLM), filtrado por escenario (precisión clínica), citas a nivel de documento/sección/página del texto original, y borrado transaccional real — todo sin ninguna llamada extra al LLM en el camino crítico de la voz.

**Fuentes principales:** [mE5 report](https://arxiv.org/pdf/2402.05672) · [BGE-M3](https://arxiv.org/pdf/2402.03216) · [Arctic-Embed 2.0](https://arxiv.org/pdf/2412.04506) · [Vespa quantization](https://blog.vespa.ai/embedding-tradeoffs-quantified/) · [sbert efficiency](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) · [BM25S](https://github.com/xhluca/bm25s) / [paper](https://arxiv.org/pdf/2407.03618) · [RRF](https://blog.serghei.pl/posts/reciprocal-rank-fusion-explained/) · [rerankers benchmark](https://aimultiple.com/rerankers) · [sbert cross-encoders](https://sbert.net/docs/cross_encoder/pretrained_models.html) · [Small-to-Big](https://medium.com/data-science/advanced-rag-01-small-to-big-retrieval-172181b396d4) · [ARAGOG](https://arxiv.org/pdf/2404.01037) · [HyDE](https://arxiv.org/pdf/2212.10496) · [sqlite-vec](https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html) / [v0.1.9](https://github.com/asg017/sqlite-vec/releases/tag/v0.1.9) / [metadata](https://github.com/asg017/sqlite-vec/issues/26) · [Chroma maintenance](https://cookbook.chromadb.dev/running/maintenance/) · [vector DB comparison](https://4xxi.com/articles/vector-database-comparison/) · [PDF parsers 2026](https://link.sc/blog/best-pdf-parsers-2026) / [200-PDF benchmark](https://pdfmux.com/blog/pdfmux-vs-pymupdf-vs-marker-vs-docling/) / [Procycons](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/) · [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) · [OCR comparison](https://modal.com/blog/8-top-open-source-ocr-models-compared) · [Argos Translate](https://github.com/argosopentech/argos-translate/) / [offline MT](https://skeptric.com/python-offline-translation/) · [Pinecone E5](https://www.pinecone.io/learn/the-practitioners-guide-to-e5/) · [Chroma delete guide](https://docs.trychroma.com/guides)
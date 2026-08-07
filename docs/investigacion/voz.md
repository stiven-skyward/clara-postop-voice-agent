He completado la investigación. Aquí está el informe.

---

# Pipeline de voz en español 100% CPU para agente médico conversacional (8 GB RAM)

## 1. STT — Voz a texto en español

### 1.1 whisper.cpp (recomendado como principal)
- **Precisión en español** (Common Voice es, aprox.): tiny ~16.6% WER, base ~11.9%, small ~7.3%, medium ~4.9%. El español es lengua "Tier 1" de Whisper (3–6% WER en audio limpio, casi paridad con inglés).
- **Cuantización**: q4_0 es el "campeón indiscutible" en CPU (más rápido que q8_0 con calidad idéntica, por menor ancho de banda de memoria); q5_0/q5_1 son el punto dulce conservador (~40% menos tamaño que F16, pérdida imperceptible). Los cuantizados recortan RAM/latencia 30–60%.
- **RAM**: tiny ~75 MB de pesos (runtime ~300 MB); base-q5 ~60 MB pesos / ~500 MB runtime; small-q5_0 ~500 MB de archivo, **~1–1.5 GB en ejecución** (incluye KV cache y buffers de espectrograma). Medición comparativa: whisper.cpp ~1049 MB vs faster-whisper 1477–2257 MB con modelos equivalentes.
- **Velocidad/latencia CPU**: small ≈ tiempo real–2× en CPU moderno de escritorio (clip de 10 s → ~4–8 s); base ≈ 2–5× tiempo real (clip de 10 s → ~2–4 s); en Raspberry Pi 5, base logra tiempo real con 4 hilos y small queda a 0.4–0.6×. Modo `--stream` con 0.5–2 s de retraso.
- Fuentes: [Benchmark i5-460M (discusión oficial)](https://github.com/ggml-org/whisper.cpp/discussions/3752), [repo whisper.cpp](https://github.com/ggml-org/whisper.cpp), [WER por idioma](https://novascribe.ai/how-accurate-is-whisper), [tamaños de modelo](https://openwhispr.com/blog/whisper-model-sizes-explained), [cuantización Whisper-small (arXiv)](https://arxiv.org/html/2511.08093).

### 1.2 faster-whisper (CTranslate2 int8)
- En CPU x86 mono-stream, velocidad **similar** a whisper.cpp con int8, pero **usa notablemente más RAM** (1.5–2.3 GB vs ~1 GB) y más latencia de arranque. Gana solo con batching o GPU. Ventaja práctica: API Python cómoda (word timestamps, VAD integrado). Evitar int4 (alucinaciones en clips largos).
- Fuentes: [comparativa 2026](https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026), [issue #1127 whisper.cpp](https://github.com/ggml-org/whisper.cpp/issues/1127), [codersera](https://codersera.com/blog/faster-whisper-vs-whisper-cpp-speech-to-text-2026/).
- **Veredicto**: en un dispositivo de 8 GB compartido, whisper.cpp gana por RAM.

### 1.3 Vosk español
- `vosk-model-small-es-0.42`: **39 MB**, WER **16.0 (Common Voice) / 11.2 (MLS)**, ~**300 MB RAM** runtime, Apache 2.0. Modelo grande `vosk-model-es-0.42`: 1.4 GB, WER 7.5 CV / 5.8 MLS.
- Fortaleza: **streaming real** con parciales en <200–500 ms, vocabulario reconfigurable en runtime (útil para inyectar términos médicos). Debilidad: arquitectura Kaldi antigua; su WER duplica al de whisper-small con acentos y jerga.
- Fuentes: [modelos oficiales Vosk](https://alphacephei.com/vosk/models), [Vosk vs Whisper 2026](https://www.sinologic.net/en/2026-05/vosk-vs-whisper-local-the-ultimate-2026-guide-to-self-hosted-speech-recognition-stt.html).

### 1.4 sherpa-onnx / moonshine / distil-whisper
- **sherpa-onnx**: NO existe zipformer/paraformer en español (catálogo centrado en chino/inglés); sí puede ejecutar modelos Whisper en ONNX. [Catálogo](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html).
- **Moonshine v2**: sí tiene modelo español dedicado (tamaño Base, muy rápido), pero **licencia no comercial** para modelos no ingleses — descartado para producto médico sin licencia. [moonshine-v2](https://github.com/moonshine-ai/moonshine-v2).
- **distil-whisper**: **solo inglés** en los checkpoints oficiales. [Confirmación](https://huggingface.co/distil-whisper/distil-large-v3/discussions/2).

### 1.5 Acento colombiano y jerga
- Estudio MDPI sobre Common Voice español documenta **sesgo por acento** en Whisper (WER mayor en ciertos acentos latinoamericanos): [MDPI 2024](https://www.mdpi.com/2076-3417/14/11/4734). Benchmark de consultas médicas en español LatAm (medRxiv, 10 modelos): [preprint](https://www.medrxiv.org/content/10.64898/2026.07.14.26358062v2) — ningún modelo open-source está optimizado para habla médica LatAm; el fine-tuning de Whisper mejora resultados.
- Mitigación práctica sin GPU: usar `initial_prompt` de whisper.cpp con vocabulario médico/regional, y post-corrección con el LLM local ya presente (patrón validado en [npj Digital Medicine](https://www.nature.com/articles/s41746-026-02490-z)).

## 2. VAD — Silero VAD (única opción seria en CPU)
- Modelo ONNX de **~2 MB**, procesa chunks de 30–32 ms en **<1 ms por chunk en 1 hilo de CPU** (RTF ~0.004). RAM total <50 MB con onnxruntime ya cargado.
- Caveat: la detección de **fin de habla** añade unos cientos de ms por diseño (min_silence_duration ~300–500 ms recomendado para no cortar pausas del paciente).
- Fuentes: [repo silero-vad](https://github.com/snakers4/silero-vad), [guía ONNX Runtime](https://dev.to/kiarina/extracting-speech-segments-with-silero-vad-and-onnx-runtime-3h8a), [comparativa VADs](https://picovoice.ai/blog/best-voice-activity-detection-vad/).

## 3. TTS — Kokoro-82M vs Piper

### 3.1 Kokoro-82M (español: ef_dora, em_alex, em_santa)
- **Calidad**: la mejor naturalidad en su clase de tamaño (StyleTTS2 + ISTFTNet, Apache 2.0). ef_dora es sólida y natural; punto clave para tu caso: las 3 voces españolas tienen **pronunciación latinoamericana** ([issue #246](https://github.com/hexgrad/kokoro/issues/246)) — ideal para pacientes colombianos. Contras: el español es menos pulido que el inglés (menos horas de entrenamiento), G2P vía espeak-ng, conviene pre-normalizar números/abreviaturas médicas.
- **Velocidad CPU**: **3–11× tiempo real** (RTF ~0.1–0.35). Benchmark en EPYC 4 cores: todo más rápido que tiempo real; ONNX ≈ PyTorch (ONNX gana en textos medios/largos). [Benchmark](https://gist.github.com/efemaer/23d9a3b949b751dde315192b4dcf0653), [kokoro-onnx](https://www.ttsinsider.com/kokoro-82m-onnx/).
- **RAM real**: PyTorch ~1.5–2 GB; **ONNX ~780–840 MB medidos** (fp32 310 MB / int8 103 MB en disco; int8 apenas baja RAM y puede ser MÁS LENTO en algunos CPUs — [medición sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/issues/2374)). Usar `kokoro-onnx` (thewh1teagle) + `voices-v1.0.bin` multilingüe (el pack ONNX de onnx-community trae solo voces inglesas).
- **Latencia de primer audio**: ~0.5–1.5 s por oración corta en CPU moderno. **Streaming por frases: sí** (el modelo no hace streaming intra-frase, pero sintetizar oración por oración funciona bien).

### 3.2 Piper
- **Voces español**: es_ES: **davefx (medium)**, **sharvard (medium)**, carlfm (x_low), mls_9972/mls_10246 (low); es_MX: **ale (medium)**, **claude (high)**. Lista completa: [VOICES.md](https://github.com/rhasspy/piper/blob/master/VOICES.md), [HF rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/es/es_ES), [muestras de audio](https://k2-fsa.github.io/sherpa/onnx/tts/all/Spanish/index.html).
- **Rendimiento**: RTF **0.008–0.03** en CPU de escritorio (10 s de audio en ~80–300 ms), **primer audio en ~40 ms**, **<100 MB RAM**, tiempo real incluso en Raspberry Pi 4. MIT.
- **Calidad**: claramente inferior a Kokoro en naturalidad/prosodia ("voz de asistente", entonación plana). Para contexto médico empático, davefx/sharvard-medium son inteligibles pero menos cálidas; es_MX-claude-high es la mejor pero más lenta.
- Fuentes: [comparativa Kokoro/Piper/XTTS](https://contracollective.com/blog/kokoro-vs-piper-vs-xtts-local-text-to-speech-m5-max-2026), [benchmark CPU KittenTTS/Piper/Kokoro](https://github.com/KittenML/KittenTTS/issues/40), [12 TTS comparados](https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-part-2).

### 3.3 Veredicto TTS
Regla consolidada en la comunidad: **Kokoro como principal** (naturalidad, empatía, acento latino) y **Piper como fallback** cuando la latencia o la RAM aprieten. Ambos con licencia comercial (Apache 2.0 / MIT).

## 4. Recomendación final de pipeline

```
Navegador (getUserMedia, PCM 16 kHz via WebSocket)
  → Silero VAD ONNX (chunks 32 ms; fin de habla: 400-500 ms de silencio)
  → whisper.cpp small-q5_0, idioma=es, initial_prompt con léxico médico/colombiano
  → LLM local (post-corrige jerga/términos si hace falta)
  → segmentación por oraciones del stream del LLM (corte en . ? ! :)
  → Kokoro-82M ONNX (ef_dora) por oración → chunks WAV/Opus al navegador (MediaSource/Audio streaming)
```

**Presupuesto de RAM (8 GB):**

| Componente | RAM |
|---|---|
| SO + navegador servidor base | ~1.5–2.0 GB |
| LLM (~2–2.5 GB) | 2.5 GB |
| Embeddings | ~0.5 GB |
| whisper.cpp small-q5_0 | ~1.0–1.3 GB |
| Silero VAD + onnxruntime | ~0.05 GB |
| Kokoro-82M ONNX fp32 | ~0.8 GB |
| **Total** | **~6.4–7.2 GB** ✔ cabe justo |

**Plan B si el margen es insuficiente o el CPU es débil** (libera ~1.7 GB): whisper.cpp **base-q5_0** (~500 MB, WER es ~12%) + **Piper es_MX-claude-high o es_ES-davefx-medium** (~100 MB). Latencia total percibida cae a <1.5 s.

**Latencia percibida estimada (combo principal, CPU x86 moderno, respuesta de paciente de 8 s):** fin de habla detectado +0.4 s → STT ~2–4 s → primer token LLM ~0.5–1 s → primera oración TTS ~0.7 s ⇒ **~4–6 s hasta primer audio**; con base-q5 baja a ~2.5–3.5 s. Trucos clave: transcribir en cuanto el VAD cierre (no esperar buffer fijo), lanzar el TTS de la primera oración mientras el LLM sigue generando, y enviar audio en chunks para que el navegador empiece a reproducir de inmediato.

**Notas de investigación relevantes**: no existe zipformer/paraformer español en sherpa-onnx; distil-whisper es solo inglés; Moonshine español tiene licencia no comercial; el int8 de Kokoro ahorra disco pero no RAM y puede ser más lento; y las voces españolas de Kokoro son de pronunciación latinoamericana (ventaja para Colombia, no defecto).
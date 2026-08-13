# Diagramas — arquitectura y flujo de decisión

> Entregable 02. Cada elemento corresponde a código real del repositorio
> (módulo indicado entre paréntesis).

## Versión en imagen (oficial)

![Arquitectura de la solución](diagramas/arquitectura.svg)

![Flujo de decisión del agente por turno](diagramas/flujo-decision.svg)

Fuentes editables: [`diagramas/arquitectura.svg`](diagramas/arquitectura.svg) y
[`diagramas/flujo-decision.svg`](diagramas/flujo-decision.svg). Cada imagen incluye
su recuadro de convenciones con la función de cada componente y sus conexiones.

---

## Versión mermaid (referencia rápida en texto)

### Arquitectura de la solución

```mermaid
flowchart LR
  subgraph NAV[Navegador del paciente]
    MIC[Micrófono<br/>AudioWorklet 16 kHz] --> WSC[WebSocket /ws/call]
    WSC --> SPK[Reproducción<br/>por oraciones]
  end
  subgraph SRV[Servidor local — 100% CPU]
    WSC --> VAD["Silero VAD ONNX (app/vad.py)<br/>fin de habla: 800 ms de silencio"]
    VAD --> STT["whisper.cpp small-q5 es (app/stt.py)<br/>initial_prompt léxico médico"]
    STT --> ORQ["Orquestador (app/agent/orchestrator.py)"]
    ORQ --> LLM["llama-server · Llama 3.2 3B Q4_K_M<br/>(app/llm.py) chat streaming + JSON-schema"]
    ORQ --> RAGQ["Búsqueda híbrida (app/rag/search.py)<br/>BM25S + e5-small + RRF + small-to-big"]
    RAGQ --> KB[("knowledge.db<br/>sqlite-vec (app/rag/store.py)")]
    ORQ --> TTSX["TTS Kokoro-82M / Piper (app/tts.py)<br/>síntesis por oraciones"] --> WSC
    ORQ --> LOGS[["data/logs/*.jsonl (app/metrics.py)<br/>turnos · llamadas · alertas"]]
  end
  subgraph ADM[Consola de administración /admin]
    UP[Subir documento] --> ING["Ingesta (app/rag/ingest.py)<br/>PyMuPDF → secciones → chunks → OCR si escaneado"]
    ING --> KB
    DEL[Eliminar documento] --> DELT["DELETE transaccional + rebuild BM25"] --> KB
    LIST[Listar + estado 'procesado y disponible'] --> KB
  end
```

### Flujo de decisión del agente (por turno de paciente)

```mermaid
flowchart TD
  A[Transcripción del turno] --> QR{"Atajo léxico rojo<br/>(quick_red_scan, con guardia de negación)"}
  QR -->|"pus / mal olor / líquido amarillento<br/>herida abierta / sangrado abundante…"| R
  QR -->|sin señal instantánea| B["Extracción estructurada a JSON<br/>(SCHEMA_EXTRACCION, GBNF: inválido imposible)"]
  B --> SAN["Anclaje léxico determinista<br/>(sanitize_extraction): un síntoma solo se acepta<br/>si el texto lo respalda; secreción amarillenta<br/>= purulenta aunque el paciente la minimice"]
  SAN --> C["Merge al estado de la llamada<br/>(SymptomState.merge — el cuadro solo se completa)"]
  C --> D{"Motor de reglas deterministas<br/>(triage.evaluate)<br/>+ detector de incongruencia: ≥2 dominios<br/>blandos degradados → amarillo; ≥3 → rojo<br/>(calibrado a 0 falsos negativos en el dataset)"}
  D -->|"fiebre ≥38.5 · dolor ≥8 · secreción purulenta<br/>herida abierta · sangrado · disnea · red flags texto"| R[ROJO]
  D -->|"fiebre 38-38.4 · dolor 5-7 · enrojecimiento<br/>síntomas a vigilar"| Y[AMARILLO]
  D -->|sin signos de alarma| G[VERDE]
  R --> R1["Alerta persistente (alertas.jsonl):<br/>paciente, síntomas, razones, transcripción"]
  R1 --> R2["Protocolo al paciente: qué va a pasar<br/>+ instrucción de urgencias si empeora"]
  Y --> Y1{¿Datos ambiguos?<br/>fiebre sin termómetro, dolor sin número}
  Y1 -->|sí| Y2[INDAGAR antes de decidir<br/>pregunta dirigida]
  Y1 -->|no| Y3[Comunicar vigilancia + aviso a enfermería hoy]
  G --> Q{¿El paciente preguntó algo?<br/>campo 'pregunta' de la extracción}
  Y3 --> Q
  Q -->|sí| S["RAG híbrido con filtro de escenario<br/>→ FUENTES [n] al prompt"]
  S --> S1{¿Hay fuentes?}
  S1 -->|sí| S2["Respuesta fundamentada citando [n]<br/>(cita registrada en el resumen)"]
  S1 -->|no| S3["Límite honesto: 'no tengo esa información,<br/>la remito al equipo de salud'"]
  Q -->|no| W[Siguiente dominio del guion:<br/>dolor→fiebre→herida→movilidad→apetito/sueño]
  W --> Z{¿Guion completo?}
  Z -->|sí| F["Cierre: próximos pasos +<br/>resumen estructurado (SCHEMA_RESUMEN)<br/>persistido en llamadas.jsonl"]
  R2 --> F
```

**Invariantes de seguridad:** el nivel de triaje de la llamada solo sube
(`triage.combine`); el falso negativo pesa más que el falso positivo (reglas
conservadoras + combinación fiebre+herida→rojo); toda respuesta clínica lleva
fuente o declaración explícita de límite.

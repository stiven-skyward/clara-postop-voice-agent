# Reglas fijas del proyecto — Agente de voz para seguimiento postoperatorio

Fuente: `repokit/ParticipantArtifacts-main/` (README, `docs/rubrica-evaluacion.md`,
`docs/stack-tecnico.md`) + restricciones del propietario del proyecto.
**Estas reglas son innegociables. Toda decisión de diseño debe verificarse contra este documento.**

---

## R1 — Restricciones de hardware (del propietario)

- **Cero GPU.** Ni en desarrollo, ni en pruebas, ni en la solución final. Toda inferencia
  (LLM, STT, TTS, embeddings, OCR) corre en **CPU + RAM**.
- **Objetivo: dispositivo básico con 8 GB de RAM totales** (compartidos con el SO).
  Presupuesto de RAM de la solución completa: **≤ 5.5 GB** para dejar margen al SO y navegador.
- La solución debe ser ligera: modelos cuantizados, sin frameworks pesados innecesarios.

## R2 — Modelo de lenguaje (compuerta eliminatoria G3)

Solo se permite UNO de estos como modelo que razona:

| Modelo | Ejecución | ¿Compatible con R1? |
|---|---|---|
| Gemini 1.5 Flash | Nube (free tier, 15 RPM) | Sí (no usa RAM local) pero depende de internet |
| Llama 3.1 70B vía Groq | Nube (free tier) | Sí, pero depende de internet |
| **Llama 3.2 1B / 3B Instruct** | **Local, CPU** | **Sí — candidato principal** |
| **Phi-3.5 Mini (3.8B)** | **Local, CPU** | **Sí — candidato** |

- El informe final DEBE declarar cuál se usó y por qué.
- Se verifica contra dependencias, configuración y código. Usar otro modelo = descalificación.
- El resto del stack (orquestación, voz, RAG, embeddings) es **libre**.

## R3 — Compuertas eliminatorias (si falla una, no se evalúa)

1. **G1** — 4 entregables completos: repositorio público GitHub, diagrama de arquitectura
   y flujo de decisión, informe final, video (demo + 2 preguntas frente a cámara).
2. **G2** — Levantable en **≤ 15 minutos** siguiendo SOLO el README (credenciales y URLs incluidas).
3. **G3** — Modelo permitido (ver R2).
4. **G4** — Conversación de **voz en tiempo real** funciona: el jurado habla, el agente responde con voz.
5. **G5** — **Conocimiento vivo** desde la consola: subir un documento nuevo (que no está en el
   corpus) → el agente lo usa; eliminarlo → el agente lo olvida. Se prueba en vivo.

## R4 — Superficies obligatorias (contrato funcional, no estético)

- **Consola de administración**: subir documento · listar documentos · eliminar documento ·
  indicador visible de "procesado y disponible".
- **Interfaz de llamada**: iniciar llamada de voz desde el **navegador** · hablar por micrófono ·
  escuchar al agente. Sin telefonía real.

## R5 — Comportamiento clínico obligatorio

- Conversa en **español** con pacientes colombianos (regionalismos, descripciones ambiguas).
- **Cero tolerancia a alucinaciones**: respuestas clínicas fundamentadas en el corpus; si no
  sabe, lo declara y redirige. Nunca inventar dosis, medicamentos ni procedimientos.
- **Asimetría clínica**: el falso negativo (no alertar cuando había que alertar) es la falla
  catastrófica; pesa más que el falso positivo. Ante la duda, escalar.
- Ante ambigüedad: **indagar antes de decidir**.
- **Trazabilidad**: cada respuesta clínica registra qué documento la sustenta, verificable
  contra la fuente.
- Al alertar: registro estructurado y persistente + comunicar al paciente el siguiente paso.
- Al terminar cada llamada: **resumen estructurado** (paciente, procedimiento, síntomas
  reportados, decisión de triaje, referencias usadas, próximos pasos).
- Resistente a **prompt injection** (caer en una anula el apartado de voz), pacientes hostiles
  o asustados, interrupciones, audio degradado, peticiones fuera de misión.
- Respuestas **cortas** aptas para voz; tono empático y profesional; manejo de silencios.

## R6 — Métricas obligatorias en el README (verificadas contra logs en vivo)

- **Latencia** P50 y P95: desde que el paciente termina de hablar hasta que empieza a sonar
  el audio del agente.
- **Consumo**: tokens entrada/salida por turno y por llamada; invocaciones al modelo por
  turno; consultas RAG por llamada.
- **Costo estimado por llamada** (si es local: extrapolar a precios de API de producción y
  explicar el cálculo).
- Reportar números que no se sostienen contra los logs es peor que no reportarlos →
  la solución debe **loggear estas métricas automáticamente**.

## R7 — Datos del reto (fijos)

- `dataset_final.xlsx`: 3.991 turnos, 40 pacientes, 160 casos (días postop 1/3/7/14),
  capas `capa1_limpia` y `capa2_ruidosa` (mismo `caso_id`; sufijos `_c2`, `_c2_tercero`).
  `label_ground_truth`: verde/amarillo/rojo, constante por caso. Desbalance: 123 verde,
  25 amarillo, 12 rojo (a nivel de caso).
- `trayectorias_postop_silver.xlsx` (160): cuadro real por llamada — dolor_nrs, fiebre_c,
  movilidad, herida, apetito, sueño, arquetipo. Join: `caso_id = "caso_" + trayectoria_id`.
- `perfiles_clinicos_pacientes_silver_contest.xlsx` (40): procedimiento (Apendicectomía,
  Colecistectomía, Colectomía, Mastectomía, Reemplazo cadera/rodilla — 8 c/u), fecha cirugía,
  edad, género, comorbilidades (JSON en celda de texto).
- `perfiles_pacientes_co.xlsx` (40): demografía colombiana sintética (nombre, ciudad, EPS...).
- `textos/`: 107 PDFs ES/EN en 5 carpetas (dos con espacios en el nombre; hay duplicados;
  un PDF de Appendicitis/ escaneado **sin capa de texto** → requiere OCR).
  Ojo: la carpeta `breast_cancer/` mezcla documentos de cáncer de cuello uterino.
- **El material de evaluación incluirá conocimiento que el agente no ha visto** → el RAG
  vivo no es opcional, es el mecanismo central.

## R8 — Idioma del conocimiento (del propietario)

- Todo el sistema opera **en español** de cara al paciente.
- El corpus mezcla español e inglés → la solución debe resolver el cruce de idiomas
  (embeddings cross-lingual y/o traducción en ingesta), sin depender de servicios de pago.

## R9 — Entrega y proceso

- Repositorio público en GitHub con dependencias declaradas y fijadas, README reproducible,
  historia de commits real.
- El diagrama debe corresponder al código (el jurado toma elementos al azar y los busca).
- El demo del video debe corresponder al repositorio entregado.
- Desempate: núcleo funcional (RAG + decisión) → menor costo por llamada verificado.
- Fecha de entrega: **10 de agosto de 2026** (hoy es 7 de agosto).

## R10 — Voz (preferencia del propietario)

- TTS: **Kokoro-82M (español)** o **Piper (voces regionales)** — ambos CPU-friendly.
- STT y VAD: libres, pero CPU-only y ligeros (R1).

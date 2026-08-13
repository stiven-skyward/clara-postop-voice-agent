# Guion de grabación — video Tech Sphere Challenge 2026

Entregable 04: **demo con grabación de pantalla** + **dos preguntas frente a cámara**.
Repo que debe verse: `https://github.com/stiven-skyward/clara-postop-voice-agent`

Duración objetivo: **8–10 min de pantalla** + **2–3 min de cámara**.
Fecha de entrega: **hoy 12 ago 2026, medianoche**.

El video vale **15 pts** por sí mismo, y es la evidencia de **G4** (voz) y **G5**
(conocimiento vivo). El jurado también puntúa RAG (20) y triaje (20) con lo que
vea correr. Un demo que no coincida con el repo levanta bandera de integridad.

---

## Tarjeta de frases (tenla en papel o en otra pantalla)

Paciente en el desplegable: **Mauricio Juan González Sánchez — Apendicectomía**. Día **3**.

| Cuándo | Di exactamente esto, de un tirón, y espera |
|---|---|
| Tras el saludo (llamada verde) | El dolor está en dos y no he tenido fiebre. |
| Si pregunta la herida | La herida se ve normal, sin enrojecimiento ni secreción. |
| Pregunta RAG nueva | Una pregunta: ¿cuál es el código de alta temprana de esta clínica y qué significa? |
| Si no dice MANGO-47 | ¿Cuál es el código mango cuarenta y siete? |
| Tras borrar el doc | El dolor está en dos. Una pregunta: ¿cuál es el código de alta temprana mango cuarenta y siete? |
| Llamada roja, turno 1 | El dolor está como en ocho y va subiendo, y anoche me midieron treinta y ocho nueve de fiebre. |
| Llamada roja, herida | Sí señora, la herida me está soltando un liquidito amarillo que huele feo. |
| Si pide confirmación | Sí, quedó claro. |

---

## 0. Antes de grabar (5 minutos)

1. Abrir **solo** `http://localhost:8000/` y `http://localhost:8000/admin` (nunca una IP).
2. Permitir micrófono: candado de Chrome → Sitio → Micrófono → **Permitir**. Recargar.
3. Cerrar Discord, Spotify, Meet y otras pestañas con audio.
4. Tener a mano el archivo  
   `E:\source_meridian\postop-voice-agent\docs\demo\protocolo-alta-temprana-mango47.md`
5. En `/admin`, confirmar que **no** aparece «Protocolo institucional de alta temprana».
   Si está, elimínalo **antes** de pulsar grabar.
6. Hablar **frases completas**, despacio, y esperar a que Clara termine de hablar.
7. Ensayo mudo de 10 s: Iniciar llamada, di «hola», confirma que Clara responde con voz.
   Si responde, **cuelga** y recarga. Ese ensayo no va en el video.

### Cómo grabar en Windows

1. Abre Chrome a pantalla completa con dos pestañas: llamada (`/`) y consola (`/admin`).
2. **Win + G** (Xbox Game Bar) → **Win + Alt + R** para empezar y parar.
   Alternativa: Clipchamp o la Cámara de Windows para el tramo frente a cámara.
3. Graba **pantalla primero**, **cámara después**. Únelos en Clipchamp (gratis en Windows)
   en este orden: pantalla → tú hablando. Exporta MP4.
4. No hace falta mostrar el código. Sí conviene **5 segundos** del repo en GitHub
   al inicio, para que el demo coincida con lo entregado.

Si el servidor no responde: en WSL, desde el repo,

```bash
POSTOP_HOST=0.0.0.0 POSTOP_VENV=/home/forge/.venvs/postop bash scripts/run.sh
```

---

## 1. Demo en pantalla (qué mostrar y qué decir)

Habla tú en voz alta; el jurado evalúa **voz real**, no texto.

### Bloque A — Las dos superficies (≈40 s)

**Tú (narración):**  
«Esta es Clara, el agente de voz de seguimiento postoperatorio. El código está
en GitHub, público, bajo MIT. Corre 100 % local, en CPU, con Llama 3.2 3B.
Aquí está la interfaz de llamada y, en la otra pestaña, la consola de
conocimiento vivo.»

**Haz:** 5 s del repo `stiven-skyward/clara-postop-voice-agent` → `/` → `/admin`.
En admin, señalar la columna **Estado**: «procesado y disponible» y, abajo,
**Métricas del sistema** (P50/P95). No leas los números: basta con que se vean.

### Bloque B — Conocimiento vivo: alta (G5) (≈50 s)

**Tú:**  
«El reto pide que el conocimiento cambie en caliente. Voy a subir un protocolo
que el agente nunca ha visto: el código de alta temprana MANGO-47.»

**Haz:** en `/admin`, subir `protocolo-alta-temprana-mango47.md`.
Espera a que el estado pase a **✔ procesado y disponible**.

### Bloque C — Llamada verde + RAG del documento nuevo (≈2,5 min)

Paciente: **Mauricio Juan González Sánchez** · día postop **3**.

1. **Iniciar llamada.** Espera el saludo.
2. Cuando pregunte el dolor:  
   **«El dolor está en dos y no he tenido fiebre.»**
3. Si pregunta la herida:  
   **«La herida se ve normal, sin enrojecimiento ni secreción.»**
4. Pregunta de prueba (única, no está en el corpus original):  
   **«Una pregunta: ¿cuál es el código de alta temprana de esta clínica y qué significa?»**

**Éxito:** Clara debe mencionar **MANGO-47** (o alta en 72 horas).  
Si dice que no tiene la información, no borres el video: repite la pregunta
más despacio, con «código mango cuarenta y siete».

5. **Colgar.** Mostrar el resumen estructurado unos 5 segundos.

### Bloque D — Olvido (G5) (≈2 min)

**Tú:**  
«Ahora lo elimino. El agente debe olvidarlo de inmediato.»

**Haz:** en `/admin`, **eliminar** el documento MANGO-47. Confirmar el diálogo.

Nueva llamada, mismo paciente, día 3:

1. Espera el saludo.
2. **«El dolor está en dos. Una pregunta: ¿cuál es el código de alta temprana mango cuarenta y siete?»**

**Éxito:** Clara declara que **no tiene esa información** y ofrece remitir al equipo.
Eso es el olvido transaccional.

3. **Colgar.**

### Bloque E — Escalamiento rojo (20 pts de decisión) (≈2 min)

Nueva llamada, mismo Mauricio, día 3.

1. Espera el saludo.
2. **«El dolor está como en ocho y va subiendo, y anoche me midieron treinta y ocho nueve de fiebre.»**
3. Si pregunta la herida:  
   **«Sí señora, la herida me está soltando un liquidito amarillo que huele feo.»**
4. Si pide confirmación: **«Sí, quedó claro.»**

**Éxito:** pastilla **rojo · ALERTA**, protocolo de urgencias, y al colgar el
resumen con `alerta_generada`.

**Tú (cierre de pantalla):**  
«Eso es el núcleo del reto: voz en tiempo real, citas sobre conocimiento vivo,
y un triaje que no deja pasar un rojo.»

---

## 2. Preguntas frente a cámara (obligatorias)

Graba **mirando a cámara**, sin pantalla. Lee natural, no recites a toda velocidad.

### Pregunta 1 — Cliente / valor

**Enunciado oficial:**  
Si debes convencer a un cliente de que adopte el agente que construiste, ¿cómo
presentarías el problema que resuelve, por qué tu solución es la adecuada y qué
valor diferencial ofrece frente a otras alternativas?

**Guion (≈70 s):**

«El problema es que el seguimiento postoperatorio hoy depende de una enfermera
llamando uno a uno. No escala, se satura, y el paciente describe síntomas en
lenguaje cotidiano: “un liquidito amarillito”, “uno aguanta”. Si alguien no
alerta a tiempo, el costo clínico es altísimo.

Clara hace esa llamada en español, por voz, desde el navegador. Entiende al
paciente, se fundamenta en las guías de la clínica —no inventa dosis— y decide
cuándo escalar a un humano. Al colgar deja un resumen estructurado: síntomas,
triaje, fuentes y próximos pasos.

¿Por qué esta solución y no un chatbot de nube? Porque corre 100 % local, en
CPU, a costo cero de APIs, en un equipo de 8 GB. El conocimiento es vivo: si
cambia un protocolo, se sube por consola y Clara ya lo usa; si se retira, lo
olvida. Y el triaje no lo improvisa el modelo: lo deciden reglas clínicas
auditables, sesgadas a no dejar pasar un rojo.

El valor diferencial es ese trío: voz real, conocimiento que se actualiza en
caliente, y una decisión de alarma que se puede auditar línea a línea.»

### Pregunta 2 — Decisión técnica

**Enunciado oficial:**  
Elige la decisión técnica más relevante… alternativas, por qué las descartaste,
riesgos, y qué harías con dos semanas más.

**Guion (≈80 s):**

«La decisión más importante fue que **el LLM no decide el triaje**.

Probamos que Llama 3.2 3B, incluso con JSON forzado, clasificó como verde un
caso rojo claro: fiebre 38.9, dolor 8 en aumento y secreción purulenta. En
salud, ese falso negativo es la falla catastrófica.

Las alternativas: dejar el triaje al 3B con few-shot —lo descartamos porque
subestima—. Mandar el caso a un 70B en la nube —lo descartamos porque el
requisito era local, sin internet y sin cuota—. Entrenar un clasificador con
160 casos —pocos datos y poca explicabilidad clínica.

Lo que hicimos: el 3B **extrae** síntomas a JSON con gramática restringida, y
un motor de reglas determinista decide verde, amarillo o rojo. El nivel solo
puede subir. Ante ambigüedad, Clara indaga antes de decidir. Contra el dataset
del reto: cero falsos negativos y recall 12 de 12 en rojos, también en la capa
ruidosa.

Riesgos: reglas incompletas ante un síntoma nuevo —lo mitigamos con red flags
de texto— y errores de transcripción —lo mitigamos anclando cifras y
confirmaciones al habla real.

Con dos semanas más: validaría las reglas con un clínico, verificaría las citas
después de generar, y afinaria Whisper al acento colombiano.»

---

## 3. Si algo sale mal en la grabación

| Síntoma | Qué hacer (en cámara se ve bien) |
|---|---|
| No escucha | Candado → micrófono → Permitir. Recargar. Usar `localhost`. |
| «No alcancé a escucharle» | Repetir la **misma** frase, más cerca y completa. |
| No dice MANGO-47 | Confirmar en `/admin` que el doc está **disponible**. Preguntar «código mango cuarenta y siete». |
| Tras borrar, aún lo menciona | Colgar, esperar 3 s, **nueva** llamada. No reutilizar la anterior. |
| Se queda pegado al colgar | Recargar la página y empezar otra llamada. |

No narres bugs. Si un intento falla, di «voy a repetir el turno» y sigue.

---

## 4. Qué debe verse (mapeo a la rúbrica)

| Bloque | Compuerta / puntos | Prueba de éxito en pantalla |
|---|---|---|
| A | G4 + métricas | Voz + consola + «procesado y disponible» + P50/P95 |
| B + C | G5 alta + RAG 20 | Sube MANGO-47 → Clara lo cita en la llamada |
| C colgar | Decisión 20 | Resumen: paciente, síntomas, triaje, fuentes |
| D | G5 baja | Tras eliminar, Clara declara que no tiene esa info |
| E | Decisión 20 | Pastilla **rojo · ALERTA** + resumen con alerta |
| Cámara | Video 15 | Las dos preguntas oficiales, mirando a cámara |

---

## 5. Entrega

1. Sube el MP4 a YouTube (no listado) o Drive (cualquiera con el enlace).
2. Pega el enlace en el formulario de **entrega final** de
   [sourcemeridian.com/tech-sphere-challenge](https://sourcemeridian.com/tech-sphere-challenge)
   junto con la URL del repo.
3. El video **no** va en GitHub. El archivo `docs/DEMO-GRABACION.md` es solo para ti.

"""Configuración central. Todo es sobreescribible por variables de entorno POSTOP_*."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("POSTOP_MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.getenv("POSTOP_DATA_DIR", ROOT / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
LOGS_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "knowledge.db"
DATASET_DIR = Path(os.getenv(
    "POSTOP_DATASET_DIR",
    ROOT.parent / "repokit" / "ParticipantArtifacts-main" / "dataset",
))

# Perfil de hardware: "principal" (~7 GB) o "ligero" (~5.5 GB). Ver docs/DECISIONES-ARQUITECTURA.md
PROFILE = os.getenv("POSTOP_PROFILE", "principal")

# --- LLM (llama-server) ---
LLM_MODEL = MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLM_HOST = os.getenv("POSTOP_LLM_HOST", "127.0.0.1")
LLM_PORT = int(os.getenv("POSTOP_LLM_PORT", "8081"))
LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}"
LLM_CTX = int(os.getenv("POSTOP_LLM_CTX", "4096"))
# 4 hilos fijos: el equipo de despliegue objetivo tiene 4 núcleos. Las métricas
# se miden con este límite aunque la máquina de desarrollo tenga más.
LLM_THREADS = int(os.getenv("POSTOP_LLM_THREADS", "4"))
LLAMA_SERVER_BIN = MODELS_DIR / "llama-b10313" / "llama-server"

# --- STT (whisper.cpp vía pywhispercpp) ---
STT_MODEL = MODELS_DIR / ("ggml-small-q5_1.bin" if PROFILE == "principal" else "ggml-base-q5_1.bin")
STT_THREADS = int(os.getenv("POSTOP_STT_THREADS", "4"))
# Léxico que orienta a Whisper hacia el dominio (initial_prompt)
STT_PROMPT = (
    "Llamada de seguimiento postoperatorio en Colombia. Apendicectomía, colecistectomía, "
    "colectomía, mastectomía, reemplazo de cadera o rodilla. Dolor, fiebre, herida, "
    "enrojecimiento, secreción, hinchazón, acetaminofén, ibuprofeno. "
    "Tengo treinta y ocho punto cinco de fiebre. El dolor está en siete de diez."
)
# prompt mínimo para respuestas cortas: solo vocabulario, sin frases que el
# modelo pueda "continuar" cuando el audio es breve
STT_PROMPT_CORTO = "Dolor, fiebre, herida, hinchazón, secreción. Uno, dos, tres, cuatro, cinco, seis, siete, ocho, nueve, diez."

# --- TTS ---
TTS_ENGINE = os.getenv("POSTOP_TTS", "kokoro" if PROFILE == "principal" else "piper")
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"
KOKORO_VOICE = os.getenv("POSTOP_KOKORO_VOICE", "ef_dora")
TTS_SPEED = float(os.getenv("POSTOP_TTS_SPEED", "1.0"))
PIPER_VOICE = MODELS_DIR / os.getenv("POSTOP_PIPER_VOICE", "es_MX-claude-high.onnx")

# --- VAD ---
VAD_MODEL = MODELS_DIR / "silero_vad.onnx"
# 650 ms de silencio para cerrar el turno: margen para hablantes lentos o con
# dificultades del habla, sin penalizar demasiado la latencia
VAD_SILENCE_MS = int(os.getenv("POSTOP_VAD_SILENCE_MS", "800"))
VAD_THRESHOLD = float(os.getenv("POSTOP_VAD_THRESHOLD", "0.35"))
# tope de un solo turno hablado: protege de ruido continuo que nunca cierre
VAD_MAX_UTTERANCE_S = float(os.getenv("POSTOP_VAD_MAX_UTTERANCE_S", "30"))
# segundos de silencio del paciente (contados DESDE que Clara deja de hablar)
# antes de que el agente retome la conversación
SILENCIO_S = float(os.getenv("POSTOP_SILENCIO_S", "20"))
# diagnóstico: guarda en data/logs/audio/ lo que llega del navegador
GUARDAR_AUDIO = os.getenv("POSTOP_GUARDAR_AUDIO", "0") == "1"

# --- RAG ---
EMB_MODEL = MODELS_DIR / "e5-small-q8.onnx"
EMB_TOKENIZER = MODELS_DIR / "e5-tokenizer.json"
EMB_DIM = 384
CHILD_TOKENS = 250        # tamaño de chunk hijo (aprox, en tokens del tokenizer e5)
CHILD_OVERLAP = 40
PARENT_MAX_TOKENS = 1100  # tamaño máximo de sección padre
TOP_K_LEXICAL = 30
TOP_K_DENSE = 30
RRF_K = 60
# multiplicador para documentos dirigidos al paciente en la fusión
BOOST_PACIENTE = float(os.getenv("POSTOP_BOOST_PACIENTE", "1.6"))
# distancia coseno máxima admitida: por encima, no hay fuente pertinente
DIST_MAX = float(os.getenv("POSTOP_DIST_MAX", "0.42"))
# 2 fuentes para no inflar el prefill en CPU de 4 núcleos (cada fuente extra
# son ~500 tokens ≈ 8-10 s de latencia); la fusión RRF ya ordenó por relevancia
TOP_K_FINAL = int(os.getenv("POSTOP_TOP_K", "2"))
USE_RERANKER = os.getenv("POSTOP_RERANKER", "0") == "1"

# --- Servidor web ---
WEB_HOST = os.getenv("POSTOP_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("POSTOP_PORT", "8000"))

SCENARIOS = {
    "apendicectomia": ["apendicitis", "apendicectom", "appendicitis", "appendectomy", "appendix"],
    "colecistectomia": ["colecist", "vesícula", "vesicula biliar", "cholecyst", "gallbladder", "biliar"],
    "colectomia": ["colectom", "colorrectal", "colorectal", "colon", "rectal"],
    "mastectomia": ["mastectom", "mama", "breast", "seno", "cuello uterino", "cervical", "cervix", "cérvix"],
    "reemplazo_articular": ["cadera", "rodilla", "articular", "artroplast", "hip", "knee", "joint replacement", "arthroplasty"],
}

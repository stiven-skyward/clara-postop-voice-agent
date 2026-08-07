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
LLM_THREADS = int(os.getenv("POSTOP_LLM_THREADS", str(max(2, (os.cpu_count() or 4) - 2))))
LLAMA_SERVER_BIN = MODELS_DIR / "llama-b10313" / "llama-server"

# --- STT (whisper.cpp vía pywhispercpp) ---
STT_MODEL = MODELS_DIR / ("ggml-small-q5_1.bin" if PROFILE == "principal" else "ggml-base-q5_1.bin")
STT_THREADS = int(os.getenv("POSTOP_STT_THREADS", "4"))
# Léxico que orienta a Whisper hacia el dominio (initial_prompt)
STT_PROMPT = (
    "Llamada de seguimiento postoperatorio en Colombia. Apendicectomía, colecistectomía, "
    "colectomía, mastectomía, reemplazo de cadera o rodilla. Dolor, fiebre, herida, "
    "enrojecimiento, secreción, hinchazón, acetaminofén, ibuprofeno."
)

# --- TTS ---
TTS_ENGINE = os.getenv("POSTOP_TTS", "kokoro" if PROFILE == "principal" else "piper")
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"
KOKORO_VOICE = os.getenv("POSTOP_KOKORO_VOICE", "ef_dora")
PIPER_VOICE = MODELS_DIR / os.getenv("POSTOP_PIPER_VOICE", "es_MX-claude-high.onnx")

# --- VAD ---
VAD_MODEL = MODELS_DIR / "silero_vad.onnx"
VAD_SILENCE_MS = int(os.getenv("POSTOP_VAD_SILENCE_MS", "500"))  # fin de habla
VAD_THRESHOLD = float(os.getenv("POSTOP_VAD_THRESHOLD", "0.5"))

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
TOP_K_FINAL = 4
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

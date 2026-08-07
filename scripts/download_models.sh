#!/usr/bin/env bash
# Descarga todos los modelos (CPU-only) a models/. Total ≈ 3.1 GB.
set -euo pipefail
cd "$(dirname "$0")/../models"

dl() { [ -f "$2" ] && { echo "✔ $2 (ya existe)"; return; }; echo "↓ $2"; curl -L --fail --progress-bar -o "$2" "$1"; }

# LLM — Llama 3.2 3B Instruct Q4_K_M (modelo permitido por el reto, local CPU)
dl "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Runtime llama.cpp (llama-server precompilado, Linux x64)
if [ ! -d llama-b10313 ]; then
  dl "https://github.com/ggml-org/llama.cpp/releases/download/b10313/llama-b10313-bin-ubuntu-x64.tar.gz" llama-cpp.tar.gz
  tar xzf llama-cpp.tar.gz && rm llama-cpp.tar.gz
fi

# STT — whisper.cpp (small = perfil principal, base = perfil ligero)
dl "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin" ggml-small-q5_1.bin
dl "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin" ggml-base-q5_1.bin

# TTS — Kokoro-82M ONNX + voces (principal) y Piper (ligero)
dl "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" kokoro-v1.0.onnx
dl "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" voices-v1.0.bin
dl "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx" es_MX-claude-high.onnx
dl "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json" es_MX-claude-high.onnx.json

# VAD — Silero
dl "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" silero_vad.onnx

# Embeddings — multilingual-e5-small int8 ONNX + tokenizer
dl "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/onnx/model_quantized.onnx" e5-small-q8.onnx
dl "https://huggingface.co/Xenova/multilingual-e5-small/resolve/main/tokenizer.json" e5-tokenizer.json

echo "✅ Modelos listos."

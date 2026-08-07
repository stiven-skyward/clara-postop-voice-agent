"""Embeddings multilingual-e5-small int8 ONNX, CPU-only.

e5 exige prefijos: "query: ..." para consultas, "passage: ..." para chunks.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app import config


class Embedder:
    def __init__(self) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(config.EMB_MODEL), opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(config.EMB_TOKENIZER))
        self.tokenizer.enable_truncation(max_length=512)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encs = self.tokenizer.encode_batch(texts)
        maxlen = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), maxlen), dtype=np.int64)
        mask = np.zeros((len(encs), maxlen), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = e.attention_mask
        out = self.session.run(
            None, {"input_ids": ids, "attention_mask": mask}
        )[0]  # (B, T, 384) last_hidden_state
        # mean pooling con máscara + normalización L2 (receta oficial e5)
        m = mask[:, :, None].astype(np.float32)
        emb = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
        return emb.astype(np.float32)

    def embed_passages(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            vecs.append(self._encode_batch([f"passage: {t}" for t in texts[i : i + batch_size]]))
        return np.vstack(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode_batch([f"query: {text}"])[0]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

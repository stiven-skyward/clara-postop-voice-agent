"""Recuperación híbrida: BM25S + denso (sqlite-vec) → RRF → small-to-big.

El índice BM25 vive en memoria y se reconstruye (hot-swap atómico) en cada
alta/baja de documento — es cuestión de segundos a esta escala y garantiza
que el conocimiento eliminado no deja rastro (G5).
"""
from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field

import bm25s

from app import config
from app.rag.embed import get_embedder
from app.rag.lexicon import expand_query
from app.rag.store import get_store

_STOP_ES = set("""a al algo como con de del desde donde dos el ella ellos en entre era eran es esa ese eso esta estas este estos fue ha han hasta hay la las le les lo los mas me mi mientras muy no nos o os otra otro para pero poco por porque que quien se ser si sin sobre su sus te tiene tienen un una uno unos y ya""".split())
_STOP_EN = set("""a an and are as at be by for from has have in is it its of on or that the this to was were will with""".split())


def _tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    words = re.findall(r"[a-z0-9]{2,}", t)
    return [w for w in words if w not in _STOP_ES and w not in _STOP_EN]


@dataclass
class Source:
    """Una fuente entregada al LLM y citable."""
    n: int
    doc_id: str
    titulo: str
    seccion: str
    pagina_ini: int
    pagina_fin: int
    texto: str
    score: float
    chunk_ids: list[int] = field(default_factory=list)
    hijo: str = ""     # fragmento que realmente casó con la consulta

    def ventana(self, max_chars: int = 1200) -> str:
        """Texto para el prompt: ventana del padre CENTRADA en el fragmento que
        casó. Cortar por el principio del padre dejaba fuera lo relevante."""
        if len(self.texto) <= max_chars:
            return self.texto
        pos = self.texto.find(self.hijo[:80]) if self.hijo else -1
        if pos < 0:
            return self.texto[:max_chars]
        ini = max(0, pos - max_chars // 4)
        return ("…" if ini else "") + self.texto[ini:ini + max_chars] + "…"


class Searcher:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rebuild_lock = threading.Lock()
        self._bm25: bm25s.BM25 | None = None
        self._ids: list[int] = []
        self.rebuild()

    def rebuild(self) -> None:
        """Reconstruye el índice BM25 desde el store y lo intercambia atómicamente.

        Serializado: dos rebuilds concurrentes (alta y baja a la vez) podían
        intercambiar en orden inverso y publicar un índice obsoleto.
        """
        with self._rebuild_lock:
            self._rebuild()

    def _rebuild(self) -> None:
        store = get_store()
        chunks = store.all_chunks()
        titles = {d["doc_id"]: d["titulo"] for d in store.list_documents()}
        ids, corpus = [], []
        for c in chunks:
            ids.append(c.chunk_id)
            corpus.append(_tokenize(f"{titles.get(c.doc_id, '')} {c.seccion} {c.texto}"))
        bm = None
        if corpus:
            bm = bm25s.BM25()
            bm.index(corpus, show_progress=False)
        with self._lock:
            self._bm25, self._ids = bm, ids

    def _lexical(self, query: str, k: int) -> list[tuple[int, float]]:
        with self._lock:
            bm, ids = self._bm25, self._ids
        if bm is None or not ids:
            return []
        toks = _tokenize(query)
        if not toks:
            return []
        res, scores = bm.retrieve([toks], k=min(k, len(ids)), show_progress=False)
        return [(ids[int(i)], float(s)) for i, s in zip(res[0], scores[0]) if s > 0]

    def search(self, query: str, escenario: str | None = None) -> list[Source]:
        store = get_store()
        expanded = expand_query(query)

        lex = self._lexical(expanded, config.TOP_K_LEXICAL)
        qvec = get_embedder().embed_query(query)
        den = store.dense_search(qvec, config.TOP_K_DENSE, escenario=escenario)

        # filtro de escenario sobre la rama léxica (la densa ya filtra en SQL)
        if escenario and lex:
            meta = store.get_chunks([i for i, _ in lex])
            docs = {d["doc_id"]: d["escenario"] for d in store.list_documents()}
            # meta.get: el índice BM25 puede traer ids ya borrados por la consola
            # (KeyError aquí mataba el turno completo del paciente)
            lex = [(i, s) for i, s in lex
                   if i in meta and docs.get(meta[i].doc_id) in (escenario, "general")]

        # Fusión RRF
        rrf: dict[int, float] = {}
        for rank, (cid, _) in enumerate(lex):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        for rank, (cid, _) in enumerate(den):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        top = sorted(rrf.items(), key=lambda x: -x[1])[: config.TOP_K_FINAL * 3]
        if not top:
            return []

        # Small-to-big: hijo → sección padre, dedupe por parent_id
        chunks = store.get_chunks([cid for cid, _ in top])
        sources: list[Source] = []
        seen_parents: set[int] = set()
        seen_text: set[str] = set()  # dedupe de docs casi idénticos del corpus
        for cid, score in top:
            ch = chunks.get(cid)
            if ch is None or ch.parent_id in seen_parents:
                continue
            seen_parents.add(ch.parent_id)
            parent = store.get_parent(ch.parent_id)
            if parent is None:
                continue
            doc_id, seccion, p0, p1, texto = parent
            key = re.sub(r"\s+", "", texto)[:300].lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            sources.append(Source(
                n=len(sources) + 1, doc_id=doc_id, titulo=store.doc_title(doc_id),
                seccion=seccion, pagina_ini=p0, pagina_fin=p1, texto=texto,
                score=score, chunk_ids=[cid], hijo=ch.texto,
            ))
            if len(sources) >= config.TOP_K_FINAL:
                break
        return sources


_searcher: Searcher | None = None


def get_searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return _searcher

"""Almacén de conocimiento: SQLite + sqlite-vec en un único fichero.

Tablas:
  documents(doc_id PK, titulo, fichero, idioma, escenario, paginas, n_chunks, estado, ts)
  parents(parent_id PK, doc_id, seccion, pagina_ini, pagina_fin, texto)
  chunks(chunk_id PK, doc_id, parent_id, seccion, pagina, texto)   -- hijos indexables
  vec_chunks(chunk_id, embedding float[384])                        -- sqlite-vec

El borrado de un documento es UNA transacción sobre las 4 tablas → olvido real e
inmediato, verificable con SELECT (compuerta G5).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np
import sqlite_vec

from app import config


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    parent_id: int
    seccion: str
    pagina: int
    texto: str


class Store:
    """Una conexión SQLite POR HILO (threading.local).

    Compartir una sola conexión entre los hilos de las llamadas y el de ingesta
    provoca lecturas sucias, `cannot commit - SQL statements in progress` y
    cursores invalidados. Con WAL + conexión por hilo, lectores y escritor
    conviven; `lock` solo serializa a los escritores entre sí.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.lock = threading.RLock()
        self.db_path = str(db_path or config.DB_PATH)
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = self._connect()
        return c

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents(
                  doc_id TEXT PRIMARY KEY, titulo TEXT, fichero TEXT, idioma TEXT,
                  escenario TEXT, paginas INTEGER, n_chunks INTEGER,
                  estado TEXT DEFAULT 'procesando', ts REAL);
                CREATE TABLE IF NOT EXISTS parents(
                  parent_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, seccion TEXT,
                  pagina_ini INTEGER, pagina_fin INTEGER, texto TEXT);
                CREATE TABLE IF NOT EXISTS chunks(
                  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT,
                  parent_id INTEGER, seccion TEXT, pagina INTEGER, texto TEXT);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_parents_doc ON parents(doc_id);
                """
            )
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                f"chunk_id INTEGER PRIMARY KEY, embedding float[{config.EMB_DIM}])"
            )

    # --- Ingesta ---
    def doc_exists(self, doc_id: str) -> bool:
        r = self.conn.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return r is not None

    def add_document(
        self,
        doc_id: str,
        titulo: str,
        fichero: str,
        idioma: str,
        escenario: str,
        paginas: int,
        sections: list[dict],   # {seccion, pagina_ini, pagina_fin, texto, children:[{texto, pagina}]}
        embeddings: np.ndarray,  # (n_children_total, 384) en el mismo orden que children aplanados
    ) -> int:
        n = 0
        with self.lock, self.conn:
            # limpieza previa: dos ingestas concurrentes del mismo fichero
            # pasarían ambas el check de doc_exists y duplicarían los chunks
            self.conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE doc_id=?)", (doc_id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM parents WHERE doc_id=?", (doc_id,))
            self.conn.execute(
                "INSERT OR REPLACE INTO documents VALUES(?,?,?,?,?,?,?, 'procesando', ?)",
                (doc_id, titulo, fichero, idioma, escenario, paginas, 0, time.time()),
            )
            for sec in sections:
                cur = self.conn.execute(
                    "INSERT INTO parents(doc_id, seccion, pagina_ini, pagina_fin, texto) VALUES(?,?,?,?,?)",
                    (doc_id, sec["seccion"], sec["pagina_ini"], sec["pagina_fin"], sec["texto"]),
                )
                pid = cur.lastrowid
                for ch in sec["children"]:
                    cur2 = self.conn.execute(
                        "INSERT INTO chunks(doc_id, parent_id, seccion, pagina, texto) VALUES(?,?,?,?,?)",
                        (doc_id, pid, sec["seccion"], ch["pagina"], ch["texto"]),
                    )
                    self.conn.execute(
                        "INSERT INTO vec_chunks(chunk_id, embedding) VALUES(?,?)",
                        (cur2.lastrowid, embeddings[n].tobytes()),
                    )
                    n += 1
            self.conn.execute(
                "UPDATE documents SET n_chunks=?, estado='disponible' WHERE doc_id=?", (n, doc_id)
            )
        return n

    def delete_document(self, doc_id: str) -> bool:
        """Olvido total en una transacción. Devuelve True si el documento existía."""
        with self.lock, self.conn:
            if not self.doc_exists(doc_id):
                return False
            # subconsulta en vez de IN (?,?,…): un doc con miles de chunks
            # superaría el límite de variables de SQLite
            self.conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE doc_id=?)", (doc_id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM parents WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        return True

    # --- Consulta ---
    def list_documents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT doc_id, titulo, fichero, idioma, escenario, paginas, n_chunks, estado, ts "
            "FROM documents ORDER BY ts DESC"
        ).fetchall()
        keys = ["doc_id", "titulo", "fichero", "idioma", "escenario", "paginas", "n_chunks", "estado", "ts"]
        return [dict(zip(keys, r)) for r in rows]

    def all_chunks(self, escenario: str | None = None) -> list[Chunk]:
        sql = "SELECT chunk_id, doc_id, parent_id, seccion, pagina, texto FROM chunks"
        args: tuple = ()
        if escenario:
            sql += " WHERE doc_id IN (SELECT doc_id FROM documents WHERE escenario=? OR escenario='general')"
            args = (escenario,)
        return [Chunk(*r) for r in self.conn.execute(sql, args).fetchall()]

    def dense_search(self, qvec: np.ndarray, k: int, escenario: str | None = None) -> list[tuple[int, float]]:
        """Devuelve [(chunk_id, distancia)] por similitud coseno (vectores normalizados)."""
        if escenario:
            rows = self.conn.execute(
                """SELECT v.chunk_id, vec_distance_cosine(v.embedding, ?) AS d
                   FROM vec_chunks v JOIN chunks c ON c.chunk_id = v.chunk_id
                   WHERE c.doc_id IN (SELECT doc_id FROM documents WHERE escenario=? OR escenario='general')
                   ORDER BY d LIMIT ?""",
                (qvec.tobytes(), escenario, k),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? AND k=? ORDER BY distance",
                (qvec.tobytes(), k),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_chunks(self, ids: list[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, doc_id, parent_id, seccion, pagina, texto FROM chunks WHERE chunk_id IN ({q})",
            ids,
        ).fetchall()
        return {r[0]: Chunk(*r) for r in rows}

    def get_parent(self, parent_id: int) -> tuple[str, str, int, int, str] | None:
        """(doc_id, seccion, pagina_ini, pagina_fin, texto)"""
        return self.conn.execute(
            "SELECT doc_id, seccion, pagina_ini, pagina_fin, texto FROM parents WHERE parent_id=?",
            (parent_id,),
        ).fetchone()

    def doc_title(self, doc_id: str) -> str:
        r = self.conn.execute("SELECT titulo FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return r[0] if r else doc_id


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store

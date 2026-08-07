"""Ingesta de documentos (PDF/TXT/MD) → secciones padre → chunks hijos → índices.

- PyMuPDF para PDFs; detección de secciones por TOC o heurística de tamaño de fuente.
- PDF sin capa de texto → OCR con tesseract (spa+eng) vía subproceso.
- Clasificación de escenario quirúrgico e idioma por heurística de keywords.
- Chunk hijo lleva header contextual [doc | sección] al embeber (no se guarda en texto).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

import pymupdf as fitz

from app import config
from app.rag.embed import get_embedder
from app.rag.store import get_store

_ES_HINTS = [" de ", " la ", " el ", " que ", " los ", " para ", " con ", " del "]
_EN_HINTS = [" the ", " of ", " and ", " with ", " for ", " that ", " this "]


def detect_language(text: str) -> str:
    low = f" {text.lower()} "
    es = sum(low.count(w) for w in _ES_HINTS)
    en = sum(low.count(w) for w in _EN_HINTS)
    return "es" if es >= en else "en"


def detect_scenario(title: str, text: str) -> str:
    hay = (title + " " + text[:5000]).lower()
    scores = {
        esc: sum(hay.count(kw) for kw in kws) for esc, kws in config.SCENARIOS.items()
    }
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else "general"


def _ocr_pdf(path: Path) -> fitz.Document:
    """OCR página a página con tesseract; devuelve doc con texto extraíble."""
    doc = fitz.open(str(path))
    texts = []
    with tempfile.TemporaryDirectory() as td:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            img = Path(td) / f"p{i}.png"
            pix.save(str(img))
            r = subprocess.run(
                ["tesseract", str(img), "-", "-l", "spa+eng", "--psm", "1"],
                capture_output=True, text=True, timeout=120,
            )
            texts.append(r.stdout)
    out = fitz.open()
    for t in texts:
        p = out.new_page()
        p.insert_text((36, 36), t or " ", fontsize=9)
    return out


def _extract_sections_pdf(path: Path) -> tuple[list[dict], int, str]:
    """Devuelve (secciones, n_páginas, texto_muestra). Sección: dict con
    seccion, pagina_ini, pagina_fin, texto."""
    doc = fitz.open(str(path))
    n_pages = doc.page_count
    sample = "".join(doc[i].get_text() for i in range(min(3, n_pages)))
    if len(sample.strip()) < 50:  # sin capa de texto → OCR
        doc = _ocr_pdf(path)
        sample = "".join(doc[i].get_text() for i in range(min(3, doc.page_count)))

    toc = doc.get_toc(simple=True)
    sections: list[dict] = []
    if toc and len(toc) >= 3:
        # Secciones desde el TOC (nivel <= 2)
        marks = [(t[1].strip(), max(0, t[2] - 1)) for t in toc if t[0] <= 2 and t[1].strip()]
        for i, (title, p0) in enumerate(marks):
            p1 = marks[i + 1][1] if i + 1 < len(marks) else doc.page_count - 1
            text = "".join(doc[p].get_text() for p in range(p0, min(p1 + 1, doc.page_count)))
            if text.strip():
                sections.append({"seccion": title[:120], "pagina_ini": p0 + 1,
                                 "pagina_fin": p1 + 1, "texto": text})
    if not sections:
        # Heurística: títulos = líneas cortas con fuente mayor a la mediana del doc
        sizes = []
        pages_blocks = []
        for p in range(doc.page_count):
            blocks = doc[p].get_text("dict")["blocks"]
            pages_blocks.append(blocks)
            for b in blocks:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        if s["text"].strip():
                            sizes.append(s["size"])
        body = sorted(sizes)[len(sizes) // 2] if sizes else 10
        cur = {"seccion": "Inicio", "pagina_ini": 1, "texto": ""}
        for p, blocks in enumerate(pages_blocks):
            for b in blocks:
                for l in b.get("lines", []):
                    line = "".join(s["text"] for s in l.get("spans", [])).strip()
                    if not line:
                        continue
                    mx = max(s["size"] for s in l["spans"])
                    is_head = (mx > body * 1.18 and 3 < len(line) < 90
                               and not line[-1] in ".,;")
                    if is_head and len(cur["texto"]) > 400:
                        cur["pagina_fin"] = p + 1
                        sections.append(cur)
                        cur = {"seccion": line[:120], "pagina_ini": p + 1, "texto": ""}
                    else:
                        cur["texto"] += line + "\n"
        cur["pagina_fin"] = doc.page_count
        if cur["texto"].strip():
            sections.append(cur)
    return sections, n_pages, sample


def _split_parent(sections: list[dict], count_tokens) -> list[dict]:
    """Limita padres a PARENT_MAX_TOKENS partiendo por párrafos si hace falta."""
    out = []
    for sec in sections:
        text = re.sub(r"\n{3,}", "\n\n", sec["texto"]).strip()
        if not text:
            continue
        if count_tokens(text) <= config.PARENT_MAX_TOKENS:
            out.append({**sec, "texto": text})
            continue
        paras = re.split(r"\n\n+", text)
        buf, part = [], 1
        for para in paras:
            buf.append(para)
            if count_tokens("\n\n".join(buf)) > config.PARENT_MAX_TOKENS:
                out.append({**sec, "seccion": f"{sec['seccion']} ({part})",
                            "texto": "\n\n".join(buf[:-1]) or para})
                buf, part = [para], part + 1
        if buf and "\n\n".join(buf).strip():
            suffix = f" ({part})" if part > 1 else ""
            out.append({**sec, "seccion": f"{sec['seccion']}{suffix}", "texto": "\n\n".join(buf)})
    return out


def _split_children(parent_text: str, count_tokens) -> list[str]:
    """Trocea el padre en hijos ~CHILD_TOKENS con solape por oraciones."""
    sents = re.split(r"(?<=[.!?])\s+|\n(?=[A-ZÁÉÍÓÚÑ•\-])", parent_text)
    sents = [s.strip() for s in sents if s.strip()]
    children, buf, buft = [], [], 0
    for s in sents:
        t = count_tokens(s)
        if buft + t > config.CHILD_TOKENS and buf:
            children.append(" ".join(buf))
            # solape: conserva las últimas oraciones ~CHILD_OVERLAP tokens
            keep, kt = [], 0
            for prev in reversed(buf):
                kt += count_tokens(prev)
                keep.insert(0, prev)
                if kt >= config.CHILD_OVERLAP:
                    break
            buf, buft = keep, kt
        buf.append(s)
        buft += t
    if buf:
        children.append(" ".join(buf))
    return [c for c in children if len(c) > 60]


def ingest_file(path: Path, original_name: str | None = None) -> dict:
    """Ingesta completa de un fichero. Devuelve el registro del documento."""
    path = Path(path)
    name = original_name or path.name
    raw = path.read_bytes()
    doc_id = hashlib.sha256(raw).hexdigest()[:16]
    store = get_store()
    if store.doc_exists(doc_id):
        return {"doc_id": doc_id, "estado": "duplicado", "titulo": name}

    if path.suffix.lower() == ".pdf":
        sections, n_pages, sample = _extract_sections_pdf(path)
    else:
        text = raw.decode("utf-8", errors="replace")
        sample = text[:3000]
        n_pages = 1
        # Markdown/TXT: secciones por encabezados '#' o bloques
        parts = re.split(r"\n(?=#{1,3} )", text)
        sections = []
        for part in parts:
            m = re.match(r"#{1,3} (.+)", part)
            sections.append({
                "seccion": (m.group(1).strip() if m else name)[:120],
                "pagina_ini": 1, "pagina_fin": 1, "texto": part,
            })

    emb = get_embedder()
    idioma = detect_language(sample or name)
    titulo = Path(name).stem.replace("_", " ").replace("-", " ").strip()[:150]
    escenario = detect_scenario(titulo, sample)

    parents = _split_parent(sections, emb.count_tokens)
    all_children_texts: list[str] = []
    for sec in parents:
        children = _split_children(sec["texto"], emb.count_tokens)
        sec["children"] = [{"texto": c, "pagina": sec["pagina_ini"]} for c in children]
        # header contextual SOLO para el embedding y BM25 (mejora retrieval)
        all_children_texts += [
            f"{titulo} | {sec['seccion']} | {c}" for c in children
        ]
    parents = [s for s in parents if s["children"]]
    if not all_children_texts:
        return {"doc_id": doc_id, "estado": "error_sin_texto", "titulo": titulo}

    embeddings = emb.embed_passages(all_children_texts)
    n = store.add_document(doc_id, titulo, name, idioma, escenario, n_pages, parents, embeddings)

    from app.rag.search import get_searcher
    get_searcher().rebuild()  # hot-swap del índice BM25

    return {"doc_id": doc_id, "estado": "disponible", "titulo": titulo,
            "idioma": idioma, "escenario": escenario, "n_chunks": n}


def delete_document(doc_id: str) -> bool:
    ok = get_store().delete_document(doc_id)
    if ok:
        from app.rag.search import get_searcher
        get_searcher().rebuild()
    return ok

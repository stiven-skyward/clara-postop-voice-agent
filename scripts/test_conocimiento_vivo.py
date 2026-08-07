"""Prueba de la compuerta G5: conocimiento vivo.

1. Sube por la API un documento NUEVO (no está en ningún corpus del reto).
2. Verifica que la búsqueda del agente lo recupera de inmediato.
3. Lo elimina por la API.
4. Verifica que el agente lo olvidó por completo (búsqueda y SQL).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx

from app import config

DOC = """# Protocolo interno XK-2026 de hidratación postoperatoria

## Regla del vaso violeta
Tras una apendicectomía, el protocolo XK-2026 indica beber exactamente siete
vasos de agua al día durante los primeros cinco días, usando la técnica del
"vaso violeta": pequeños sorbos cada veinte minutos para no distender el
estómago. Si el paciente presenta hipo persistente por más de dos horas, debe
suspenderse la regla y avisar a enfermería.
"""

BASE = f"http://{config.WEB_HOST}:{config.WEB_PORT}"


def search_hits(q: str) -> list[str]:
    from app.rag.search import Searcher
    s = Searcher()  # índice fresco desde la BD en cada verificación
    return [f"{r.titulo} | {r.seccion}" for r in s.search(q, escenario=None)]


def main() -> None:
    tmp = Path(config.DATA_DIR) / "uploads" / "_test_g5.md"
    tmp.write_text(DOC, encoding="utf-8")

    with httpx.Client(timeout=120) as c:
        r = c.post(f"{BASE}/api/docs", files={"file": ("protocolo-xk-2026.md", DOC.encode())}).json()
        print("1) SUBIDA:", r)
        doc_id = r["doc_id"]

        t0 = time.time()
        hits = search_hits("¿Cuántos vasos de agua debo tomar según el protocolo XK?")
        found = any("xk 2026" in h.lower() or "protocolo xk" in h.lower() for h in hits)
        print(f"2) BÚSQUEDA tras subir ({time.time()-t0:.1f}s): "
              f"{'✔ RECUPERADO' if found else '✗ NO ENCONTRADO'}")
        for h in hits[:3]:
            print("   -", h)

        r = c.delete(f"{BASE}/api/docs/{doc_id}").json()
        print("3) ELIMINACIÓN:", r)

        hits = search_hits("¿Cuántos vasos de agua debo tomar según el protocolo XK?")
        found_after = any("xk" in h.lower() for h in hits)
        print(f"4) BÚSQUEDA tras eliminar: "
              f"{'✗ AÚN APARECE (FALLO G5)' if found_after else '✔ OLVIDADO'}")

        import sqlite3
        conn = sqlite3.connect(str(config.DB_PATH))
        n = conn.execute("SELECT count(*) FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()[0]
        d = conn.execute("SELECT count(*) FROM documents WHERE doc_id=?", (doc_id,)).fetchone()[0]
        print(f"5) VERIFICACIÓN SQL: chunks={n}, documents={d} "
              f"{'✔ olvido total' if n == 0 and d == 0 else '✗ RASTROS'}")

    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

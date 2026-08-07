"""Ingesta masiva del corpus del reto (dataset/textos) al conocimiento del agente."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.rag import ingest

def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DATASET_DIR / "textos"
    pdfs = sorted(base.rglob("*.pdf"))
    print(f"Ingesta de {len(pdfs)} PDFs desde {base}")
    t0, ok, err = time.time(), 0, 0
    for i, p in enumerate(pdfs, 1):
        try:
            t = time.time()
            info = ingest.ingest_file(p, p.name)
            ok += 1
            print(f"[{i}/{len(pdfs)}] {info['estado']:<12} {info.get('idioma','?')} "
                  f"{info.get('escenario','?'):<20} {info.get('n_chunks',0):>4} chunks "
                  f"{time.time()-t:5.1f}s  {p.name[:60]}")
        except Exception as e:
            err += 1
            print(f"[{i}/{len(pdfs)}] ERROR {p.name[:60]}: {e}")
    print(f"\nListo: {ok} ok, {err} errores en {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

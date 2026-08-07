"""Genera el bloque de métricas del README desde data/logs/*.jsonl.

Así lo reportado es, por construcción, lo que dicen los logs (rúbrica §5-§6).
Uso: python scripts/report_metrics.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, metrics

# Tarifas públicas de referencia para la extrapolación de costo (USD/1M tokens).
# El agente corre 100% local (costo real $0); esto responde "¿cuánto costaría
# cada llamada si el mismo tráfico fuera a una API de producción?".
TARIFAS = {
    "Groq Llama-3.3-70B": {"in": 0.59, "out": 0.79},
    "Gemini 2.5 Flash":   {"in": 0.30, "out": 2.50},
}


def main() -> None:
    agg = metrics.aggregate()
    if not agg.get("turnos"):
        print("Sin datos en data/logs/turnos.jsonl")
        return

    path = Path(config.LOGS_DIR) / "turnos.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    calls: dict[str, list[dict]] = {}
    for r in rows:
        calls.setdefault(r["call_id"], []).append(r)

    tok_in_call = agg["tokens_por_llamada_prom"]
    # aproximación 90/10 entrada/salida según promedios por turno
    ti, to = agg["tokens_entrada_por_turno_prom"], agg["tokens_salida_por_turno_prom"]
    frac_in = ti / max(1e-9, ti + to)

    print(f"Turnos: {agg['turnos']}  Llamadas: {agg['llamadas']}\n")
    print("| Métrica | Valor medido |")
    print("|---|---|")
    print(f"| Latencia P50 (fin de habla → primer audio) | {agg['latencia_p50_s']} s |")
    print(f"| Latencia P95 | {agg['latencia_p95_s']} s |")
    print(f"| Tokens por turno (entrada / salida, prom.) | {ti} / {to} |")
    print(f"| Tokens por llamada (prom.) | {tok_in_call} |")
    print(f"| Invocaciones LLM por turno (prom.) | {agg['invocaciones_llm_por_turno_prom']} |")
    print(f"| Consultas RAG por llamada (prom.) | {agg['consultas_rag_por_llamada_prom']} |")
    costos = []
    for nombre, t in TARIFAS.items():
        c = (tok_in_call * frac_in * t["in"] + tok_in_call * (1 - frac_in) * t["out"]) / 1e6
        costos.append(f"{nombre}: ${c:.4f}")
    print(f"| Costo por llamada | $0 real (local) · extrapolado: {' · '.join(costos)} |")
    print("\nCálculo del costo: tokens_por_llamada × tarifa pública por 1M tokens, "
          f"repartidos {frac_in*100:.0f}% entrada / {(1-frac_in)*100:.0f}% salida.")


if __name__ == "__main__":
    main()

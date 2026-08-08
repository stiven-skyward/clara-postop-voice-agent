"""Evalúa la lógica de triaje (extracción LLM + reglas) contra el dataset del reto.

Reconstruye cada conversación (capa elegible), pasa los turnos del paciente por
el pipeline real (extracción estructurada → merge → reglas) y compara el nivel
final con label_ground_truth. Reporta matriz de confusión y, sobre todo,
FALSOS NEGATIVOS (casos rojo/amarillo clasificados por debajo).

Uso: python scripts/eval_triage.py [n_verde] [capa]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app import config, llm
from app.agent import prompts
from app.agent.orchestrator import _worth_extracting
from app.agent.triage import (SymptomState, combine, evaluate, fallback_extract,
                              merge_worst, quick_red_scan, sanitize_extraction)

N_VERDE = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CAPA = sys.argv[2] if len(sys.argv) > 2 else "capa1_limpia"


def main():
    llm.ensure_server()
    df = pd.read_excel(config.DATASET_DIR / "dataset_final.xlsx", sheet_name="result")
    clin = pd.read_excel(
        config.DATASET_DIR / "perfiles_clinicos_pacientes_silver_contest.xlsx",
        sheet_name="result")
    proc = dict(zip(clin.paciente_id, clin.procedimiento))

    df = df[df.capa == CAPA]
    cases = df.groupby("caso_id").agg(
        label=("label_ground_truth", "first"),
        paciente=("paciente_id", "first"),
        dia=("dia_postop", "first")).reset_index()
    rojos = cases[cases.label == "rojo"]
    amarillos = cases[cases.label == "amarillo"]
    verdes = cases[cases.label == "verde"].head(N_VERDE)
    sample = pd.concat([rojos, amarillos, verdes])
    print(f"Evaluando {len(sample)} casos ({len(rojos)} rojos, {len(amarillos)} amarillos, "
          f"{len(verdes)} verdes) capa={CAPA}")

    results, t0 = [], time.time()
    for _, case in sample.iterrows():
        turns = df[(df.caso_id == case.caso_id) & (df.hablante == "paciente")
                   ].sort_values("turno_idx")
        state, nivel = SymptomState(), "verde"
        for _, t in turns.iterrows():
            # mismo pipeline que producción: el atajo léxico rojo escala directo
            if quick_red_scan(str(t.texto)):
                nivel = "rojo"
            ext = {}
            if _worth_extracting(str(t.texto)):
                try:
                    ext = llm.structured(
                        [{"role": "system", "content": prompts.SYSTEM_EXTRACCION},
                         {"role": "user", "content": str(t.texto)}],
                        prompts.SCHEMA_EXTRACCION, max_tokens=260)
                    ext = sanitize_extraction(ext, str(t.texto))
                except Exception:
                    ext = {}
            # red de seguridad determinista, igual que en producción
            state.merge(merge_worst(ext, fallback_extract(str(t.texto))))
            tri = evaluate(state, proc.get(case.paciente, ""), int(case.dia))
            nivel = combine(nivel, tri.nivel)
        results.append({"caso_id": case.caso_id, "esperado": case.label,
                        "predicho": nivel, "sintomas": state.as_dict()})
        e, p = case.label, nivel
        mark = "✓" if e == p else ("⚠FN" if ["verde","amarillo","rojo"].index(p) <
                                   ["verde","amarillo","rojo"].index(e) else "FP")
        print(f"  {case.caso_id}: esperado={e} predicho={p} {mark}", flush=True)

    out = Path(config.LOGS_DIR) / f"eval_triage_{CAPA}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    order = ["verde", "amarillo", "rojo"]
    print("\nMatriz (filas=esperado, cols=predicho):")
    print(f"{'':>10}" + "".join(f"{c:>10}" for c in order))
    for e in order:
        row = [sum(1 for r in results if r["esperado"] == e and r["predicho"] == p)
               for p in order]
        print(f"{e:>10}" + "".join(f"{v:>10}" for v in row))
    fn = sum(1 for r in results
             if order.index(r["predicho"]) < order.index(r["esperado"]))
    rojo_ok = sum(1 for r in results if r["esperado"] == "rojo" and r["predicho"] == "rojo")
    n_rojo = sum(1 for r in results if r["esperado"] == "rojo")
    print(f"\nFalsos negativos (subestimación): {fn}/{len(results)}")
    print(f"Recall rojo: {rojo_ok}/{n_rojo}")
    print(f"Tiempo: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

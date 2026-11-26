"""
FASE B.4 — Sensibilidad de ef_search (local)
============================================

A diferencia de M y ef_construction (parámetros de construcción), ef_search es el
parámetro de *consulta* que gobierna el equilibrio Recall/latencia en tiempo de
ejecución. 04_benchmark.py ya lo barre sobre todos los índices, en los dos modelos, así
que aquí no se reconstruye nada: se extraen las curvas por M con el ef_construction de
la configuración de referencia.

Salida: results/ef_search_sensitivity.csv
"""

import pandas as pd

import config as cfg


def main():
    ref = cfg.load_reference()
    df = pd.read_csv(cfg.results_csv_path())
    sub = df[df["ef_construction"] == ref["ef_construction"]]
    cols = ["model", "M", "ef_construction", "ef_search", "recall@1", "recall@5", "recall@10",
            "recall@10_sd", "latency_ms", "latency_ms_iqr", "latency_p95_ms"]
    sub = sub[cols].sort_values(["model", "M", "ef_search"])
    out = cfg.RESULTS_DIR / "ef_search_sensitivity.csv"
    sub.to_csv(out, index=False)
    print(sub.to_string(index=False))
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()

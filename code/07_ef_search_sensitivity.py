"""
FASE B.3 — Análisis de sensibilidad de ef_search (local, CPU)
============================================================

Responde a la observación O1 de la revisión por pares: ef_search se fijó en 50
para las 36 configuraciones sin estudiar su efecto. A diferencia de M y
ef_construction (parámetros de construcción), ef_search es el parámetro de
*consulta* que gobierna el equilibrio Recall/latencia en tiempo de ejecución.

Aquí fijamos la MEJOR configuración del estudio (SciBERT, coseno, M=32,
ef_construction=400) y variamos únicamente ef_search, para trazar la curva de
compromiso Recall@K vs. latencia sobre el mismo hardware (CPU local) en que se
generó la Tabla 2 del paper.

El índice se construye UNA sola vez; solo se reasigna index.hnsw.efSearch entre
mediciones (así aislamos el efecto del parámetro de consulta). La latencia se
promedia sobre varias repeticiones porque las consultas son sub-milisegundo y
una sola pasada es ruidosa.

Salida: results/ef_search_sensitivity.csv (una fila por valor de ef_search).
"""

import time
import numpy as np
import pandas as pd
import faiss

import config as cfg
from importlib import import_module

# Reutilizamos las mismas métricas del benchmark principal para que los
# resultados sean directamente comparables con results.csv.
_bench = import_module("04_benchmark")
recall_at_k = _bench.recall_at_k
mrr = _bench.mrr
normalized = _bench.normalized

# ---------------------------------------------------------------------------
# Configuración fija: la mejor del estudio (por Recall@10)
# ---------------------------------------------------------------------------
BEST = {"model": "scibert", "metric": "cosine", "M": 32, "ef_construction": 400}

# Rejilla de ef_search a estudiar. Incluye el valor fijo original (50) para que
# la fila coincida exactamente con la Tabla 2 y sirva de anclaje/validación.
# ef_search >= K_MAX (=10) es requisito para una búsqueda con sentido.
EF_SEARCH_GRID = [10, 20, 50, 100, 200, 400]

N_REPEATS = 5   # repeticiones de la búsqueda para promediar latencia


def build_best_index():
    """Construye el índice HNSW de la mejor configuración una sola vez."""
    corpus = np.load(cfg.corpus_emb_path(BEST["model"]))
    queries = np.load(cfg.query_emb_path(BEST["model"]))
    truth = np.load(cfg.groundtruth_path(BEST["model"], BEST["metric"]))

    d = corpus.shape[1]
    # coseno -> vectores normalizados + producto interno (idéntico a 04_benchmark)
    corpus = normalized(corpus)
    queries = normalized(queries)
    index = faiss.IndexHNSWFlat(d, BEST["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = BEST["ef_construction"]

    t0 = time.perf_counter()
    index.add(corpus)
    build_time = time.perf_counter() - t0
    print(f"Índice construido en {build_time:.1f}s "
          f"({BEST['model']}, {BEST['metric']}, M={BEST['M']}, "
          f"ef_c={BEST['ef_construction']})\n")
    return index, queries, truth


def measure(index, queries, truth, ef_search):
    """Mide Recall@K, MRR y latencia media para un valor de ef_search."""
    index.hnsw.efSearch = ef_search

    # Latencia: promedio sobre N_REPEATS pasadas (descartamos la primera como
    # warm-up para que cachés y JIT interno no sesguen el tiempo).
    _, retrieved = index.search(queries, cfg.K_MAX)   # warm-up
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _, retrieved = index.search(queries, cfg.K_MAX)
        times.append(time.perf_counter() - t0)
    latency_ms = np.mean(times) / len(queries) * 1000

    return {
        "model": BEST["model"],
        "metric": BEST["metric"],
        "M": BEST["M"],
        "ef_construction": BEST["ef_construction"],
        "ef_search": ef_search,
        "latency_ms": round(latency_ms, 4),
        "recall@1": round(recall_at_k(retrieved, truth, 1), 4),
        "recall@5": round(recall_at_k(retrieved, truth, 5), 4),
        "recall@10": round(recall_at_k(retrieved, truth, 10), 4),
        "mrr": round(mrr(retrieved, truth), 4),
    }


def main():
    index, queries, truth = build_best_index()

    rows = []
    for ef_s in EF_SEARCH_GRID:
        row = measure(index, queries, truth, ef_s)
        rows.append(row)
        marker = "  <- valor original (Tabla 2)" if ef_s == cfg.EF_SEARCH else ""
        print(f"ef_search={ef_s:<4d} | "
              f"R@1={row['recall@1']:.3f} R@5={row['recall@5']:.3f} "
              f"R@10={row['recall@10']:.3f} MRR={row['mrr']:.3f} "
              f"lat={row['latency_ms']:.3f}ms{marker}")

    df = pd.DataFrame(rows)
    out = cfg.RESULTS_DIR / "ef_search_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()

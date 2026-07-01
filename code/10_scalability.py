"""
FASE B.6 — Escalabilidad: latencia HNSW vs. búsqueda exacta según tamaño (local, CPU)
=====================================================================================

Responde a la observación O9 de la revisión: el estudio usa 49\,000 documentos, un
tamaño modesto, y la ventaja de HNSW sobre la búsqueda exacta ---cuya complejidad
es O(N·d) frente a O(log N)--- solo se vuelve dramática al crecer el corpus.

Aquí medimos, sobre la mejor configuración (SciBERT, coseno, M=32,
ef_construction=400, ef_search=50), la latencia por consulta de HNSW y de la
búsqueda exacta (IndexFlatIP) para subconjuntos ANIDADOS y reproducibles del
corpus. Como solo disponemos de 49\,000 embeddings, el barrido llega hasta ese
tamaño; el objetivo es exhibir la diferencia de pendiente (lineal vs. logarítmica),
no un tamaño de producción.

Para cada N medimos además el Recall@10 de HNSW frente a la búsqueda exacta sobre
ese mismo subconjunto, para comprobar que la exactitud no se degrada al crecer N.

Salida: results/scalability.csv (una fila por tamaño).
"""

import time
import numpy as np
import pandas as pd
import faiss

import config as cfg
from importlib import import_module

_bench = import_module("04_benchmark")
normalized = _bench.normalized
recall_at_k = _bench.recall_at_k

MODEL = "scibert"
M, EF_C = 32, 400
N_REPEATS = 5

# Si existe el pool de escalabilidad (100k, generado por colab_embed_scale.py) lo
# usamos y extendemos la curva; si no, caemos a los 49k del experimento principal.
SCALE_PATH = cfg.DATA_DIR / "corpus_scibert_scale.npy"


def latency_ms(index, queries):
    """Latencia media por consulta (ms), con warm-up y promedio de N_REPEATS."""
    index.search(queries, cfg.K_MAX)  # warm-up
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        _, ret = index.search(queries, cfg.K_MAX)
        times.append(time.perf_counter() - t0)
    return np.mean(times) / len(queries) * 1000, ret


def main():
    if SCALE_PATH.exists():
        corpus = normalized(np.load(SCALE_PATH))
        print(f"Usando pool de escalabilidad: {corpus.shape[0]:,} vectores")
    else:
        corpus = normalized(np.load(cfg.corpus_emb_path(MODEL)))
        print(f"Pool de 100k no encontrado; usando el corpus de {corpus.shape[0]:,}")
    queries = normalized(np.load(cfg.query_emb_path(MODEL)))
    d = corpus.shape[1]

    # Tamaños hasta el máximo disponible
    grid = [5_000, 10_000, 20_000, 35_000, 49_000, 75_000, 100_000]
    N_LIST = [n for n in grid if n <= len(corpus)]
    if len(corpus) not in N_LIST:
        N_LIST.append(len(corpus))

    # Orden aleatorio fijo -> subconjuntos anidados y reproducibles
    order = np.random.default_rng(cfg.SEED).permutation(len(corpus))

    rows = []
    for N in N_LIST:
        sub = np.ascontiguousarray(corpus[order[:N]])

        # Búsqueda exacta (verdad de referencia a este tamaño)
        flat = faiss.IndexFlatIP(d)
        flat.add(sub)
        exact_ms, exact_ret = latency_ms(flat, queries)

        # HNSW
        hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = EF_C
        hnsw.add(sub)
        hnsw.hnsw.efSearch = cfg.EF_SEARCH
        hnsw_ms, hnsw_ret = latency_ms(hnsw, queries)

        recall = recall_at_k(hnsw_ret, exact_ret, 10)
        speedup = exact_ms / hnsw_ms if hnsw_ms else float("nan")

        row = {
            "N": N,
            "exact_ms": round(exact_ms, 4),
            "hnsw_ms": round(hnsw_ms, 4),
            "speedup": round(speedup, 2),
            "recall@10": round(recall, 4),
        }
        rows.append(row)
        print(f"N={N:6d} | exacta={row['exact_ms']:.3f}ms  HNSW={row['hnsw_ms']:.3f}ms  "
              f"speedup={row['speedup']:.1f}x  R@10={row['recall@10']:.3f}")

    df = pd.DataFrame(rows)
    out = cfg.RESULTS_DIR / "scalability.csv"
    df.to_csv(out, index=False)
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()

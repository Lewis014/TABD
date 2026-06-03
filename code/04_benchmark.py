"""
FASE B.2 — Benchmark HNSW (local, CPU)
=======================================

Recorre las 36 configuraciones (2 modelos x 3 M x 3 ef_construction x 2 métricas).
Para cada una: construye el índice HNSW en FAISS, indexa el corpus, lanza las
1 000 consultas y mide:

  - build_time_s : tiempo de construcción del índice
  - latency_ms   : tiempo medio de consulta (ms)
  - Recall@1, @5, @10 : solape con el ground truth exacto
  - MRR          : rango recíproco medio del vecino más cercano verdadero

Escribe results/results.csv con una fila por configuración.

Recordatorio de cómo FAISS maneja las métricas en HNSW:
  - L2     -> IndexHNSWFlat(d, M, METRIC_L2) sobre vectores crudos
  - coseno -> IndexHNSWFlat(d, M, METRIC_INNER_PRODUCT) sobre vectores normalizados
"""

import time
import numpy as np
import pandas as pd
import faiss

import config as cfg


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def recall_at_k(retrieved: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Promedio sobre consultas de |retrieved[:k] ∩ truth[:k]| / k."""
    hits = 0.0
    for r, t in zip(retrieved, truth):
        hits += len(set(r[:k]) & set(t[:k])) / k
    return hits / len(retrieved)


def mrr(retrieved: np.ndarray, truth: np.ndarray) -> float:
    """Rango recíproco medio del vecino más cercano verdadero (truth[:,0])."""
    total = 0.0
    for r, t in zip(retrieved, truth):
        true_nn = t[0]
        pos = np.where(r == true_nn)[0]
        if len(pos):
            total += 1.0 / (pos[0] + 1)
    return total / len(retrieved)


def build_and_query(corpus, queries, M, ef_c, metric):
    d = corpus.shape[1]
    if metric == "cosine":
        corpus, queries = normalized(corpus), normalized(queries)
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    else:
        corpus = np.ascontiguousarray(corpus, dtype=np.float32)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)

    index.hnsw.efConstruction = ef_c

    t0 = time.perf_counter()
    index.add(corpus)
    build_time = time.perf_counter() - t0

    index.hnsw.efSearch = cfg.EF_SEARCH

    t0 = time.perf_counter()
    _, retrieved = index.search(queries, cfg.K_MAX)
    search_time = time.perf_counter() - t0
    latency_ms = search_time / len(queries) * 1000

    return retrieved, build_time, latency_ms


def main():
    rows = []
    total = len(cfg.MODELS) * len(cfg.M_VALUES) * len(cfg.EF_CONSTRUCTION) * len(cfg.METRICS)
    i = 0

    for model in cfg.MODELS:
        corpus = np.load(cfg.corpus_emb_path(model))
        queries = np.load(cfg.query_emb_path(model))

        for metric in cfg.METRICS:
            truth = np.load(cfg.groundtruth_path(model, metric))

            for M in cfg.M_VALUES:
                for ef_c in cfg.EF_CONSTRUCTION:
                    i += 1
                    retrieved, build_t, lat = build_and_query(
                        corpus, queries, M, ef_c, metric
                    )
                    row = {
                        "model": model,
                        "metric": metric,
                        "M": M,
                        "ef_construction": ef_c,
                        "ef_search": cfg.EF_SEARCH,
                        "build_time_s": round(build_t, 3),
                        "latency_ms": round(lat, 4),
                        "recall@1": round(recall_at_k(retrieved, truth, 1), 4),
                        "recall@5": round(recall_at_k(retrieved, truth, 5), 4),
                        "recall@10": round(recall_at_k(retrieved, truth, 10), 4),
                        "mrr": round(mrr(retrieved, truth), 4),
                    }
                    rows.append(row)
                    print(
                        f"[{i:2d}/{total}] {model:8s} {metric:7s} "
                        f"M={M:<2d} ef_c={ef_c:<3d} | "
                        f"R@10={row['recall@10']:.3f} MRR={row['mrr']:.3f} "
                        f"lat={row['latency_ms']:.3f}ms build={row['build_time_s']:.1f}s"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(cfg.results_csv_path(), index=False)
    print(f"\nResultados guardados en {cfg.results_csv_path()}")


if __name__ == "__main__":
    main()

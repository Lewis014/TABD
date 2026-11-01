"""
FASE B.2 — Benchmark HNSW (local, CPU)
=======================================

Recorre los 18 índices (2 modelos x 3 M x 3 ef_construction) y, sobre cada uno, los
valores de ef_search de la rejilla. Cada índice se construye cfg.N_BUILDS veces con el
orden de inserción barajado, para separar la variación entre construcciones de la
variación entre consultas. Para cada punto de operación mide:

  - build_time_s : tiempo de construcción del índice (cfg.BUILD_THREADS hilos)
  - index_mb     : tamaño del índice serializado
  - latency_ms   : latencia media por consulta, y sus percentiles 50 y 95
  - Recall@1, @5, @10 : solape con el ground truth exacto
  - MRR          : rango recíproco medio del vecino más cercano verdadero

La latencia se mide con 1 hilo y una consulta a la vez, que es lo que vive un usuario;
medir en lote con varios hilos reparte el tiempo entre consultas y lo subestima.

El MRR no aporta información aparte en esta evaluación: si HNSW visita al vecino
verdadero lo devuelve primero, y si no lo visita no aparece, así que coincide con
Recall@1. Se conserva la columna como comprobación.

Coseno = IndexHNSWFlat(d, M, METRIC_INNER_PRODUCT) sobre vectores normalizados.

Salida:
  results/results_builds.csv  una fila por punto de operación y construcción
  results/results.csv         resumen entre construcciones (recall: media y desviación;
                              tiempos: mediana)
  data/clean/perquery_fidelity.npz  aciertos por consulta, para 08_significance.py
"""

import time
import numpy as np
import pandas as pd
import faiss

import config as cfg

N_WARMUP = 50    # consultas que calientan cachés antes de medir
ADD_CHUNK = 50_000


def normalized(x: np.ndarray) -> np.ndarray:
    # Copia: faiss.normalize_L2 opera en el sitio y no debe alterar el array recibido.
    x = np.array(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def per_query_hits(retrieved: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """|retrieved[:k] ∩ truth[:k]| de cada consulta."""
    return np.array([len(set(r[:k]) & set(t[:k])) for r, t in zip(retrieved, truth)],
                    dtype=np.uint8)


def recall_at_k(retrieved: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Promedio sobre consultas de |retrieved[:k] ∩ truth[:k]| / k."""
    return per_query_hits(retrieved, truth, k).mean() / k


def mrr(retrieved: np.ndarray, truth: np.ndarray) -> float:
    """Rango recíproco medio del vecino más cercano verdadero (truth[:,0])."""
    total = 0.0
    for r, t in zip(retrieved, truth):
        true_nn = t[0]
        pos = np.where(r == true_nn)[0]
        if len(pos):
            total += 1.0 / (pos[0] + 1)
    return total / len(retrieved)


def build_index(corpus, M, ef_c, seed):
    """Construye el índice insertando en un orden barajado. Devuelve (índice, orden, segundos)."""
    perm = np.random.default_rng(seed).permutation(len(corpus))
    index = faiss.IndexHNSWFlat(corpus.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_c

    # Por tramos para no duplicar en memoria un corpus grande; hasta ADD_CHUNK es uno solo
    faiss.omp_set_num_threads(cfg.BUILD_THREADS)
    build_time = 0.0
    for chunk in np.array_split(perm, max(1, len(perm) // ADD_CHUNK)):
        shuffled = corpus[chunk]
        t0 = time.perf_counter()
        index.add(shuffled)
        build_time += time.perf_counter() - t0
    return index, perm, build_time


def index_size_mb(index) -> float:
    return faiss.serialize_index(index).size / 2**20


def search_timed(index, queries, k):
    """Busca una consulta a la vez con 1 hilo. Devuelve (ids, latencia por consulta en ms)."""
    faiss.omp_set_num_threads(1)
    ids = np.empty((len(queries), k), dtype=np.int64)
    times = np.empty(len(queries))
    for i in range(len(queries)):
        q = queries[i:i + 1]
        t0 = time.perf_counter()
        _, found = index.search(q, k)
        times[i] = time.perf_counter() - t0
        ids[i] = found[0]
    faiss.omp_set_num_threads(cfg.BUILD_THREADS)
    return ids, times * 1000


def evaluate(index, perm, queries, truth, ef_search):
    """Mide un punto de operación. Devuelve (métricas, aciertos por consulta para cada k)."""
    index.hnsw.efSearch = ef_search
    index.search(queries[:N_WARMUP], cfg.K_MAX)
    found, times = search_timed(index, queries, cfg.K_MAX)
    # Las posiciones del índice se traducen al orden original del corpus
    retrieved = np.where(found >= 0, perm[found], -1)

    hits = {k: per_query_hits(retrieved, truth, k) for k in cfg.K_VALUES}
    metrics = {
        "latency_ms": times.mean(),
        "latency_p50_ms": np.percentile(times, 50),
        "latency_p95_ms": np.percentile(times, 95),
        **{f"recall@{k}": hits[k].mean() / k for k in cfg.K_VALUES},
        "mrr": mrr(retrieved, truth),
    }
    return metrics, hits


def aggregate(raw: pd.DataFrame, keys) -> pd.DataFrame:
    """Resumen entre construcciones de cada punto de operación.

    Los tiempos van por mediana: otro proceso del equipo puede duplicar la latencia de
    una construcción sin que cambie el índice. El recall, que no depende de eso, por media.
    """
    grouped = raw.groupby(keys)
    means = grouped[["index_mb", "recall@1", "recall@5", "recall@10", "mrr"]].mean()
    sds = grouped[["recall@1", "recall@5", "recall@10"]].std(ddof=1).add_suffix("_sd")
    medians = grouped[["build_time_s", "latency_ms", "latency_p50_ms", "latency_p95_ms"]].median()
    iqr = (grouped["latency_ms"].quantile(0.75) - grouped["latency_ms"].quantile(0.25)).rename("latency_ms_iqr")
    return pd.concat([medians, iqr, means, sds], axis=1).reset_index().round(4)


def main():
    keys = ["model", "M", "ef_construction", "ef_search"]
    rows, perquery = [], {}
    total = len(cfg.MODELS) * len(cfg.M_VALUES) * len(cfg.EF_CONSTRUCTION) * cfg.N_BUILDS
    i = 0

    for model in cfg.MODELS:
        corpus = normalized(np.load(cfg.corpus_emb_path(model)))
        queries = normalized(np.load(cfg.query_emb_path(model)))
        truth = np.load(cfg.groundtruth_path(model))

        for M in cfg.M_VALUES:
            for ef_c in cfg.EF_CONSTRUCTION:
                hits_by_ef = {ef_s: {1: [], 10: []} for ef_s in cfg.EF_SEARCH_GRID}
                for build in range(cfg.N_BUILDS):
                    i += 1
                    index, perm, build_t = build_index(corpus, M, ef_c, cfg.SEED + build)
                    size_mb = index_size_mb(index)

                    for ef_s in cfg.EF_SEARCH_GRID:
                        metrics, hits = evaluate(index, perm, queries, truth, ef_s)
                        rows.append({"model": model, "M": M, "ef_construction": ef_c,
                                     "ef_search": ef_s, "build": build,
                                     "build_time_s": build_t, "index_mb": size_mb, **metrics})
                        hits_by_ef[ef_s][1].append(hits[1])
                        hits_by_ef[ef_s][10].append(hits[10])

                    print(f"[{i:2d}/{total}] {model:8s} M={M:<2d} ef_c={ef_c:<3d} build={build} | "
                          f"{build_t:5.1f}s {size_mb:5.0f}MB | efS=50: "
                          f"R@10={rows[-len(cfg.EF_SEARCH_GRID) + 2]['recall@10']:.3f} "
                          f"lat={rows[-len(cfg.EF_SEARCH_GRID) + 2]['latency_ms']:.3f}ms", flush=True)
                    del index

                for ef_s, by_k in hits_by_ef.items():
                    for k, runs in by_k.items():
                        perquery[f"{model}_{M}_{ef_c}_{ef_s}_r{k}"] = np.stack(runs)

                # Se guarda tras cada configuración para no perder la corrida si se interrumpe
                raw = pd.DataFrame(rows)
                raw.round(4).to_csv(cfg.RESULTS_DIR / "results_builds.csv", index=False)
                aggregate(raw, keys).to_csv(cfg.results_csv_path(), index=False)

    np.savez_compressed(cfg.perquery_path(), **perquery)
    print(f"\nResultados guardados en {cfg.results_csv_path()}")


if __name__ == "__main__":
    main()

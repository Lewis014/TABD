"""
FASE B.7 — Escalabilidad: latencia HNSW vs. búsqueda exacta según tamaño (local, CPU)
=====================================================================================

Mide, para subconjuntos ANIDADOS y reproducibles de un pool grande de vectores, la
latencia por consulta de la búsqueda exacta (IndexFlatIP) y de HNSW con M y
ef_construction de la configuración de referencia.

Tres cuidados que la primera versión no tenía:
  - Las consultas se quitan del pool: su proporción variaba con N y falseaba la curva.
  - La latencia principal es de 1 hilo y una consulta a la vez. La de lote con varios
    hilos se mide aparte: ahí la exacta aprovecha BLAS y el tiempo por consulta ya no
    representa lo que espera un usuario. Es lo que hacía parecer que la exacta ganaba
    hasta unas 60 000 filas.
  - Con ef_search fijo el recall cae al crecer N, así que comparar latencias no es
    justo. HNSW se barre en ef_search y se compara también a recall constante.

Cada tamaño se construye N_BUILDS veces (una sola desde LARGE_N, por su costo). Se ajusta
la pendiente log-log de cada curva.

Salida (results/):
  scalability_{modelo}.csv          por N y ef_search (tiempos: mediana; recall: media)
  scalability_targets_{modelo}.csv  por N y objetivo de recall: ef_search necesario,
                                    latencia, ventaja sobre la exacta
  scalability_fit_{modelo}.csv      pendientes log-log con IC 95 %

Uso: python 10_scalability.py --model scibert
"""

import argparse
import time
import numpy as np
import pandas as pd
import faiss
from scipy.stats import linregress, t as t_dist

import config as cfg
from importlib import import_module

_bench = import_module("04_benchmark")
build_index = _bench.build_index
search_timed = _bench.search_timed
normalized = _bench.normalized

N_BUILDS = 3
LARGE_N = 500_000
BATCH_THREADS = 12      # hilos lógicos del i5-11400H, los que usa FAISS por defecto
N_EXACT_QUERIES = 200   # la exacta es estable y lenta: no hacen falta las 1 000
GRID = [5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
EF_GRID = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800]


def load_vectors(model):
    """Vectores del pool y sus metadatos. Con tramos de Colab, reserva una sola vez."""
    folder = cfg.DATA_DIR / f"scale_{model}"
    if folder.exists():
        parts = sorted(folder.glob(f"part_{model}_*.npy"))
        sizes = [np.load(p, mmap_mode="r").shape for p in parts]
        emb = np.empty((sum(s[0] for s in sizes), sizes[0][1]), dtype=np.float32)
        start = 0
        for part, size in zip(parts, sizes):
            emb[start:start + size[0]] = np.load(part)
            start += size[0]
        meta = pd.read_parquet(folder / f"corpus_{model}_scale_meta.parquet").iloc[:len(emb)]
        return emb, meta
    return (np.load(cfg.DATA_DIR / f"corpus_{model}_scale.npy").astype(np.float32),
            pd.read_parquet(cfg.DATA_DIR / "corpus_scale_meta.parquet"))


def load_pool(model):
    """Pool normalizado y barajado, sin las consultas del experimento."""
    emb, meta = load_vectors(model)
    keep = ~meta["id"].astype(str).isin(set(cfg.read_ids(cfg.query_ids_path()))).to_numpy()
    print(f"Pool {model}: {len(emb):,} vectores, {(~keep).sum()} consultas retiradas", flush=True)
    if not keep.all():
        emb = emb[keep]
    faiss.normalize_L2(emb)
    # Barajar en el sitio: los prefijos son subconjuntos anidados sin copiar el pool
    np.random.default_rng(cfg.SEED).shuffle(emb, axis=0)
    return emb


def batch_ms(index, queries):
    """Tiempo por consulta buscando todas juntas con todos los hilos."""
    faiss.omp_set_num_threads(BATCH_THREADS)
    t0 = time.perf_counter()
    index.search(queries, cfg.K_MAX)
    elapsed = time.perf_counter() - t0
    faiss.omp_set_num_threads(cfg.BUILD_THREADS)
    return elapsed / len(queries) * 1000


def slope_row(model, curve, n, latency):
    fit = linregress(np.log(n), np.log(latency))
    half = t_dist.ppf(0.975, len(n) - 2) * fit.stderr if len(n) > 2 else float("nan")
    return {"model": model, "curve": curve, "n_points": len(n), "slope": fit.slope,
            "ci95_low": fit.slope - half, "ci95_high": fit.slope + half, "r2": fit.rvalue ** 2}


def run(model):
    ref = cfg.load_reference()
    corpus = load_pool(model)
    queries = normalized(np.load(cfg.query_emb_path(model)))
    d = corpus.shape[1]

    n_list = [n for n in GRID if n <= len(corpus)]
    if len(corpus) not in n_list:
        n_list.append(len(corpus))

    rows, exact_rows = [], []
    for N in n_list:
        sub = corpus[:N]

        flat = faiss.IndexFlatIP(d)
        flat.add(sub)
        faiss.omp_set_num_threads(cfg.BUILD_THREADS)
        _, truth = flat.search(queries, cfg.K_MAX)   # verdad a este tamaño
        _, times = search_timed(flat, queries[:N_EXACT_QUERIES], cfg.K_MAX)
        exact_rows.append({"N": N, "exact_ms": times.mean(), "exact_p95_ms": np.percentile(times, 95),
                           "exact_batch_ms": batch_ms(flat, queries)})
        del flat

        for build in range(1 if N >= LARGE_N else N_BUILDS):
            index, perm, build_t = build_index(sub, ref["M"], ref["ef_construction"], cfg.SEED + build)
            size_mb = _bench.index_size_mb(index)
            for ef_s in EF_GRID:
                metrics, _ = _bench.evaluate(index, perm, queries, truth, ef_s)
                rows.append({"N": N, "ef_search": ef_s, "build": build, "build_time_s": build_t,
                             "index_mb": size_mb, "latency_ms": metrics["latency_ms"],
                             "latency_p95_ms": metrics["latency_p95_ms"],
                             "latency_batch_ms": batch_ms(index, queries),
                             "recall@10": metrics["recall@10"]})
            del index
        print(f"N={N:>9,} | exacta={times.mean():8.3f} ms | construcción={build_t:7.1f} s "
              f"| {size_mb:7.0f} MB", flush=True)

    raw = pd.DataFrame(rows)
    grouped = raw.groupby(["N", "ef_search"])
    table = pd.concat([
        grouped[["build_time_s", "latency_ms", "latency_p95_ms", "latency_batch_ms"]].median(),
        grouped[["index_mb", "recall@10"]].mean(),
        grouped[["recall@10"]].std(ddof=1).add_suffix("_sd"),
    ], axis=1).reset_index().merge(pd.DataFrame(exact_rows), on="N")
    table.round(4).to_csv(cfg.RESULTS_DIR / f"scalability_{model}.csv", index=False)

    targets = []
    for target in cfg.RECALL_TARGETS:
        for N, group in table.groupby("N"):
            ok = group[group["recall@10"] >= target].sort_values("ef_search")
            if ok.empty:
                targets.append({"N": N, "target": target})
                continue
            r = ok.iloc[0]
            targets.append({"N": N, "target": target, "ef_search": int(r["ef_search"]),
                            "recall@10": r["recall@10"], "hnsw_ms": r["latency_ms"],
                            "hnsw_p95_ms": r["latency_p95_ms"], "exact_ms": r["exact_ms"],
                            "speedup": r["exact_ms"] / r["latency_ms"],
                            "hnsw_batch_ms": r["latency_batch_ms"], "exact_batch_ms": r["exact_batch_ms"],
                            "speedup_batch": r["exact_batch_ms"] / r["latency_batch_ms"],
                            "build_time_s": r["build_time_s"], "index_mb": r["index_mb"]})
    targets = pd.DataFrame(targets)
    targets.round(4).to_csv(cfg.RESULTS_DIR / f"scalability_targets_{model}.csv", index=False)

    exact = pd.DataFrame(exact_rows)
    fits = [slope_row(model, "exact", exact["N"].to_numpy(float), exact["exact_ms"].to_numpy())]
    for target in cfg.RECALL_TARGETS:
        cur = targets[(targets["target"] == target) & targets["hnsw_ms"].notna()]
        if len(cur) == len(exact):
            fits.append(slope_row(model, f"hnsw@{target}", cur["N"].to_numpy(float), cur["hnsw_ms"].to_numpy()))
    fits = pd.DataFrame(fits)
    fits.round(4).to_csv(cfg.RESULTS_DIR / f"scalability_fit_{model}.csv", index=False)

    print("\n" + targets.round(3).to_string(index=False))
    print("\n" + fits.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="scibert", choices=list(cfg.ALL_MODELS))
    run(parser.parse_args().model)

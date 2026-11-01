"""
FASE B.9 — HNSW frente a otras familias de índices (local, CPU)
===============================================================

Pone a HNSW en contexto con las mismas consultas y el mismo ground truth:
  - Exacta        : IndexFlatIP, la referencia sin error.
  - IVF-Flat      : cuantización vectorial con listas invertidas; se barre nprobe.
  - IVF-PQ        : lo mismo con vectores comprimidos por cuantización de producto, con
                    rotación OPQ y distancia L2 sobre vectores unitarios (ordena igual que
                    coseno). Es la configuración justa: con producto interno y sin OPQ,
                    el Recall@10 de PQ no pasa de 0.37 en estos embeddings.
  - IVF-PQ+Refine : IVF-PQ que reordena los candidatos con los vectores originales. Da
                    a PQ una oportunidad justa de llegar a recall alto, a costa de
                    guardar los vectores y perder la ventaja de memoria.
  - BallTree      : árbol exacto de scikit-learn, sobre distancia euclídea (ordena igual
                    que coseno con vectores unitarios). Muestra qué aporta un árbol
                    en esta dimensión.
  - Annoy         : árboles de proyección aleatoria, si la librería está instalada.

Latencia con 1 hilo y una consulta a la vez, como en 04. Como cada familia tiene su
propio parámetro de búsqueda, se comparan curvas recall-latencia y, para cada objetivo
de recall, la opción más rápida que lo alcanza junto con su memoria y construcción.

Salida:
  results/index_families.csv          un punto por familia y parámetro de búsqueda
  results/index_families_targets.csv  la opción más rápida por familia y objetivo
"""

import pickle
import time
import numpy as np
import pandas as pd
import faiss

import config as cfg
from importlib import import_module

_bench = import_module("04_benchmark")
normalized = _bench.normalized

NPROBE_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ANNOY_TREES = [10, 50]
ANNOY_SEARCH_K = [1_000, 5_000, 20_000, 80_000]
N_EXACT_QUERIES = 200
PQ_DIVISOR = 16        # PQ de d/16 subcuantizadores de 8 bits: 32 veces menos que float32
REFINE_FACTOR = 10     # candidatos que reordena Refine: k x REFINE_FACTOR


def point(model, family, params, retrieved, truth, times, build_s, size_mb):
    return {"model": model, "family": family, "params": params,
            **{f"recall@{k}": _bench.recall_at_k(retrieved, truth, k) for k in cfg.K_VALUES},
            "latency_ms": times.mean(), "latency_p95_ms": np.percentile(times, 95),
            "build_time_s": build_s, "index_mb": size_mb}


def run_exact(model, corpus, queries, truth):
    index = faiss.IndexFlatIP(corpus.shape[1])
    t0 = time.perf_counter()
    index.add(corpus)
    build_s = time.perf_counter() - t0
    found, times = _bench.search_timed(index, queries[:N_EXACT_QUERIES], cfg.K_MAX)
    return [point(model, "Exact", "-", found, truth[:N_EXACT_QUERIES], times, build_s,
                  _bench.index_size_mb(index))]


def run_hnsw(model, corpus, queries, truth, ref):
    index, perm, build_s = _bench.build_index(corpus, ref["M"], ref["ef_construction"], cfg.SEED)
    size_mb = _bench.index_size_mb(index)
    rows = []
    for ef_s in cfg.EF_SEARCH_GRID:
        index.hnsw.efSearch = ef_s
        found, times = _bench.search_timed(index, queries, cfg.K_MAX)
        retrieved = np.where(found >= 0, perm[found], -1)
        rows.append(point(model, "HNSW", f"M={ref['M']},efC={ref['ef_construction']},efS={ef_s}",
                          retrieved, truth, times, build_s, size_mb))
    return rows


def sweep_nprobe(model, family, params, index, ivf, queries, truth, build_s):
    size_mb = _bench.index_size_mb(index)
    rows = []
    for nprobe in NPROBE_GRID:
        ivf.nprobe = nprobe
        found, times = _bench.search_timed(index, queries, cfg.K_MAX)
        rows.append(point(model, family, f"{params},nprobe={nprobe}", found, truth, times, build_s, size_mb))
    return rows


def run_ivf_flat(model, corpus, queries, truth, factory):
    index = faiss.index_factory(corpus.shape[1], factory, faiss.METRIC_INNER_PRODUCT)
    faiss.omp_set_num_threads(cfg.BUILD_THREADS)
    t0 = time.perf_counter()
    index.train(corpus)
    index.add(corpus)
    build_s = time.perf_counter() - t0
    return sweep_nprobe(model, "IVF-Flat", factory, index, faiss.extract_index_ivf(index),
                        queries, truth, build_s)


def run_ivf_pq(model, corpus, queries, truth, factory):
    """IVF-PQ con y sin reordenamiento, sobre un único entrenamiento: OPQ tarda minutos."""
    base = faiss.index_factory(corpus.shape[1], factory, faiss.METRIC_L2)
    faiss.omp_set_num_threads(cfg.BUILD_THREADS)
    t0 = time.perf_counter()
    base.train(corpus)
    index = faiss.IndexRefineFlat(base)
    index.k_factor = REFINE_FACTOR
    index.add(corpus)
    build_s = time.perf_counter() - t0
    ivf = faiss.extract_index_ivf(base)
    return (sweep_nprobe(model, "IVF-PQ", factory, base, ivf, queries, truth, build_s) +
            sweep_nprobe(model, "IVF-PQ+Refine", f"{factory},Refine", index, ivf, queries, truth, build_s))


def run_balltree(model, corpus, queries, truth):
    from sklearn.neighbors import BallTree
    t0 = time.perf_counter()
    tree = BallTree(corpus)
    build_s = time.perf_counter() - t0
    size_mb = len(pickle.dumps(tree, protocol=pickle.HIGHEST_PROTOCOL)) / 2**20
    found, times = [], []
    for q in queries[:N_EXACT_QUERIES]:
        t0 = time.perf_counter()
        _, ids = tree.query(q[None, :], k=cfg.K_MAX)
        times.append((time.perf_counter() - t0) * 1000)
        found.append(ids[0])
    return [point(model, "BallTree", "exact", np.array(found), truth[:N_EXACT_QUERIES],
                  np.array(times), build_s, size_mb)]


def run_annoy(model, corpus, queries, truth):
    try:
        from annoy import AnnoyIndex
    except ImportError:
        print("  Annoy no está instalado: se omite.", flush=True)
        return []
    rows = []
    for n_trees in ANNOY_TREES:
        index = AnnoyIndex(corpus.shape[1], "dot")
        index.set_seed(cfg.SEED)
        for i, v in enumerate(corpus):
            index.add_item(i, v)
        t0 = time.perf_counter()
        index.build(n_trees, n_jobs=cfg.BUILD_THREADS)
        build_s = time.perf_counter() - t0
        # Annoy no serializa a memoria: se mide el tamaño del archivo que guarda
        path = cfg.CLEAN_DIR / f"_annoy_{model}_{n_trees}.ann"
        index.save(str(path))
        size_mb = path.stat().st_size / 2**20
        for search_k in ANNOY_SEARCH_K:
            found, times = [], []
            for q in queries:
                t0 = time.perf_counter()
                ids = index.get_nns_by_vector(q, cfg.K_MAX, search_k=search_k)
                times.append((time.perf_counter() - t0) * 1000)
                found.append(ids + [-1] * (cfg.K_MAX - len(ids)))
            rows.append(point(model, "Annoy", f"trees={n_trees},search_k={search_k}", np.array(found),
                              truth, np.array(times), build_s, size_mb))
        index.unload()
        path.unlink()
    return rows


def main():
    ref = cfg.load_reference()
    rows = []
    for model in cfg.MODELS:
        corpus = normalized(np.load(cfg.corpus_emb_path(model)))
        queries = normalized(np.load(cfg.query_emb_path(model)))
        truth = np.load(cfg.groundtruth_path(model))
        d = corpus.shape[1]
        nlist = int(4 * np.sqrt(len(corpus)))
        pq_m = d // PQ_DIVISOR
        pq = f"OPQ{pq_m},IVF{nlist},PQ{pq_m}x8"

        print(f"=== {model} ({pq}) ===", flush=True)
        for name, run in (
            ("Exacta", lambda: run_exact(model, corpus, queries, truth)),
            ("HNSW", lambda: run_hnsw(model, corpus, queries, truth, ref)),
            ("IVF-Flat", lambda: run_ivf_flat(model, corpus, queries, truth, f"IVF{nlist},Flat")),
            ("IVF-PQ y Refine", lambda: run_ivf_pq(model, corpus, queries, truth, pq)),
            ("BallTree", lambda: run_balltree(model, corpus, queries, truth)),
            ("Annoy", lambda: run_annoy(model, corpus, queries, truth)),
        ):
            out = run()
            rows += out
            if out:
                best = max(out, key=lambda r: r["recall@10"])
                print(f"  {name:14s} construcción={best['build_time_s']:6.1f}s {best['index_mb']:6.0f}MB "
                      f"| recall máximo={best['recall@10']:.3f} a {best['latency_ms']:.2f} ms", flush=True)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(cfg.RESULTS_DIR / "index_families.csv", index=False)

    targets = []
    for (model, family), group in df.groupby(["model", "family"], sort=False):
        for target in cfg.RECALL_TARGETS:
            ok = group[group["recall@10"] >= target]
            if ok.empty:
                targets.append({"model": model, "family": family, "target": target,
                                "max_recall@10": group["recall@10"].max()})
            else:
                best = ok.sort_values("latency_ms").iloc[0]
                targets.append({"model": model, "family": family, "target": target,
                                "max_recall@10": group["recall@10"].max(), "params": best["params"],
                                "recall@10": best["recall@10"], "latency_ms": best["latency_ms"],
                                "latency_p95_ms": best["latency_p95_ms"],
                                "build_time_s": best["build_time_s"], "index_mb": best["index_mb"]})
    targets = pd.DataFrame(targets)
    targets.to_csv(cfg.RESULTS_DIR / "index_families_targets.csv", index=False)
    print("\n" + targets.to_string(index=False))


if __name__ == "__main__":
    main()

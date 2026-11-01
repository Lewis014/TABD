"""
FASE B.5 — Consultas en lenguaje natural: known-item title->abstract (local)
============================================================================

La evaluación principal usa abstracts completos como consultas, pero el sistema real
recibe frases cortas en lenguaje natural. Aquí se evalúa ese escenario mediante
*known-item search*:

  - Se toma una muestra reproducible de documentos del corpus.
  - Su TÍTULO (una frase corta, ~10 palabras) es la consulta. El título NO está
    indexado; solo lo están los abstracts.
  - El documento relevante es unívoco: el abstract del mismo artículo.
  - Se mide si ese abstract aparece en el top-K y su rango recíproco.

Cada modelo se evalúa con búsqueda exacta y con HNSW en la configuración de referencia:
la exacta da el límite del modelo, y la diferencia entre ambas es el error del índice.

La búsqueda exacta da el rango completo del abstract relevante, y con él la similitud
título-abstract propio frente a la del documento mejor situado, sobre toda la muestra.

Salida:
  splits/known_item_ids.txt          la muestra, congelada para todos los modelos
  data/clean/title_{modelo}.npy      títulos vectorizados
  data/clean/natural_ranks_{modelo}.npz  rangos por consulta, para 08_significance.py
  results/natural_queries.csv        una fila por modelo y tipo de búsqueda
"""

import numpy as np
import pandas as pd
import faiss
import torch

import config as cfg
from importlib import import_module

_bench = import_module("04_benchmark")
normalized = _bench.normalized

N_SAMPLE = 1000          # nº de consultas title->abstract (muestra reproducible)
SAMPLE_SEED = cfg.SEED + 2
N_BUILDS = 3             # construcciones de HNSW por modelo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sample():
    """Muestra reproducible de documentos del corpus con (fila_embedding, título)."""
    meta = pd.read_parquet(cfg.corpus_meta_path())      # orden == filas de los .npy
    text = pd.read_parquet(cfg.text_path())[["id", "title"]]
    meta["id"] = meta["id"].astype(str)
    text["id"] = text["id"].astype(str)

    id2row = {doc_id: i for i, doc_id in enumerate(meta["id"].tolist())}
    id2title = dict(zip(text["id"], text["title"]))

    if cfg.known_item_ids_path().exists():
        sample_ids = cfg.read_ids(cfg.known_item_ids_path())
    else:
        # documentos con título no vacío
        valid_ids = [i for i in meta["id"].tolist()
                     if isinstance(id2title.get(i), str) and len(id2title[i].strip()) > 0]
        rng = np.random.default_rng(SAMPLE_SEED)
        sample_ids = rng.choice(valid_ids, size=N_SAMPLE, replace=False).tolist()
        cfg.write_ids(cfg.known_item_ids_path(), sample_ids)

    titles = [id2title[i].replace("\n", " ").strip() for i in sample_ids]
    target_rows = np.array([id2row[i] for i in sample_ids])
    return titles, target_rows


def title_embeddings(model_key, titles):
    """Vectoriza los títulos una sola vez por modelo y reutiliza el resultado."""
    path = cfg.title_emb_path(model_key)
    if path.exists():
        return np.load(path)
    from encoders import load_encoder, encode
    model = load_encoder(model_key, DEVICE)
    emb = encode(model_key, model, titles, "short_query", DEVICE)
    np.save(path, emb)
    return emb


def rank_metrics(ranks, n):
    """Rango 1-based del documento relevante; 0 si no aparece. Métricas truncadas en K_MAX."""
    in_top = (ranks > 0) & (ranks <= cfg.K_MAX)
    res = {f"recall@{k}": ((ranks > 0) & (ranks <= k)).mean() for k in cfg.K_VALUES}
    rr = np.zeros(n)
    rr[in_top] = 1.0 / ranks[in_top]
    res["mrr@10"] = rr.mean()
    return res


def exact_ranks(corpus, q, targets):
    """Rango completo del documento relevante y similitudes, por búsqueda exhaustiva."""
    scores = q @ corpus.T
    own = scores[np.arange(len(q)), targets]
    ranks = (scores > own[:, None]).sum(axis=1) + 1
    return ranks, own, scores.max(axis=1)


def hnsw_ranks(corpus, q, targets, ref, seed):
    index, perm, _ = _bench.build_index(corpus, ref["M"], ref["ef_construction"], seed)
    index.hnsw.efSearch = ref["ef_search"]
    _, found = index.search(q, cfg.K_MAX)
    retrieved = np.where(found >= 0, perm[found], -1)
    ranks = np.zeros(len(q), dtype=np.int32)
    for i, row in enumerate(retrieved):
        pos = np.where(row == targets[i])[0]
        if len(pos):
            ranks[i] = pos[0] + 1
    return ranks


def main():
    ref = cfg.load_reference()
    titles, targets = load_sample()
    n = len(titles)
    print(f"Muestra: {n} consultas title->abstract. Dispositivo: {DEVICE}. Ejemplos:")
    for t in titles[:3]:
        print("   -", (t[:80] + "...") if len(t) > 80 else t)

    rows = []
    for model in cfg.available_models():
        print(f"\n=== {model} ===", flush=True)
        q = normalized(title_embeddings(model, titles))
        corpus = normalized(np.load(cfg.corpus_emb_path(model)))

        faiss.omp_set_num_threads(cfg.BUILD_THREADS)
        rank_exact, sim_own, sim_top = exact_ranks(corpus, q, targets)
        rank_hnsw = np.stack([hnsw_ranks(corpus, q, targets, ref, cfg.SEED + b)
                              for b in range(N_BUILDS)])
        np.savez_compressed(cfg.natural_ranks_path(model), exact=rank_exact, hnsw=rank_hnsw)

        exact = rank_metrics(rank_exact, n)
        rows.append({"model": model, "search": "exact", **exact,
                     "median_rank": float(np.median(rank_exact)),
                     "sim_own": sim_own.mean(), "sim_top1": sim_top.mean(),
                     "own_is_top1": (rank_exact == 1).mean()})

        per_build = pd.DataFrame([rank_metrics(r, n) for r in rank_hnsw])
        rows.append({"model": model, "search": "hnsw", **per_build.mean().to_dict(),
                     **per_build.std(ddof=1).add_suffix("_sd").to_dict()})

        print(f"  exacta: R@1={exact['recall@1']:.3f} R@5={exact['recall@5']:.3f} "
              f"R@10={exact['recall@10']:.3f} MRR@10={exact['mrr@10']:.3f} "
              f"| sim propio={sim_own.mean():.3f} top-1={sim_top.mean():.3f} "
              f"rango mediano={np.median(rank_exact):.0f}")
        print(f"  HNSW  : R@10={per_build['recall@10'].mean():.3f} +- {per_build['recall@10'].std(ddof=1):.3f} "
              f"MRR@10={per_build['mrr@10'].mean():.3f}", flush=True)

    df = pd.DataFrame(rows).round(4)
    df.to_csv(cfg.RESULTS_DIR / "natural_queries.csv", index=False)
    print("\nGuardado en results/natural_queries.csv")


if __name__ == "__main__":
    main()

"""
FASE B.1 — Ground truth (local, CPU)
=====================================

Para cada combinación (modelo, métrica) calcula los K vecinos exactos de cada
consulta contra el corpus, por fuerza bruta. Esto es la "verdad" contra la que
luego se mide HNSW.

  - L2     -> faiss.IndexFlatL2  sobre vectores crudos
  - coseno -> faiss.IndexFlatIP  sobre vectores normalizados (= coseno exacto)

Guarda groundtruth_{modelo}_{metrica}.npy con forma (N_QUERIES, K_MAX): cada
fila son los índices (posición en el corpus) de los K vecinos verdaderos.
"""

import time
import numpy as np
import faiss

import config as cfg


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def exact_neighbors(corpus: np.ndarray, queries: np.ndarray, metric: str) -> np.ndarray:
    d = corpus.shape[1]
    if metric == "cosine":
        index = faiss.IndexFlatIP(d)
        index.add(normalized(corpus))
        _, ids = index.search(normalized(queries), cfg.K_MAX)
    else:  # l2
        index = faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(corpus, dtype=np.float32))
        _, ids = index.search(np.ascontiguousarray(queries, dtype=np.float32), cfg.K_MAX)
    return ids


def main():
    for model in cfg.MODELS:
        corpus = np.load(cfg.corpus_emb_path(model))
        queries = np.load(cfg.query_emb_path(model))
        print(f"\n[{model}] corpus {corpus.shape}  queries {queries.shape}")

        for metric in cfg.METRICS:
            t0 = time.perf_counter()
            gt = exact_neighbors(corpus, queries, metric)
            np.save(cfg.groundtruth_path(model, metric), gt)
            dt = time.perf_counter() - t0
            print(f"  {metric:7s}  ground truth {gt.shape}  ({dt:.1f}s)")

    print("\nGround truth completo.")


if __name__ == "__main__":
    main()

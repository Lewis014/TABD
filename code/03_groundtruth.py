"""
FASE B.1 — Ground truth (local, CPU)
=====================================

Para cada modelo calcula los K vecinos exactos de cada consulta contra el corpus, por
fuerza bruta. Esto es la "verdad" contra la que luego se mide HNSW.

faiss.IndexFlatIP sobre vectores normalizados equivale a coseno exacto.

Guarda groundtruth_{modelo}.npy con forma (N_QUERIES, K_MAX): cada fila son los índices
(posición en el corpus) de los K vecinos verdaderos.
"""

import time
import numpy as np
import faiss

import config as cfg


def normalized(x: np.ndarray) -> np.ndarray:
    # Copia: faiss.normalize_L2 opera en el sitio y no debe alterar el array recibido.
    x = np.array(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def exact_neighbors(corpus: np.ndarray, queries: np.ndarray) -> np.ndarray:
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(normalized(corpus))
    _, ids = index.search(normalized(queries), cfg.K_MAX)
    return ids


def main():
    for model in cfg.available_models():
        corpus = np.load(cfg.corpus_emb_path(model))
        queries = np.load(cfg.query_emb_path(model))
        print(f"\n[{model}] corpus {corpus.shape}  queries {queries.shape}")

        t0 = time.perf_counter()
        gt = exact_neighbors(corpus, queries)
        np.save(cfg.groundtruth_path(model), gt)
        print(f"  ground truth {gt.shape}  ({time.perf_counter() - t0:.1f}s)")

    print("\nGround truth completo.")


if __name__ == "__main__":
    main()

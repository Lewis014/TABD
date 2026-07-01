"""
FASE B.5 — Consultas en lenguaje natural: known-item title->abstract (local, CPU)
=================================================================================

Responde a la observación O4 de la revisión: la evaluación principal usa
abstracts completos como consultas (abstract-a-abstract), pero el sistema real
recibe frases cortas en lenguaje natural. Aquí evaluamos el escenario realista
consulta_corta -> abstract mediante *known-item search*:

  - Tomamos una muestra reproducible de documentos del corpus.
  - Usamos su TÍTULO (una frase corta, ~10 palabras, en lenguaje natural) como
    consulta. El título NO está indexado; solo lo están los abstracts.
  - El documento relevante es unívoco: el propio abstract del que salió el título.
  - Medimos si ese abstract aparece en el top-K (Success@K = Recall@K de known-item)
    y su rango recíproco medio (MRR).

Los títulos se vectorizan con EXACTAMENTE el mismo método que los abstracts del
corpus (mean pooling en SciBERT; prefijo "query: " en E5), reutilizando
colab_embed.load_model/encode, para que la comparación sea válida. La búsqueda usa
la mejor configuración del estudio (coseno, M=32, ef_construction=400, ef_search=50).

Salida: results/natural_queries.csv (una fila por modelo) + results/natural_queries.txt
"""

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, models

import config as cfg
from importlib import import_module

_bench = import_module("04_benchmark")
normalized = _bench.normalized


# Réplica EXACTA del método de vectorización de la Fase A (colab_embed.py), sin
# importar ese módulo para evitar su dependencia de `datasets` (solo usada en la
# ingesta de Colab). SciBERT = Transformer + mean pooling; E5 trae su pooling.
def load_model(model_key, device):
    spec = cfg.MODELS[model_key]
    if model_key == "scibert":
        word = models.Transformer(spec["hf"], max_seq_length=cfg.MAX_SEQ_LEN)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[word, pool], device=device)
    model = SentenceTransformer(spec["hf"], device=device)
    model.max_seq_length = cfg.MAX_SEQ_LEN
    return model


def encode(model, texts, prefix, device):
    if prefix:
        texts = [prefix + t for t in texts]
    emb = model.encode(
        texts, batch_size=cfg.BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=False, device=device,
    )
    return emb.astype(np.float32)

CONFIG = {"metric": "cosine", "M": 32, "ef_construction": 400}
N_SAMPLE = 1000          # nº de consultas title->abstract (muestra reproducible)
SAMPLE_SEED = cfg.SEED + 2
DEVICE = "cpu"
K_LIST = [1, 5, 10]


def load_sample():
    """Muestra reproducible de documentos del corpus con (fila_embedding, título)."""
    meta = pd.read_parquet(cfg.corpus_meta_path())      # orden == filas de los .npy
    text = pd.read_parquet(cfg.DATA_DIR / "corpus_text.parquet")[["id", "title"]]

    id2row = {doc_id: i for i, doc_id in enumerate(meta["id"].tolist())}
    id2title = dict(zip(text["id"], text["title"]))

    # documentos con título no vacío
    valid_ids = [i for i in meta["id"].tolist()
                 if isinstance(id2title.get(i), str) and len(id2title[i].strip()) > 0]

    rng = np.random.default_rng(SAMPLE_SEED)
    sample_ids = rng.choice(valid_ids, size=N_SAMPLE, replace=False)

    titles = [id2title[i].replace("\n", " ").strip() for i in sample_ids]
    target_rows = np.array([id2row[i] for i in sample_ids])
    return titles, target_rows


def embed_titles(model_key, titles):
    """Vectoriza los títulos con el mismo método que el corpus (idéntico a Fase A)."""
    spec = cfg.MODELS[model_key]
    model = load_model(model_key, DEVICE)
    prefix = "query: " if spec["prefix"] else None
    return encode(model, titles, prefix, DEVICE)


def evaluate(model_key, titles, target_rows):
    corpus = normalized(np.load(cfg.corpus_emb_path(model_key)))
    d = corpus.shape[1]
    index = faiss.IndexHNSWFlat(d, CONFIG["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = CONFIG["ef_construction"]
    index.add(corpus)
    index.hnsw.efSearch = cfg.EF_SEARCH

    q = normalized(embed_titles(model_key, titles))
    _, retrieved = index.search(q, cfg.K_MAX)   # top-K_MAX filas por consulta

    n = len(titles)
    success = {k: 0 for k in K_LIST}
    rr_sum = 0.0
    for i in range(n):
        row = retrieved[i]
        pos = np.where(row == target_rows[i])[0]
        rank = pos[0] + 1 if len(pos) else None
        if rank is not None:
            rr_sum += 1.0 / rank
            for k in K_LIST:
                if rank <= k:
                    success[k] += 1

    res = {"model": model_key}
    for k in K_LIST:
        res[f"recall@{k}"] = round(success[k] / n, 4)
    res["mrr"] = round(rr_sum / n, 4)
    return res


def main():
    titles, target_rows = load_sample()
    print(f"Muestra: {len(titles)} consultas title->abstract "
          f"(seed={SAMPLE_SEED}). Ejemplos de título-consulta:")
    for t in titles[:3]:
        print("   -", (t[:80] + "...") if len(t) > 80 else t)
    print()

    rows = []
    for m in cfg.MODELS:
        print(f"=== {m} ===")
        res = evaluate(m, titles, target_rows)
        rows.append(res)
        print(f"  R@1={res['recall@1']:.3f}  R@5={res['recall@5']:.3f}  "
              f"R@10={res['recall@10']:.3f}  MRR={res['mrr']:.3f}\n")

    df = pd.DataFrame(rows)
    df.to_csv(cfg.RESULTS_DIR / "natural_queries.csv", index=False)

    lines = ["Known-item title->abstract (O4) — mejor config: coseno, M=32, "
             f"ef_c=400, ef_search={cfg.EF_SEARCH}, n={len(titles)}", ""]
    lines += [df.to_string(index=False)]
    (cfg.RESULTS_DIR / "natural_queries.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Guardado en results/natural_queries.csv y .txt")


if __name__ == "__main__":
    main()

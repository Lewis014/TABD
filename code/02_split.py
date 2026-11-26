"""
FASE B.0 — Separación real de consultas y corpus (local)
========================================================

La primera versión de colab_embed.py dejó 982 de las 1 000 consultas dentro del corpus
indexado: stratified_sample termina en reset_index, de modo que
subset.drop(index=queries.index) borró las filas 0-999 en lugar de las consultas.
Con ellas dentro, el vecino exacto de casi toda consulta era ella misma.

Este script quita esas consultas del corpus usando los embeddings ya generados, sin
volver a vectorizar, y deja en data/clean/ los archivos que usa el resto del pipeline.
Congela además los ids en splits/ (versionados en git), que pasan a definir la muestra.

Salida:
  data/clean/corpus_{modelo}.npy, query_{modelo}.npy, corpus_meta.parquet, query_meta.parquet
  splits/corpus_ids.txt, splits/query_ids.txt
  results/corpus_composition.csv   composición por grupo temático
"""

import numpy as np
import pandas as pd

import config as cfg


def top_group(major) -> str:
    """Primer grupo temático: 'hep-ph hep-ex' -> 'hep-ph'; 'math.CO cs.DM' -> 'math'."""
    tokens = str(major).split()
    return tokens[0].split(".")[0] if tokens else "unknown"


def main():
    corpus_meta = pd.read_parquet(cfg.DATA_DIR / "corpus_meta.parquet")
    query_meta = pd.read_parquet(cfg.DATA_DIR / "query_meta.parquet")
    corpus_meta["id"] = corpus_meta["id"].astype(str)
    query_meta["id"] = query_meta["id"].astype(str)

    keep = ~corpus_meta["id"].isin(set(query_meta["id"])).to_numpy()
    print(f"Corpus original: {len(corpus_meta):,}  |  consultas dentro: {(~keep).sum()}  "
          f"|  corpus limpio: {keep.sum():,}")

    for model in cfg.MODELS:
        corpus = np.load(cfg.DATA_DIR / f"corpus_{model}.npy")
        queries = np.load(cfg.DATA_DIR / f"query_{model}.npy")
        assert len(corpus) == len(corpus_meta) and len(queries) == len(query_meta)
        np.save(cfg.corpus_emb_path(model), corpus[keep])
        np.save(cfg.query_emb_path(model), queries)
        print(f"  {model}: corpus {corpus[keep].shape}  consultas {queries.shape}")

    clean_meta = corpus_meta[keep].reset_index(drop=True)
    clean_meta.to_parquet(cfg.corpus_meta_path())
    query_meta.to_parquet(cfg.query_meta_path())
    assert not set(clean_meta["id"]) & set(query_meta["id"])

    cfg.write_ids(cfg.corpus_ids_path(), clean_meta["id"])
    cfg.write_ids(cfg.query_ids_path(), query_meta["id"])

    composition = pd.concat([
        clean_meta["major"].map(top_group).value_counts(normalize=True).rename("corpus"),
        query_meta["major"].map(top_group).value_counts(normalize=True).rename("queries"),
    ], axis=1).fillna(0).mul(100).round(2).sort_values("corpus", ascending=False)
    composition.index.name = "group"
    composition.to_csv(cfg.RESULTS_DIR / "corpus_composition.csv")
    print("\nComposición por grupo temático (%):")
    print(composition.head(10).to_string())


if __name__ == "__main__":
    main()

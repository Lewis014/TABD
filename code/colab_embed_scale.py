"""
FASE A (escalabilidad) — Embeddings de 100k para el estudio de O9 (Google Colab, GPU)
=====================================================================================

Genera un corpus más grande, SOLO con SciBERT, para el análisis de escalabilidad
(observación O9): latencia de HNSW vs. búsqueda exacta a tamaños crecientes.

No re-hace el experimento principal (49k) ni E5: únicamente produce un pool grande
de vectores SciBERT del que 10_scalability.py toma subconjuntos anidados. Las 1000
consultas del experimento original (query_scibert.npy) se reutilizan tal cual.

Ejecutar en Colab con GPU T4:

    !pip install -q datasets sentence-transformers pandas pyarrow
    # subir config.py y colab_embed.py y colab_embed_scale.py
    !python colab_embed_scale.py
    from google.colab import files; files.download("embeddings_scale.zip")

Descomprimir en E:\\TABD\\code\\data\\ -> deja:
    data/corpus_scibert_scale.npy    (N_SCALE, 768)
    data/corpus_scale_meta.parquet   (id, major)
"""

import shutil
import numpy as np
import torch
from datasets import load_dataset

import config as cfg
from importlib import import_module

_embed = import_module("colab_embed")  # reutiliza muestreo y vectorización

N_SCALE = 100_000   # tamaño del pool para escalabilidad
SCALE_SEED = cfg.SEED + 100


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Dispositivo: {device}")
    if device != "cuda":
        raise SystemExit("ERROR: sin GPU la vectorización de 100k tarda horas. "
                         "Usa Colab con T4.")

    print(f"Descargando {cfg.DATASET} ...")
    ds = load_dataset(cfg.DATASET, split="train")
    keep = [c for c in ds.column_names if c in ("id", "abstract", "categories")]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    df = ds.to_pandas()
    df["major"] = df["categories"].map(_embed.major_category)
    df["abstract"] = df["abstract"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["abstract"].str.len() > 20]
    print(f"Corpus completo: {len(df):,} abstracts")

    subset = _embed.stratified_sample(df[["id", "abstract", "major"]], N_SCALE, SCALE_SEED)
    print(f"Muestra de escalabilidad: {len(subset):,}")

    subset[["id", "major"]].to_parquet(cfg.DATA_DIR / "corpus_scale_meta.parquet")

    model = _embed.load_model("scibert", device)
    emb = _embed.encode(model, subset["abstract"].tolist(), None, device)
    np.save(cfg.DATA_DIR / "corpus_scibert_scale.npy", emb)
    print(f"Guardado corpus_scibert_scale.npy {emb.shape}")

    # Empaqueta solo los dos archivos nuevos
    import zipfile
    with zipfile.ZipFile("embeddings_scale.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(cfg.DATA_DIR / "corpus_scibert_scale.npy", "corpus_scibert_scale.npy")
        z.write(cfg.DATA_DIR / "corpus_scale_meta.parquet", "corpus_scale_meta.parquet")
    print("Listo: embeddings_scale.zip")


if __name__ == "__main__":
    main()

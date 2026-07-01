"""
FASE A — Embeddings (ejecutar en Google Colab con GPU T4)
============================================================

Hace tres cosas:
  1. Descarga arxiv-abstracts-2021 y toma 50 000 abstracts por muestreo
     estratificado por categoría mayor (conserva la mezcla temática original).
  2. Separa 1 000 consultas (holdout estratificado) y deja 49 000 de corpus.
  3. Vectoriza corpus y consultas con SciBERT y E5-large, y guarda los .npy.

E5 exige prefijos: "passage: " para los documentos del corpus y "query: " para
las consultas. SciBERT no usa prefijos. Por eso el holdout se decide ANTES de
vectorizar.

Salida (carpeta ./data):
  corpus_scibert.npy   (49000, 768)      query_scibert.npy   (1000, 768)
  corpus_e5.npy        (49000, 1024)     query_e5.npy        (1000, 1024)
  corpus_meta.parquet  (id, major)       query_meta.parquet  (id, major)

Al final empaqueta todo en embeddings.zip para descargar y copiar a
E:\\TABD\\code\\data\\ en la máquina local.

----------------------------------------------------------------------------
Celdas sugeridas en Colab:

    !pip install -q datasets sentence-transformers pandas pyarrow
    # subir config.py y colab_embed.py, o clonar el repo
    !python colab_embed.py
    from google.colab import files; files.download("embeddings.zip")
----------------------------------------------------------------------------
"""

import shutil
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models

import config as cfg


# ---------------------------------------------------------------------------
# 1. Ingesta + muestreo estratificado
# ---------------------------------------------------------------------------
def major_category(categories) -> str:
    """Reduce la categoría fina de ArXiv a su grupo mayor.

    'math.CO' -> 'math'   ·   'astro-ph.GA' -> 'astro-ph'   ·   'hep-ph' -> 'hep-ph'
    Acepta tanto listas/arrays como cadenas separadas por espacios.
    """
    if categories is None:
        return "unknown"
    if isinstance(categories, (list, tuple, np.ndarray)):
        first = categories[0] if len(categories) else "unknown"
    else:
        first = str(categories).split()[0] if str(categories).strip() else "unknown"
    return first.split(".")[0]


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Muestreo proporcional: cada estrato aporta en proporción a su tamaño."""
    frac = n / len(df)
    out = (
        df.groupby("major", group_keys=False)
          .apply(lambda g: g.sample(n=max(1, round(len(g) * frac)), random_state=seed))
    )
    # El redondeo por estrato no suma exacto: ajustamos a n.
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def build_subset() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Descargando {cfg.DATASET} ...")
    ds = load_dataset(cfg.DATASET, split="train")
    keep = [c for c in ds.column_names if c in ("id", "abstract", "categories")]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    print("Convirtiendo a DataFrame ...")
    df = ds.to_pandas()
    df["major"] = df["categories"].map(major_category)
    df["abstract"] = df["abstract"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["abstract"].str.len() > 20]  # descarta abstracts vacíos/basura

    print(f"Corpus completo: {len(df):,} abstracts en {df['major'].nunique()} grupos")

    subset = stratified_sample(df[["id", "abstract", "major"]], cfg.N_TOTAL, cfg.SEED)
    print(f"Subconjunto muestreado: {len(subset):,}")

    # Holdout estratificado de consultas
    queries = stratified_sample(subset, cfg.N_QUERIES, cfg.SEED + 1)
    corpus = subset.drop(index=queries.index).reset_index(drop=True)
    queries = queries.reset_index(drop=True)

    print(f"  corpus  = {len(corpus):,}")
    print(f"  queries = {len(queries):,}")
    return corpus, queries


# ---------------------------------------------------------------------------
# 2. Vectorización
# ---------------------------------------------------------------------------
def load_model(model_key: str, device: str) -> SentenceTransformer:
    spec = cfg.MODELS[model_key]
    # SciBERT es un BERT puro: lo envolvemos con pooling de media para obtener
    # un único vector por abstract. E5 ya trae su propio pooling.
    if model_key == "scibert":
        word = models.Transformer(spec["hf"], max_seq_length=cfg.MAX_SEQ_LEN)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[word, pool], device=device)
    model = SentenceTransformer(spec["hf"], device=device)
    model.max_seq_length = cfg.MAX_SEQ_LEN
    return model


def encode(model, texts, prefix, device) -> np.ndarray:
    if prefix:
        texts = [prefix + t for t in texts]
    emb = model.encode(
        texts,
        batch_size=cfg.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,   # la normalización para coseno se hace en local
        device=device,                # fuerza el dispositivo detectado (GPU si la hay)
    )
    return emb.astype(np.float32)


def main():
    # --- Verificación de dispositivo: aborta si no hay GPU para no perder horas ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Dispositivo de cómputo: {device}")
    if device == "cuda":
        print(f">>> GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        raise SystemExit(
            "ERROR: no se detectó GPU. En CPU la vectorización tarda horas.\n"
            "En Colab: Entorno de ejecución -> Cambiar tipo de entorno -> T4 GPU, "
            "y reinicia el entorno antes de volver a ejecutar."
        )

    corpus, queries = build_subset()

    corpus[["id", "major"]].to_parquet(cfg.corpus_meta_path())
    queries[["id", "major"]].to_parquet(cfg.query_meta_path())

    for key, spec in cfg.MODELS.items():
        print(f"\n=== Vectorizando con {key} ({spec['hf']}) ===")
        model = load_model(key, device)

        corpus_prefix = "passage: " if spec["prefix"] else None
        query_prefix = "query: " if spec["prefix"] else None

        print("corpus ...")
        c = encode(model, corpus["abstract"].tolist(), corpus_prefix, device)
        np.save(cfg.corpus_emb_path(key), c)

        print("queries ...")
        q = encode(model, queries["abstract"].tolist(), query_prefix, device)
        np.save(cfg.query_emb_path(key), q)

        print(f"  {key}: corpus {c.shape}  query {q.shape}")
        del model
        torch.cuda.empty_cache()

    print("\nEmpaquetando embeddings.zip ...")
    shutil.make_archive("embeddings", "zip", cfg.DATA_DIR)
    print("Listo. Descarga embeddings.zip y descomprímelo en E:\\TABD\\code\\data\\")


if __name__ == "__main__":
    main()

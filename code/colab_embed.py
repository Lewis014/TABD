"""
FASE A — Embeddings (ejecutar en Google Colab con GPU T4)
============================================================

Hace tres cosas:
  1. Descarga arxiv-abstracts-2021 y arma la muestra. Si existen los ids congelados en
     splits/ (los que definen el experimento publicado) usa esos; si no, toma 50 000
     abstracts por muestreo estratificado por categoría mayor.
  2. Separa 1 000 consultas (holdout estratificado) del corpus.
  3. Vectoriza corpus, consultas y títulos con los modelos pedidos, y guarda los .npy.

Los prefijos, el pooling y los adaptadores de cada modelo están en config.py y
encoders.py. El holdout se decide ANTES de vectorizar porque E5 y SPECTER2 tratan
distinto a documentos y consultas.

Salida (carpeta --out), para cada modelo:
  corpus_{modelo}.npy   query_{modelo}.npy   title_{modelo}.npy

Cada archivo se guarda al terminarlo y los que ya existen se saltan, así que tras un
corte basta con volver a ejecutar. En Colab conviene que --out esté en Google Drive.
Al final, los archivos van en E:\\TABD\\code\\data\\clean\\ en la máquina local.

La forma más simple de correrlo es el notebook colab_embeddings.ipynb, que trae todo lo
necesario (lo genera make_colab_notebook.py).

SPECTER2 necesita además la librería adapters, que puede exigir otra versión de
transformers: conviene instalarla justo antes de ese modelo.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

import config as cfg
from encoders import load_encoder, encode


# ---------------------------------------------------------------------------
# 1. Ingesta + muestreo estratificado
# ---------------------------------------------------------------------------
def major_category(categories) -> str:
    """Reduce la categoría fina de ArXiv a su grupo mayor.

    'math.CO cs.DM' -> 'math'   ·   'astro-ph.GA' -> 'astro-ph'   ·   'hep-ph hep-ex' -> 'hep-ph'
    Acepta tanto listas/arrays como cadenas separadas por espacios.
    """
    if categories is None:
        return "unknown"
    if isinstance(categories, (list, tuple, np.ndarray)):
        categories = " ".join(str(c) for c in categories)
    tokens = str(categories).split()
    return tokens[0].split(".")[0] if tokens else "unknown"


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Muestreo proporcional: cada estrato aporta en proporción a su tamaño."""
    frac = n / len(df)
    out = pd.concat([g.sample(n=max(1, round(len(g) * frac)), random_state=seed)
                     for _, g in df.groupby("major")])
    # El redondeo por estrato no suma exacto: ajustamos a n.
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.reset_index(drop=True)


def load_dataframe() -> pd.DataFrame:
    print(f"Descargando {cfg.DATASET} ...")
    ds = load_dataset(cfg.DATASET, split="train")
    keep = [c for c in ds.column_names if c in ("id", "title", "abstract", "categories")]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    print("Convirtiendo a DataFrame ...")
    df = ds.to_pandas()
    df["id"] = df["id"].astype(str)
    df["major"] = df["categories"].map(major_category)
    for col in ("title", "abstract"):
        df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["abstract"].str.len() > 20]  # descarta abstracts vacíos/basura
    print(f"Corpus completo: {len(df):,} abstracts en {df['major'].nunique()} grupos")
    return df


def build_subset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.corpus_ids_path().exists() and cfg.query_ids_path().exists():
        print("Usando los ids congelados en splits/")
        by_id = df.drop_duplicates("id").set_index("id")
        corpus = by_id.loc[cfg.read_ids(cfg.corpus_ids_path())].reset_index()
        queries = by_id.loc[cfg.read_ids(cfg.query_ids_path())].reset_index()
    else:
        subset = stratified_sample(df[["id", "abstract", "major"]], cfg.N_TOTAL, cfg.SEED)
        print(f"Subconjunto muestreado: {len(subset):,}")

        # Holdout estratificado de consultas
        queries = stratified_sample(subset, cfg.N_QUERIES, cfg.SEED + 1)
        corpus = subset[~subset["id"].isin(queries["id"])].reset_index(drop=True)
        corpus[["id", "major"]].to_parquet(cfg.corpus_meta_path())
        queries[["id", "major"]].to_parquet(cfg.query_meta_path())
        cfg.write_ids(cfg.corpus_ids_path(), corpus["id"])
        cfg.write_ids(cfg.query_ids_path(), queries["id"])

    assert not set(corpus["id"]) & set(queries["id"])
    print(f"  corpus  = {len(corpus):,}")
    print(f"  queries = {len(queries):,}")
    return corpus, queries


def load_titles(df: pd.DataFrame) -> list:
    """Títulos de la muestra known-item, en el orden que fijó 09_natural_queries.py."""
    if not cfg.known_item_ids_path().exists():
        return []
    by_id = df.drop_duplicates("id").set_index("id")
    return by_id.loc[cfg.read_ids(cfg.known_item_ids_path()), "title"].tolist()


# ---------------------------------------------------------------------------
# 2. Vectorización
# ---------------------------------------------------------------------------
def save_npy(path: Path, emb: np.ndarray) -> None:
    """Escribe con otro nombre y renombra: un corte no deja un archivo a medias."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, emb)
    os.replace(tmp, path)


def embed_model(key, jobs, device, out):
    """jobs: (nombre de archivo, textos, rol). Carga el modelo solo si falta algo."""
    spec = cfg.ALL_MODELS[key]
    print(f"\n=== {key} ({spec['hf']}) ===", flush=True)
    model = None
    for name, texts, role in jobs:
        path = out / name
        if path.exists():
            print(f"{name}: ya existe, se salta")
            continue
        if model is None:
            model = load_encoder(key, device)
        print(f"{name}: {len(texts):,} textos ...", flush=True)
        emb = encode(key, model, texts, role, device)
        assert emb.shape[1] == spec["dim"], f"{key}: dimensión {emb.shape[1]}, se esperaba {spec['dim']}"
        save_npy(path, emb)
        print(f"  guardado {emb.shape}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(cfg.ALL_MODELS),
                        choices=list(cfg.ALL_MODELS))
    parser.add_argument("--out", type=Path, default=cfg.CLEAN_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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

    df = load_dataframe()
    corpus, queries = build_subset(df)
    titles = load_titles(df)

    for key in args.models:
        jobs = [(cfg.corpus_emb_path(key).name, corpus["abstract"].tolist(), "doc"),
                (cfg.query_emb_path(key).name, queries["abstract"].tolist(), "query")]
        if titles:
            jobs.append((cfg.title_emb_path(key).name, titles, "short_query"))
        embed_model(key, jobs, device, args.out)

    print(f"\nListo. Archivos en {args.out}")


if __name__ == "__main__":
    main()

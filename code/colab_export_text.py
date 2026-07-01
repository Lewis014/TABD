"""
Exporta el TEXTO de los abstracts del corpus, para la demo Streamlit.
Ejecutar en Google Colab (NO necesita GPU).

Por qué: la fase de embeddings solo guardó id + categoría, no el texto. La demo
necesita el abstract (y el título) para mostrar resultados legibles.

Cómo funciona: toma los ids exactos de tu corpus_meta.parquet (los mismos que
están indexados) y recupera su título y abstract del dataset original. Al filtrar
por esos ids, la alineación queda garantizada sin depender de re-muestrear.

Pasos en Colab:
    !pip install -q datasets pandas pyarrow
    # subir: config.py, colab_export_text.py y corpus_meta.parquet (de tu data/)
    !python colab_export_text.py
    from google.colab import files; files.download("corpus_text.parquet")

Luego copia corpus_text.parquet a E:\\TABD\\code\\data\\
"""

import pandas as pd
from datasets import load_dataset
import config as cfg


def main():
    # Ids del corpus ya indexado (subir este archivo a Colab).
    meta = pd.read_parquet("corpus_meta.parquet")
    wanted = set(meta["id"].astype(str))
    print(f"Corpus objetivo: {len(wanted)} ids")

    print(f"Descargando {cfg.DATASET} ...")
    ds = load_dataset(cfg.DATASET, split="train")
    keep = [c for c in ds.column_names if c in ("id", "title", "abstract", "doi")]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    df = ds.to_pandas()
    df["id"] = df["id"].astype(str)

    cols = [c for c in ("id", "title", "abstract", "doi") if c in df.columns]
    text = df[df["id"].isin(wanted)][cols].copy()
    for col in ("title", "abstract"):
        if col in text.columns:
            text[col] = text[col].str.replace(r"\s+", " ", regex=True).str.strip()

    text.to_parquet("corpus_text.parquet", index=False)
    print(f"Guardado corpus_text.parquet: {len(text)}/{len(wanted)} ids encontrados")
    if len(text) < len(wanted):
        print("AVISO: algunos ids no aparecieron en el dataset (se mostrarán sin texto).")


if __name__ == "__main__":
    main()

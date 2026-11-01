"""
FASE A (escalabilidad) — Pool grande de embeddings (Google Colab, GPU)
=====================================================================

Genera un pool grande de vectores del que 10_scalability.py toma subconjuntos anidados.
No re-hace el experimento principal: las 1000 consultas originales se reutilizan tal
cual y se excluyen del pool antes de muestrear.

Una vectorización de horas puede cortarse, así que guarda por tramos en --out (conviene
que esté en Google Drive) y retoma donde quedó al volver a ejecutar. La muestra se baraja
de forma fija: aunque no se llegue a terminar, los tramos hechos son una muestra aleatoria
y sirven igual.

Salida en --out:
    corpus_{modelo}_scale_meta.parquet   (id, major) en el orden de los vectores
    part_{modelo}_{inicio}.npy           tramos de CHUNK vectores

Los archivos van en E:\\TABD\\code\\data\\scale_{modelo}\\ en la máquina local.
"""

import argparse
from pathlib import Path

import torch

import config as cfg
from encoders import load_encoder, encode
from importlib import import_module

_embed = import_module("colab_embed")  # reutiliza la ingesta y el muestreo

SCALE_SEED = cfg.SEED + 100
CHUNK = 50_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="e5", choices=list(cfg.ALL_MODELS))
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, default=cfg.DATA_DIR / "scale_e5")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Dispositivo: {device}")
    if device != "cuda":
        raise SystemExit("ERROR: sin GPU la vectorización del pool tarda demasiado.")

    df = _embed.load_dataframe()
    df = df[~df["id"].isin(set(cfg.read_ids(cfg.query_ids_path())))]
    subset = _embed.stratified_sample(df[["id", "abstract", "major"]], args.n, SCALE_SEED)
    subset = subset.sort_values("id").sample(frac=1, random_state=SCALE_SEED).reset_index(drop=True)
    subset[["id", "major"]].to_parquet(args.out / f"corpus_{args.model}_scale_meta.parquet")
    print(f"Muestra de escalabilidad: {len(subset):,}", flush=True)

    model = None
    texts = subset["abstract"].tolist()
    for start in range(0, len(texts), CHUNK):
        part = args.out / f"part_{args.model}_{start:08d}.npy"
        if part.exists():
            continue
        if model is None:
            model = load_encoder(args.model, device)
        emb = encode(args.model, model, texts[start:start + CHUNK], "doc", device, show_progress=False)
        _embed.save_npy(part, emb)
        print(f"  {min(start + CHUNK, len(texts)):,} / {len(texts):,}", flush=True)

    print(f"Listo. Archivos en {args.out}")


if __name__ == "__main__":
    main()

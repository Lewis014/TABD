"""
FASE A (escalabilidad) — Pool grande de embeddings (Google Colab, GPU)
=====================================================================

Genera un pool grande de vectores del que 10_scalability.py toma subconjuntos anidados.
No re-hace el experimento principal: las 1000 consultas originales se reutilizan tal
cual y se excluyen del pool antes de muestrear.

Una vectorización de horas puede cortarse, así que guarda por tramos en --out (conviene
que esté en Google Drive) y retoma donde quedó al volver a ejecutar. La muestra se baraja
de forma fija: aunque no se llegue a terminar, los tramos hechos son una muestra aleatoria
y sirven igual. --stop-after detiene el trabajo en esa cantidad de vectores sin cambiar la
muestra, por ejemplo para no pasarse del espacio disponible.

Por defecto usa E5-base: es de la misma familia que E5-large y tarda cerca de la mitad.

Salida en --out:
    corpus_{modelo}_scale_meta.parquet   (id, major) en el orden de los vectores
    part_{modelo}_{inicio}.npy           tramos de CHUNK vectores

Los archivos van en E:\\TABD\\code\\data\\scale_{modelo}\\ en la máquina local.
"""

import argparse
import time
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
    parser.add_argument("--model", default="e5_base", choices=list(cfg.ALL_MODELS))
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or cfg.DATA_DIR / f"scale_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    cfg.BATCH_SIZE = args.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Dispositivo: {device}")
    if device != "cuda":
        raise SystemExit("ERROR: sin GPU la vectorización del pool tarda demasiado.")

    df = _embed.load_dataframe()
    df = df[~df["id"].isin(set(cfg.read_ids(cfg.query_ids_path())))]
    subset = _embed.stratified_sample(df[["id", "abstract", "major"]], args.n, SCALE_SEED)
    subset = subset.sort_values("id").sample(frac=1, random_state=SCALE_SEED).reset_index(drop=True)
    subset[["id", "major"]].to_parquet(out / f"corpus_{args.model}_scale_meta.parquet")
    print(f"Muestra de escalabilidad: {len(subset):,}", flush=True)

    model = None
    texts = subset["abstract"].tolist()
    last = min(args.stop_after or len(texts), len(texts))
    for start in range(0, last, CHUNK):
        part = out / f"part_{args.model}_{start:08d}.npy"
        if part.exists():
            continue
        if model is None:
            model = load_encoder(args.model, device)
        t0 = time.perf_counter()
        emb = encode(args.model, model, texts[start:start + CHUNK], "doc", device, show_progress=False)
        _embed.save_npy(part, emb)
        print(f"  {min(start + CHUNK, len(texts)):,} / {last:,}  ({(time.perf_counter() - t0) / 60:.1f} min este tramo)",
              flush=True)

    print(f"Listo: {last:,} vectores en {out}")


if __name__ == "__main__":
    main()

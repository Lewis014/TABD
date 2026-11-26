"""
FASE B.8 — Latencia de extremo a extremo: vectorizar + buscar (local)
=====================================================================

Las tablas de 04 solo miden la búsqueda en el índice. En uso real la consulta primero
hay que vectorizarla, y eso puede costar mucho más que buscarla. Aquí se miden las dos
partes por modelo, para títulos y para abstracts, una consulta a la vez, en CPU y en GPU
si la hay. La búsqueda usa HNSW con la configuración de referencia y 1 hilo.

Salida: results/end_to_end_latency.csv
"""

import numpy as np
import pandas as pd
import torch
import time

import config as cfg
from encoders import load_encoder, encode
from importlib import import_module

_bench = import_module("04_benchmark")
normalized = _bench.normalized

N_TEXTS = 200
N_WARMUP = 5


def load_texts():
    """Títulos y abstracts de consultas del experimento (solo importa su longitud y contenido)."""
    text = pd.read_parquet(cfg.text_path())[["id", "title", "abstract"]].dropna()
    text["id"] = text["id"].astype(str)
    sub = text[text["id"].isin(set(cfg.read_ids(cfg.query_ids_path())))]
    return sub.sample(N_TEXTS, random_state=cfg.SEED)


def timed_encode(model_key, model, texts, role, device):
    """Vectoriza uno a uno. Devuelve (vectores, ms por texto)."""
    encode(model_key, model, texts[:N_WARMUP], role, device, show_progress=False)
    out, times = [], []
    for t in texts:
        t0 = time.perf_counter()
        v = encode(model_key, model, [t], role, device, show_progress=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        out.append(v)
    return np.concatenate(out), np.array(times)


def main():
    ref = cfg.load_reference()
    sample = load_texts()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    rows = []
    for model_key in cfg.available_models():
        corpus = normalized(np.load(cfg.corpus_emb_path(model_key)))
        index, _, _ = _bench.build_index(corpus, ref["M"], ref["ef_construction"], cfg.SEED)
        index.hnsw.efSearch = ref["ef_search"]
        del corpus

        for device in devices:
            try:
                model = load_encoder(model_key, device)
            except Exception as ex:
                print(f"{model_key} no disponible: {type(ex).__name__}")
                continue

            for column, role in (("title", "short_query"), ("abstract", "query")):
                emb, enc = timed_encode(model_key, model, sample[column].tolist(), role, device)
                _, search = _bench.search_timed(index, normalized(emb), cfg.K_MAX)
                total = enc + search
                rows.append({"model": model_key, "device": device, "text": column, "n": N_TEXTS,
                             "cpu_threads": torch.get_num_threads(),
                             "encode_ms": enc.mean(), "encode_p95_ms": np.percentile(enc, 95),
                             "search_ms": search.mean(), "search_p95_ms": np.percentile(search, 95),
                             "total_ms": total.mean(), "total_p95_ms": np.percentile(total, 95),
                             "encode_share": enc.sum() / total.sum()})
                print(f"{model_key:8s} {device:4s} {column:8s} vectorizar={enc.mean():8.2f} ms  "
                      f"buscar={search.mean():6.3f} ms  ({100 * enc.sum() / total.sum():.1f} % vectorizar)",
                      flush=True)
            del model
        del index

    out = cfg.RESULTS_DIR / "end_to_end_latency.csv"
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()

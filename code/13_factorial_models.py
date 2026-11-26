"""
FASE B.10 — Dominio frente a objetivo de entrenamiento (local)
==============================================================

SciBERT frente a E5-large mezcla dos cosas: uno es científico y sin entrenamiento para
recuperación, y el otro es general y entrenado para recuperación. Con los cuatro modelos
adicionales de config.py cada factor se puede ver por separado:

                  sin entrenamiento       con entrenamiento
                  para recuperación       para similitud o recuperación
  científico      SciBERT                 SciNCL, SPECTER2
  general         BERT                    E5-base (768d), E5-large (1024d)

  - SciBERT vs. SciNCL : objetivo de entrenamiento (SciNCL parte de los pesos de SciBERT)
  - SciBERT vs. BERT   : dominio
  - E5-base vs. E5-large: dimensión, misma familia

Los modelos adicionales solo se evalúan con la configuración de referencia, en
abstract->abstract (cfg.N_BUILDS construcciones). Título->abstract lo aporta
09_natural_queries.py. Requiere haber corrido 03 con los modelos adicionales ya
vectorizados, y después 09, para tener las dos partes.

Salida:
  data/clean/perquery_fidelity_extra.npz  aciertos por consulta, para 08_significance.py
  results/results_extra_builds.csv        una fila por modelo y construcción
  results/model_comparison.csv            fidelidad, recuperación y costo por modelo
"""

import numpy as np
import pandas as pd

import config as cfg
from importlib import import_module

_bench = import_module("04_benchmark")
normalized = _bench.normalized


def run_extra(ref):
    keys = ["model", "M", "ef_construction", "ef_search"]
    rows, perquery = [], {}
    tag = f"{ref['M']}_{ref['ef_construction']}_{ref['ef_search']}"
    for model in cfg.available_models():
        if model in cfg.MODELS:
            continue
        corpus = normalized(np.load(cfg.corpus_emb_path(model)))
        queries = normalized(np.load(cfg.query_emb_path(model)))
        truth = np.load(cfg.groundtruth_path(model))
        runs = {1: [], 10: []}
        for build in range(cfg.N_BUILDS):
            index, perm, build_t = _bench.build_index(corpus, ref["M"], ref["ef_construction"],
                                                      cfg.SEED + build)
            size_mb = _bench.index_size_mb(index)
            metrics, hits = _bench.evaluate(index, perm, queries, truth, ref["ef_search"])
            rows.append({"model": model, "M": ref["M"], "ef_construction": ref["ef_construction"],
                         "ef_search": ref["ef_search"], "build": build, "build_time_s": build_t,
                         "index_mb": size_mb, **metrics})
            runs[1].append(hits[1])
            runs[10].append(hits[10])
            del index
            print(f"{model:8s} build={build} {build_t:5.1f}s R@10={metrics['recall@10']:.3f} "
                  f"lat={metrics['latency_ms']:.3f}ms", flush=True)
        for k, hits_k in runs.items():
            perquery[f"{model}_{tag}_r{k}"] = np.stack(hits_k)

    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    raw.round(4).to_csv(cfg.RESULTS_DIR / "results_extra_builds.csv", index=False)
    np.savez_compressed(cfg.perquery_extra_path(), **perquery)
    return _bench.aggregate(raw, keys)


def main():
    ref = cfg.load_reference()
    extra = run_extra(ref)

    grid = pd.read_csv(cfg.results_csv_path())
    grid = grid[(grid["M"] == ref["M"]) & (grid["ef_construction"] == ref["ef_construction"]) &
                (grid["ef_search"] == ref["ef_search"])]
    fidelity = pd.concat([grid, extra], ignore_index=True)

    natural = pd.read_csv(cfg.RESULTS_DIR / "natural_queries.csv")
    exact = natural[natural["search"] == "exact"].set_index("model")
    hnsw = natural[natural["search"] == "hnsw"].set_index("model")

    latency_path = cfg.RESULTS_DIR / "end_to_end_latency.csv"
    encode_ms = {}
    if latency_path.exists():
        lat = pd.read_csv(latency_path)
        lat = lat[(lat["device"] == "cpu") & (lat["text"] == "title")].set_index("model")
        encode_ms = lat["encode_ms"].to_dict()

    rows = []
    for _, r in fidelity.iterrows():
        m = r["model"]
        spec = cfg.ALL_MODELS[m]
        rows.append({
            "model": m, "label": spec["label"], "domain": spec["domain"],
            "retrieval_trained": spec["retrieval_trained"], "dim": spec["dim"],
            "fidelity_r1": r["recall@1"], "fidelity_r10": r["recall@10"],
            "fidelity_r10_sd": r["recall@10_sd"], "search_ms": r["latency_ms"],
            "build_time_s": r["build_time_s"], "index_mb": r["index_mb"],
            "title_r1": exact["recall@1"].get(m), "title_r5": exact["recall@5"].get(m),
            "title_r10": exact["recall@10"].get(m), "title_mrr10": exact["mrr@10"].get(m),
            "title_r10_hnsw": hnsw["recall@10"].get(m), "title_sim_own": exact["sim_own"].get(m),
            "title_median_rank": exact["median_rank"].get(m),
            "encode_title_cpu_ms": encode_ms.get(m),
        })
    df = pd.DataFrame(rows)
    df["order"] = df["model"].map(list(cfg.ALL_MODELS).index)
    df = df.sort_values(["domain", "retrieval_trained", "dim"], ascending=[False, True, True]).drop(columns="order")
    out = cfg.RESULTS_DIR / "model_comparison.csv"
    df.round(4).to_csv(out, index=False)
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()

"""
Verificación del hallazgo de O4 (title->abstract). Separa "bug" de "efecto real".

Chequeos:
 1. Re-embeber localmente el ABSTRACT de un doc y buscar: debe recuperarse a sí
    mismo en rango 1 con similitud ~1,0. Valida alineación fila<->id y que el
    embedding local reproduce el de Colab.
 2. Similitud coseno media entre el TÍTULO y su propio abstract, por modelo.
 3. Ejemplos concretos: para 5 títulos, qué recupera cada modelo en el top-1.
"""

import numpy as np
import pandas as pd
import faiss
from importlib import import_module

import config as cfg

nq = import_module("09_natural_queries")   # reutiliza load_model/encode/normalized
load_model, encode, normalized = nq.load_model, nq.encode, nq.normalized

N = 30
DEVICE = "cpu"


def main():
    meta = pd.read_parquet(cfg.corpus_meta_path())
    text = pd.read_parquet(cfg.DATA_DIR / "corpus_text.parquet")[["id", "title", "abstract"]]
    id2row = {d: i for i, d in enumerate(meta["id"].tolist())}
    tmap = {r.id: (r.title, r.abstract) for r in text.itertuples()}

    ids = meta["id"].tolist()
    valid = [i for i in ids if isinstance(tmap.get(i, (None,))[0], str) and tmap[i][0].strip()]
    rng = np.random.default_rng(cfg.SEED + 7)
    sample = list(rng.choice(valid, size=N, replace=False))
    rows = np.array([id2row[i] for i in sample])
    titles = [tmap[i][0].replace("\n", " ").strip() for i in sample]
    abstracts = [tmap[i][1].replace("\n", " ").strip() for i in sample]

    for model_key in cfg.MODELS:
        spec = cfg.MODELS[model_key]
        print("=" * 66)
        print(f"MODELO: {model_key}")
        corpus = normalized(np.load(cfg.corpus_emb_path(model_key)))
        d = corpus.shape[1]
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 400
        index.add(corpus)
        index.hnsw.efSearch = cfg.EF_SEARCH

        model = load_model(model_key, DEVICE)
        qpref = "query: " if spec["prefix"] else None

        # (1) abstract re-embebido -> ¿se recupera a sí mismo?
        emb_abs = normalized(encode(model, abstracts, qpref, DEVICE))
        _, ret_abs = index.search(emb_abs, 10)
        self_rank1 = sum(ret_abs[k][0] == rows[k] for k in range(N))
        self_sim = float(np.mean([corpus[rows[k]] @ emb_abs[k] for k in range(N)]))

        # (2) título -> similitud con su propio abstract y recuperación
        emb_tit = normalized(encode(model, titles, qpref, DEVICE))
        _, ret_tit = index.search(emb_tit, 10)
        title_self_sim = float(np.mean([corpus[rows[k]] @ emb_tit[k] for k in range(N)]))
        tit_rank1 = sum(ret_tit[k][0] == rows[k] for k in range(N))
        tit_top10 = sum(rows[k] in ret_tit[k][:10] for k in range(N))

        print(f"(1) abstract->self  : rango1={self_rank1}/{N}  sim(abs,self)={self_sim:.3f}")
        print(f"(2) titulo ->self   : rango1={tit_rank1}/{N}  top10={tit_top10}/{N}  "
              f"sim(titulo,abstract_propio)={title_self_sim:.3f}")
        print()

    # (3) ejemplos concretos con SciBERT y E5
    print("=" * 66)
    print("EJEMPLOS (título-consulta -> top-1 recuperado por cada modelo)")
    id_by_row = {v: k for k, v in id2row.items()}
    for model_key in cfg.MODELS:
        spec = cfg.MODELS[model_key]
        corpus = normalized(np.load(cfg.corpus_emb_path(model_key)))
        index = faiss.IndexHNSWFlat(corpus.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 400
        index.add(corpus); index.hnsw.efSearch = cfg.EF_SEARCH
        model = load_model(model_key, DEVICE)
        qpref = "query: " if spec["prefix"] else None
        emb = normalized(encode(model, titles[:5], qpref, DEVICE))
        _, ret = index.search(emb, 1)
        print(f"\n--- {model_key} ---")
        for k in range(5):
            got_row = ret[k][0]
            got_id = id_by_row.get(int(got_row))
            got_title = tmap.get(got_id, ("?",))[0]
            ok = "OK" if got_row == rows[k] else "X"
            print(f"[{ok}] consulta: {titles[k][:60]}")
            print(f"      top-1 : {str(got_title)[:60]}")


if __name__ == "__main__":
    main()

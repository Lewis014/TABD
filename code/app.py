"""
Demo Streamlit — Búsqueda semántica de abstracts científicos con HNSW.

Reutiliza los embeddings ya generados (E5-large) y construye el índice HNSW con M y
ef_construction de la configuración de referencia. Se usa E5-large ---y no SciBERT---
porque recupera mucho mejor ante consultas cortas en lenguaje natural, que es el caso de
uso de esta demo (véase la evaluación known-item del paper). El usuario escribe una
consulta, se vectoriza con E5-large y se recuperan los abstracts más parecidos. La
interfaz muestra por separado lo que tarda cada paso.

Ejecutar:
    pip install -r requirements_app.txt
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

import config as cfg

MODEL = "e5"
REF = cfg.load_reference()

st.set_page_config(page_title="HNSW semantic search", layout="centered")


# ---------------------------------------------------------------------------
# Carga de recursos (una sola vez por sesión)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading E5-large...")
def load_model() -> SentenceTransformer:
    model = SentenceTransformer(cfg.MODELS[MODEL]["hf"], device="cpu")
    model.max_seq_length = cfg.MAX_SEQ_LEN
    return model


@st.cache_resource(show_spinner="Building the HNSW index (once)...")
def load_index() -> faiss.Index:
    emb = np.load(cfg.corpus_emb_path(MODEL)).astype(np.float32)
    faiss.normalize_L2(emb)  # coseno = producto interno sobre vectores normalizados
    index = faiss.IndexHNSWFlat(emb.shape[1], REF["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = REF["ef_construction"]
    index.add(emb)
    return index


@st.cache_data(show_spinner="Loading texts...")
def load_docs() -> pd.DataFrame:
    # corpus_meta conserva el orden de los .npy; le pegamos el texto por id.
    meta = pd.read_parquet(cfg.corpus_meta_path())
    text = pd.read_parquet(cfg.text_path())
    meta["id"] = meta["id"].astype(str)
    text["id"] = text["id"].astype(str)
    return meta.merge(text, on="id", how="left")


def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
    # E5 exige el prefijo "query: " en las consultas (los documentos del corpus se
    # vectorizaron con "passage: "); omitirlo degrada la calidad.
    v = model.encode(["query: " + q], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(v)
    return v


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
model = load_model()
index = load_index()
docs = load_docs()

st.title("Semantic search of scientific abstracts with HNSW")
st.caption(
    f"{len(docs):,} arXiv abstracts · E5-large (1024d) · FAISS-HNSW · cosine similarity · "
    f"M={REF['M']}, ef\\_construction={REF['ef_construction']}"
)

# ?q=... permite abrir la demo con una consulta ya escrita
st.session_state.setdefault("query", st.query_params.get("q", ""))

# Consultas sugeridas (rellenan la caja al pulsarlas)
suggested = [
    "quantum field theory",
    "machine learning for particle physics",
    "gravitational waves detection",
]
st.write("Suggested queries:")
cols = st.columns(len(suggested))
for c, s in zip(cols, suggested):
    if c.button(s, use_container_width=True):
        st.session_state.query = s
        st.rerun()

query = st.text_input("Type your query:", key="query")

c1, c2 = st.columns(2)
k = c1.slider("k (number of results)", 1, 20, 8)
ef = c2.slider("ef\\_search (search breadth)", 10, 800, REF["ef_search"], step=10)

# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------
if query.strip():
    t0 = time.perf_counter()
    qv = embed_query(model, query)
    encode_ms = (time.perf_counter() - t0) * 1000

    index.hnsw.efSearch = ef
    t0 = time.perf_counter()
    scores, ids = index.search(qv, k)
    search_ms = (time.perf_counter() - t0) * 1000

    m1, m2, m3 = st.columns(3)
    m1.metric("Query encoding", f"{encode_ms:.1f} ms")
    m2.metric("Index search", f"{search_ms:.2f} ms")
    m3.metric("Results", k)

    st.subheader(f"Top {k} results by cosine similarity")
    for rank, (i, sc) in enumerate(zip(ids[0], scores[0]), start=1):
        if i < 0:
            continue
        row = docs.iloc[int(i)]
        title = row.get("title") if isinstance(row.get("title"), str) else "(untitled)"
        cat = row.get("major") if isinstance(row.get("major"), str) else ""
        abstract = row.get("abstract") if isinstance(row.get("abstract"), str) else ""
        snippet = abstract[:400] + ("…" if len(abstract) > 400 else "")

        # Enlaces: arXiv siempre (a partir del id); DOI solo si existe.
        arxiv_id = str(row.get("id", "")).strip()
        links = []
        if arxiv_id:
            links.append(f"[arXiv](https://arxiv.org/abs/{arxiv_id})")
        doi = row.get("doi")
        if isinstance(doi, str) and doi.strip():
            d = doi.strip()
            links.append(f"DOI: [{d}](https://doi.org/{d})")

        st.markdown(f"**{rank}. {title}**")
        meta_line = f"{cat}  ·  similarity {sc:.3f}"
        if links:
            meta_line += "  ·  " + "  ".join(links)
        st.caption(meta_line)
        if snippet:
            st.write(snippet)
        st.divider()
else:
    st.info("Type a query or pick a suggested one to start.")

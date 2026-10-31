"""
Demo Streamlit — Búsqueda semántica de abstracts científicos con HNSW.

Reutiliza los embeddings ya generados (E5-large) y construye el índice HNSW con la
configuración coseno, M=32, ef_construction=400. Se usa E5-large ---y no SciBERT---
porque, aunque SciBERT es más fiel al índice exacto, E5-large recupera mucho mejor
ante consultas cortas en lenguaje natural, que es el caso de uso de esta demo
(véase la evaluación known-item en la Sección 5.2 del paper). El usuario escribe una
consulta, se vectoriza con E5-large y se recuperan los abstracts más parecidos.

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

# Modelo de recuperación para el despliegue: E5-large, entrenado para el
# emparejamiento consulta-documento (Sección 5.2 del paper). Config: coseno, M=32.
BEST = {"model": "e5", "metric": "cosine", "M": 32, "ef_construction": 400}

st.set_page_config(page_title="Búsqueda semántica HNSW", layout="centered")


# ---------------------------------------------------------------------------
# Carga de recursos (una sola vez por sesión)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo E5-large...")
def load_model() -> SentenceTransformer:
    spec = cfg.MODELS["e5"]
    model = SentenceTransformer(spec["hf"], device="cpu")
    model.max_seq_length = cfg.MAX_SEQ_LEN
    return model


@st.cache_resource(show_spinner="Construyendo índice HNSW (una vez)...")
def load_index() -> faiss.Index:
    emb = np.load(cfg.corpus_emb_path("e5")).astype(np.float32)
    faiss.normalize_L2(emb)  # coseno = producto interno sobre vectores normalizados
    index = faiss.IndexHNSWFlat(emb.shape[1], BEST["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = BEST["ef_construction"]
    index.add(emb)
    return index


@st.cache_data(show_spinner="Cargando textos...")
def load_docs() -> pd.DataFrame:
    # corpus_meta conserva el orden de los .npy; le pegamos el texto por id.
    meta = pd.read_parquet(cfg.corpus_meta_path())
    text = pd.read_parquet(cfg.DATA_DIR / "corpus_text.parquet")
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

st.title("Búsqueda semántica de abstracts científicos mediante indexación HNSW")
st.caption(
    f"Demo sobre {len(docs):,} abstracts de arXiv · E5-large (1024d) · "
    "FAISS-HNSW · similitud coseno · M=32, ef\\_c=400"
)

st.session_state.setdefault("query", "")

# Consultas sugeridas (rellenan la caja al pulsarlas)
sugeridas = [
    "quantum field theory",
    "machine learning for particle physics",
    "gravitational waves detection",
]
st.write("Consultas sugeridas:")
cols = st.columns(len(sugeridas))
for c, s in zip(cols, sugeridas):
    if c.button(s, use_container_width=True):
        st.session_state.query = s
        st.rerun()

query = st.text_input("Escribe tu consulta:", key="query")

c1, c2 = st.columns(2)
k = c1.slider("k (número de resultados)", 1, 20, 8)
ef = c2.slider("ef\\_search (amplitud de búsqueda)", 10, 10000, 50, step=10)

# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------
if query.strip():
    qv = embed_query(model, query)
    index.hnsw.efSearch = ef
    t0 = time.perf_counter()
    scores, ids = index.search(qv, k)
    latency_ms = (time.perf_counter() - t0) * 1000

    m1, m2 = st.columns(2)
    m1.metric("Latencia de búsqueda", f"{latency_ms:.1f} ms")
    m2.metric("Resultados", k)

    st.subheader(f"{k} resultados ordenados por similitud coseno")
    for rank, (i, sc) in enumerate(zip(ids[0], scores[0]), start=1):
        if i < 0:
            continue
        row = docs.iloc[int(i)]
        title = row.get("title") if isinstance(row.get("title"), str) else "(sin título)"
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
        meta_line = f"{cat}  ·  similitud {sc:.3f}"
        if links:
            meta_line += "  ·  " + "  ".join(links)
        st.caption(meta_line)
        if snippet:
            st.write(snippet)
        st.divider()
else:
    st.info("Escribe una consulta o pulsa una de las sugeridas para empezar.")

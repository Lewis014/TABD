"""
Demo Streamlit — Búsqueda semántica de abstracts científicos con HNSW.

Reutiliza los embeddings ya generados (SciBERT) y construye el índice HNSW con la
configuración ÓPTIMA hallada en el experimento: coseno, M=32, ef_construction=400.
El usuario escribe una consulta, se vectoriza con SciBERT y se recuperan los
abstracts más parecidos en milisegundos.

Ejecutar:
    pip install -r requirements_app.txt
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer, models

import config as cfg

# Configuración ganadora del experimento (Sección 5 del paper)
BEST = {"model": "scibert", "metric": "cosine", "M": 32, "ef_construction": 400}

st.set_page_config(page_title="Búsqueda semántica HNSW", layout="centered")


# ---------------------------------------------------------------------------
# Carga de recursos (una sola vez por sesión)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo SciBERT...")
def load_model() -> SentenceTransformer:
    spec = cfg.MODELS["scibert"]
    word = models.Transformer(spec["hf"], max_seq_length=cfg.MAX_SEQ_LEN)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[word, pool], device="cpu")


@st.cache_resource(show_spinner="Construyendo índice HNSW (una vez)...")
def load_index() -> faiss.Index:
    emb = np.load(cfg.corpus_emb_path("scibert")).astype(np.float32)
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
    v = model.encode([q], convert_to_numpy=True).astype(np.float32)
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
    f"Demo sobre {len(docs):,} abstracts de arXiv · SciBERT (768d) · "
    "FAISS-HNSW · similitud coseno · configuración óptima (M=32, ef\\_c=400)"
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

"""
Configuración compartida del experimento.

La importan tanto el script de Colab (embeddings) como los scripts locales
(ground truth, benchmark, exportación). Mantener un único punto de verdad evita
que las dos mitades del pipeline se desincronicen.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # aquí llegan los .npy originales que produce Colab
CLEAN_DIR = DATA_DIR / "clean"    # corpus sin las consultas (lo genera 02_split.py)
SPLITS_DIR = ROOT / "splits"      # ids congelados de corpus, consultas y títulos (en git)
RESULTS_DIR = ROOT / "results"    # aquí se escriben CSV y tablas LaTeX
for _dir in (DATA_DIR, CLEAN_DIR, SPLITS_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset y muestreo
# ---------------------------------------------------------------------------
DATASET = "gfissore/arxiv-abstracts-2021"
N_TOTAL = 50_000          # tamaño del subconjunto muestreado
N_QUERIES = 1_000         # consultas (holdout)
SEED = 42

# ---------------------------------------------------------------------------
# Modelos de embedding
#   pooling  -> el modelo es un BERT sin pooling propio; se añade este
#   prefixes -> prefijo por rol: doc (corpus), query (abstract como consulta)
#               y short_query (título como consulta)
#   adapters -> SPECTER2 usa un adaptador distinto según el rol
# ---------------------------------------------------------------------------
_E5_PREFIXES = {"doc": "passage: ", "query": "query: ", "short_query": "query: "}

# Modelos de la rejilla completa de hiperparámetros
MODELS = {
    "scibert": {
        "hf": "allenai/scibert_scivocab_uncased",
        "dim": 768,
        "pooling": "mean",
        "label": "SciBERT",
        "domain": "scientific",
        "retrieval_trained": False,
    },
    "e5": {
        "hf": "intfloat/e5-large-v2",
        "dim": 1024,
        "prefixes": _E5_PREFIXES,
        "label": "E5-large",
        "domain": "general",
        "retrieval_trained": True,
    },
}

# Modelos que solo se evalúan con la configuración de referencia, para separar el
# dominio del objetivo de entrenamiento (13_factorial_models.py)
EXTRA_MODELS = {
    "scincl": {
        "hf": "malteos/scincl",
        "dim": 768,
        "pooling": "cls",
        "label": "SciNCL",
        "domain": "scientific",
        "retrieval_trained": True,
    },
    "specter2": {
        "hf": "allenai/specter2_base",
        "dim": 768,
        "adapters": {
            "doc": "allenai/specter2",
            "query": "allenai/specter2",
            "short_query": "allenai/specter2_adhoc_query",
        },
        "label": "SPECTER2",
        "domain": "scientific",
        "retrieval_trained": True,
    },
    "bert": {
        "hf": "bert-base-uncased",
        "dim": 768,
        "pooling": "mean",
        "label": "BERT",
        "domain": "general",
        "retrieval_trained": False,
    },
    "e5_base": {
        "hf": "intfloat/e5-base-v2",
        "dim": 768,
        "prefixes": _E5_PREFIXES,
        "label": "E5-base",
        "domain": "general",
        "retrieval_trained": True,
    },
}

ALL_MODELS = {**MODELS, **EXTRA_MODELS}

MAX_SEQ_LEN = 512         # longitud máxima de tokens al vectorizar
BATCH_SIZE = 32           # ajustar según memoria de la GPU (T4: 32 es seguro)

# ---------------------------------------------------------------------------
# Rejilla de hiperparámetros HNSW
#   índices: 2 modelos x 3 M x 3 ef_construction = 18
#   puntos de operación: 18 índices x 6 ef_search = 108
# La similitud es siempre coseno (producto interno sobre vectores unitarios): con
# vectores unitarios, L2 ordena exactamente igual porque ||a-b||^2 = 2 - 2 a·b.
# ---------------------------------------------------------------------------
M_VALUES = [8, 16, 32]
EF_CONSTRUCTION = [100, 200, 400]
EF_SEARCH_GRID = [10, 20, 50, 100, 200, 400]

N_BUILDS = 5              # construcciones por configuración (orden de inserción barajado)
BUILD_THREADS = 6         # hilos al construir (núcleos físicos del i5-11400H)

# ---------------------------------------------------------------------------
# Regla de selección de la configuración de referencia (05_export.py)
#   Entre los puntos de operación que alcanzan REFERENCE_TARGET en todos los modelos
#   de la rejilla, la de menor latencia. El objetivo es un valor convencional, fijado
#   antes de correr la rejilla completa.
# ---------------------------------------------------------------------------
RECALL_TARGETS = (0.95, 0.99)
REFERENCE_TARGET = 0.99

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
K_VALUES = [1, 5, 10]
K_MAX = max(K_VALUES)     # profundidad del ground truth


# ---------------------------------------------------------------------------
# Utilidades de nombres de archivo (para que todos los scripts coincidan)
# ---------------------------------------------------------------------------
def corpus_emb_path(model: str) -> Path:
    return CLEAN_DIR / f"corpus_{model}.npy"

def query_emb_path(model: str) -> Path:
    return CLEAN_DIR / f"query_{model}.npy"

def title_emb_path(model: str) -> Path:
    return CLEAN_DIR / f"title_{model}.npy"

def corpus_meta_path() -> Path:
    return CLEAN_DIR / "corpus_meta.parquet"

def query_meta_path() -> Path:
    return CLEAN_DIR / "query_meta.parquet"

def groundtruth_path(model: str) -> Path:
    return CLEAN_DIR / f"groundtruth_{model}.npy"

def perquery_path() -> Path:
    return CLEAN_DIR / "perquery_fidelity.npz"

def perquery_extra_path() -> Path:
    return CLEAN_DIR / "perquery_fidelity_extra.npz"

def natural_ranks_path(model: str) -> Path:
    return CLEAN_DIR / f"natural_ranks_{model}.npz"

def text_path() -> Path:
    return DATA_DIR / "corpus_text.parquet"

def corpus_ids_path() -> Path:
    return SPLITS_DIR / "corpus_ids.txt"

def query_ids_path() -> Path:
    return SPLITS_DIR / "query_ids.txt"

def known_item_ids_path() -> Path:
    return SPLITS_DIR / "known_item_ids.txt"

def results_csv_path() -> Path:
    return RESULTS_DIR / "results.csv"

def reference_path() -> Path:
    return RESULTS_DIR / "reference_config.json"


def read_ids(path: Path) -> list:
    return path.read_text(encoding="utf-8").split()

def write_ids(path: Path, ids) -> None:
    path.write_text("\n".join(str(i) for i in ids) + "\n", encoding="utf-8")

def load_reference() -> dict:
    """Configuración de referencia elegida por 05_export.py: M, ef_construction, ef_search."""
    return json.loads(reference_path().read_text(encoding="utf-8"))

def available_models() -> list:
    """Modelos con embeddings de corpus ya generados."""
    return [m for m in ALL_MODELS if corpus_emb_path(m).exists()]

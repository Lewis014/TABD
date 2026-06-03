"""
Configuración compartida del experimento.

La importan tanto el script de Colab (embeddings) como los scripts locales
(ground truth, benchmark, exportación). Mantener un único punto de verdad evita
que las dos mitades del pipeline se desincronicen.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # aquí llegan los .npy que produce Colab
RESULTS_DIR = ROOT / "results"    # aquí se escriben CSV y tablas LaTeX
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset y muestreo
# ---------------------------------------------------------------------------
DATASET = "gfissore/arxiv-abstracts-2021"
N_TOTAL = 50_000          # tamaño del subconjunto muestreado
N_QUERIES = 1_000         # consultas (holdout)
N_CORPUS = N_TOTAL - N_QUERIES   # 49_000 documentos indexados
SEED = 42

# ---------------------------------------------------------------------------
# Modelos de embedding
#   prefix=True  -> el modelo E5 exige prefijos "query:" / "passage:"
# ---------------------------------------------------------------------------
MODELS = {
    "scibert": {
        "hf": "allenai/scibert_scivocab_uncased",
        "dim": 768,
        "prefix": False,
    },
    "e5": {
        "hf": "intfloat/e5-large-v2",
        "dim": 1024,
        "prefix": True,
    },
}

MAX_SEQ_LEN = 512         # longitud máxima de tokens al vectorizar
BATCH_SIZE = 32           # ajustar según memoria de la GPU (T4: 32 es seguro)

# ---------------------------------------------------------------------------
# Rejilla de hiperparámetros HNSW  ->  2 modelos x 3 M x 3 ef_c x 2 metricas = 36
# ---------------------------------------------------------------------------
M_VALUES = [8, 16, 32]
EF_CONSTRUCTION = [100, 200, 400]
METRICS = ["cosine", "l2"]

EF_SEARCH = 50            # amplitud de búsqueda en consulta (fija, no parte de la rejilla)

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
K_VALUES = [1, 5, 10]
K_MAX = max(K_VALUES)     # profundidad del ground truth


# ---------------------------------------------------------------------------
# Utilidades de nombres de archivo (para que todos los scripts coincidan)
# ---------------------------------------------------------------------------
def corpus_emb_path(model: str) -> Path:
    return DATA_DIR / f"corpus_{model}.npy"

def query_emb_path(model: str) -> Path:
    return DATA_DIR / f"query_{model}.npy"

def corpus_meta_path() -> Path:
    return DATA_DIR / "corpus_meta.parquet"

def query_meta_path() -> Path:
    return DATA_DIR / "query_meta.parquet"

def groundtruth_path(model: str, metric: str) -> Path:
    return DATA_DIR / f"groundtruth_{model}_{metric}.npy"

def results_csv_path() -> Path:
    return RESULTS_DIR / "results.csv"

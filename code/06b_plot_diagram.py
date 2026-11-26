"""
Diagrama de bloques del sistema (Fig. 1 del artículo).

Lo dibuja con matplotlib para que quede versionado y se actualice junto con el código.
Los modelos se toman de los que realmente tienen embeddings generados.

Salida: ../Images/diagrama_bloques.png

Ejecutar:
    python 06b_plot_diagram.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import config as cfg
import plot_style as ps

FILL = {"data": "#e8f1fc", "embed": "#fdeee8", "index": "#e6f6f0", "eval": "#fdf5e0"}
HEIGHT = 1.3
ROWS = [8.7, 6.8, 4.9, 3.0, 1.1]   # base de cada fila, de arriba abajo

names = [cfg.ALL_MODELS[m]["label"] for m in cfg.available_models()]
models = "\n".join(" · ".join(names[i:i + 3]) for i in range(0, len(names), 3))


def box(ax, x, y, w, title, body, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, HEIGHT, boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor=fill, edgecolor=ps.AXIS, linewidth=0.8))
    ax.text(x + w / 2, y + HEIGHT - 0.14, title, ha="center", va="top", fontsize=9,
            fontweight="bold", color=ps.INK)
    ax.text(x + w / 2, y + HEIGHT - 0.5, body, ha="center", va="top", fontsize=8,
            color=ps.INK_SOFT, linespacing=1.25)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=ps.INK_SOFT, linewidth=0.9, shrinkA=0, shrinkB=0))


def stage(ax, row, text):
    ax.text(0.0, ROWS[row] + HEIGHT + 0.14, text, ha="left", va="bottom", fontsize=8,
            fontstyle="italic", color=ps.INK_SOFT)


fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.set_xlim(-0.05, 10.05)
ax.set_ylim(ROWS[-1] - 0.05, ROWS[0] + HEIGHT + 0.5)
ax.axis("off")

stage(ax, 0, "1. Data ingestion")
box(ax, 2.0, ROWS[0], 6.0, "arXiv abstracts dataset",
    "Stratified sample with frozen ids\n48,018 indexed abstracts · 1,000 held-out queries", FILL["data"])

stage(ax, 1, "2. Embedding generation (GPU)")
box(ax, 2.0, ROWS[1], 6.0, "Embedding models", models, FILL["embed"])
arrow(ax, (5.0, ROWS[0]), (5.0, ROWS[1] + HEIGHT))

stage(ax, 2, "3. Indexing (CPU)")
box(ax, 0.0, ROWS[2], 4.7, "Exact search (ground truth)",
    "FAISS IndexFlatIP\nCosine over unit vectors", FILL["index"])
box(ax, 5.3, ROWS[2], 4.7, "HNSW index",
    "FAISS IndexHNSWFlat\n$M$, ef_construction, repeated builds", FILL["index"])
arrow(ax, (3.5, ROWS[1]), (2.35, ROWS[2] + HEIGHT))
arrow(ax, (6.5, ROWS[1]), (7.65, ROWS[2] + HEIGHT))

stage(ax, 3, "4. Evaluation")
box(ax, 0.0, ROWS[3], 4.7, "Abstract queries",
    "HNSW compared with exact search\nIndex fidelity: Recall@K", FILL["eval"])
box(ax, 5.3, ROWS[3], 4.7, "Title queries",
    "Target: abstract of the same paper\nKnown-item retrieval: Recall@K, MRR", FILL["eval"])
for start, end in ((2.35, 2.35), (7.65, 7.65), (3.6, 6.4), (6.4, 3.6)):
    arrow(ax, (start, ROWS[2]), (end, ROWS[3] + HEIGHT))

box(ax, 1.5, ROWS[4], 7.0, "Trade-off analysis",
    "Recall · latency (query encoding + search, ef_search)\nBuild time · index memory · paired statistical tests",
    FILL["eval"])
arrow(ax, (2.35, ROWS[3]), (3.4, ROWS[4] + HEIGHT))
arrow(ax, (7.65, ROWS[3]), (6.6, ROWS[4] + HEIGHT))

ps.save(fig, "diagrama_bloques.png")

"""
Genera las figuras analíticas del artículo a partir de results.csv.

Salida (carpeta ../Images):
  fig_recall_vs_m.png       Recall@10 frente a M (efecto del factor dominante)
  fig_recall_latency.png    Compromiso Recall@10 vs latencia (las 36 configuraciones)

Ejecutar:
    python 06_plots.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results" / "results.csv"
IMAGES = Path(__file__).parent.parent / "Images"
IMAGES.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS)

# Etiquetas y estilo por (modelo, métrica)
combos = [
    ("scibert", "cosine", "SciBERT · coseno", "o", "-"),
    ("scibert", "l2", "SciBERT · L2", "s", "--"),
    ("e5", "cosine", "E5-large · coseno", "^", "-"),
    ("e5", "l2", "E5-large · L2", "D", "--"),
]
COLOR = {"scibert": "#1f77b4", "e5": "#d62728"}


# ---------------------------------------------------------------------------
# Figura 1: Recall@10 frente a M (con ef_construction = 400)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
sub = df[df["ef_construction"] == 400]
for model, metric, label, marker, ls in combos:
    d = sub[(sub["model"] == model) & (sub["metric"] == metric)].sort_values("M")
    ax.plot(d["M"], d["recall@10"], marker=marker, linestyle=ls,
            color=COLOR[model], label=label, linewidth=1.8, markersize=7)

ax.set_xlabel("M (conexiones por nodo)")
ax.set_ylabel("Recall@10")
ax.set_xticks([8, 16, 32])
ax.set_ylim(0.94, 1.001)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Recall@10 según M (ef_construction = 400)", fontsize=10)
fig.tight_layout()
fig.savefig(IMAGES / "fig_recall_vs_m.png", dpi=200)
print("Guardado:", IMAGES / "fig_recall_vs_m.png")


# ---------------------------------------------------------------------------
# Figura 2: Compromiso Recall@10 vs latencia (las 36 configuraciones)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
for model, metric, label, marker, ls in combos:
    d = df[(df["model"] == model) & (df["metric"] == metric)]
    ax.scatter(d["latency_ms"], d["recall@10"], marker=marker,
               color=COLOR[model], label=label, s=55, alpha=0.8,
               edgecolors="white", linewidths=0.6)

ax.set_xlabel("Latencia por consulta (ms)")
ax.set_ylabel("Recall@10")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Compromiso exactitud–latencia (36 configuraciones)", fontsize=10)
fig.tight_layout()
fig.savefig(IMAGES / "fig_recall_latency.png", dpi=200)
print("Guardado:", IMAGES / "fig_recall_latency.png")

"""
Figura de O4: la inversión entre escenarios.

Barras agrupadas de Recall@10 para SciBERT y E5-large en los dos escenarios:
abstract->abstract (fidelidad del índice) y título->abstract (consulta corta
real). Muestra de un vistazo que la ventaja de SciBERT se invierte cuando la
consulta es una frase corta en lenguaje natural.

Lee results/natural_queries.csv (título->abstract) y results/results.csv
(abstract->abstract, mejor config coseno M=32 ef_c=400). Salida:
../Images/fig_natural_vs_fidelity.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RES = Path(__file__).parent / "results"
IMAGES = Path(__file__).parent.parent / "Images"
IMAGES.mkdir(exist_ok=True)

# Título->abstract (known-item)
nat = pd.read_csv(RES / "natural_queries.csv").set_index("model")["recall@10"]

# Abstract->abstract: mejor config de cada modelo (coseno, M=32, ef_c=400)
full = pd.read_csv(RES / "results.csv")
def abs_r10(model):
    row = full[(full.model == model) & (full.metric == "cosine") &
               (full.M == 32) & (full.ef_construction == 400)]
    return float(row["recall@10"].iloc[0])

models = ["scibert", "e5"]
labels = {"scibert": "SciBERT", "e5": "E5-large"}
COLOR = {"scibert": "#1f77b4", "e5": "#d62728"}

scenarios = ["Abstract→abstract\n(fidelidad del índice)",
             "Título→abstract\n(consulta corta real)"]
data = {
    "scibert": [abs_r10("scibert"), nat["scibert"]],
    "e5":      [abs_r10("e5"),      nat["e5"]],
}

x = np.arange(len(scenarios))
w = 0.35
fig, ax = plt.subplots(figsize=(6.2, 4))
for i, m in enumerate(models):
    bars = ax.bar(x + (i - 0.5) * w, data[m], w, label=labels[m],
                  color=COLOR[m], edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Recall@10")
ax.set_ylim(0, 1.08)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
ax.set_title("La ventaja de SciBERT se invierte con consultas reales", fontsize=10)
fig.tight_layout()
fig.savefig(IMAGES / "fig_natural_vs_fidelity.png", dpi=200)
print("Guardado:", IMAGES / "fig_natural_vs_fidelity.png")

"""
Figura de fidelidad frente a recuperación, por modelo.

Barras agrupadas de Recall@10 en los dos escenarios: abstract->abstract (cuánto se parece
HNSW a la búsqueda exacta) y título->abstract (si encuentra el abstract del artículo a
partir de una frase corta). Los modelos se agrupan por dominio y entrenamiento.

Lee results/model_comparison.csv. Salida: ../Images/fig_natural_vs_fidelity.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
import plot_style as ps

ps.apply()
df = pd.read_csv(cfg.RESULTS_DIR / "model_comparison.csv")

x = np.arange(len(df))
w = 0.38
fig, ax = plt.subplots(figsize=(max(5.2, 1.5 + 1.0 * len(df)), 3.2))
bars_f = ax.bar(x - w / 2 - 0.01, df["fidelity_r10"], w, color=ps.SERIES[0],
                label="Abstract→abstract (index fidelity)")
bars_t = ax.bar(x + w / 2 + 0.01, df["title_r10_hnsw"], w, color=ps.SERIES[1],
                label="Title→abstract (known-item retrieval)")
ax.bar_label(bars_t, fmt="%.3f", fontsize=7, padding=2, color=ps.INK_SOFT)

ax.set_xticks(x)
ax.set_xticklabels(df["label"])
ax.set_ylabel("Recall@10")
ax.set_ylim(0, 1.0)
ax.grid(False, axis="x")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ps.save(fig, "fig_natural_vs_fidelity.png")

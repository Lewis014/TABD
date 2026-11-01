"""
Figura de HNSW frente a otras familias de índices.

Lee results/index_families.csv y produce ../Images/fig_index_families.png: curvas
Recall@10-latencia por familia, un panel por modelo. La búsqueda exacta es una línea
vertical, porque tiene un solo punto y recall 1. Cada curva sigue el orden de su
parámetro de búsqueda.

Ejecutar:
    python 12b_plot_families.py
"""

import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
import plot_style as ps

ps.apply()
df = pd.read_csv(cfg.RESULTS_DIR / "index_families.csv")

STYLE = {"HNSW": (ps.SERIES[0], "o"), "IVF-Flat": (ps.SERIES[1], "s"),
         "IVF-PQ": (ps.SERIES[2], "^"), "IVF-PQ+Refine": (ps.SERIES[3], "D"),
         "BallTree": (ps.SERIES[4], "v"), "Annoy": (ps.SERIES[5], "P")}

fig, axes = plt.subplots(1, len(cfg.MODELS), figsize=(7.2, 3.2), sharey=True)
for ax, model in zip(axes, cfg.MODELS):
    d = df[df["model"] == model]
    for family, (color, marker) in STYLE.items():
        for i, (_, g) in enumerate(d[d["family"] == family].groupby("build_time_s", sort=False)):
            ax.plot(g["latency_ms"], g["recall@10"], marker=marker, color=color,
                    label=family if i == 0 else None)

    exact = d[d["family"] == "Exact"]["latency_ms"].iloc[0]
    ax.axvline(exact, color=ps.INK_SOFT, linewidth=0.8)
    ax.annotate("Exact", (exact, 0.05), xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=8, color=ps.INK_SOFT)
    ax.set_xscale("log")
    ps.plain_ticks(ax.xaxis, [0.1, 1, 10, 100])
    ax.set_xlabel("Latency per query (ms, one thread, log scale)")
    ax.set_title(cfg.MODELS[model]["label"], fontsize=9)

axes[0].set_ylabel("Recall@10")
axes[0].set_ylim(0, 1.03)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels))
ps.save(fig, "fig_index_families.png")

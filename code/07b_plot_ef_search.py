"""
Figura de sensibilidad de ef_search.

Lee results/ef_search_sensitivity.csv y produce ../Images/fig_ef_search.png: dos paneles
con un solo eje cada uno, Recall@10 y latencia por consulta frente a ef_search, para los
dos modelos y cada M. El color es el modelo y el marcador es M.

Ejecutar:
    python 07b_plot_ef_search.py
"""

import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
import plot_style as ps

ps.apply()
df = pd.read_csv(cfg.RESULTS_DIR / "ef_search_sensitivity.csv")
ref = cfg.load_reference()

M_STYLE = {8: (":", "v"), 16: ("--", "s"), 32: ("-", "o")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
for model in cfg.MODELS:
    for M, (ls, marker) in M_STYLE.items():
        d = df[(df["model"] == model) & (df["M"] == M)].sort_values("ef_search")
        label = f"{cfg.MODELS[model]['label']}, $M$={M}"
        for ax, col in ((ax1, "recall@10"), (ax2, "latency_ms")):
            ax.plot(d["ef_search"], d[col], linestyle=ls, marker=marker,
                    color=ps.MODEL_COLOR[model], label=label)

for ax in (ax1, ax2):
    ax.set_xscale("log")
    ax.set_xticks(cfg.EF_SEARCH_GRID)
    ax.set_xticklabels(cfg.EF_SEARCH_GRID)
    ax.minorticks_off()
    ax.set_xlabel(r"$\mathit{ef\_search}$")

ax1.set_ylabel("Recall@10")
ax1.axhline(cfg.REFERENCE_TARGET, color=ps.AXIS, linewidth=0.8)
ax1.annotate(f"target {cfg.REFERENCE_TARGET}", (10, cfg.REFERENCE_TARGET), xytext=(0, -9),
             textcoords="offset points", fontsize=7, color=ps.INK_SOFT)
ax2.set_yscale("log")
ps.plain_ticks(ax2.yaxis, [0.1, 0.2, 0.5, 1, 2])
ax2.set_ylabel("Latency per query (ms, log scale)")
ax2.legend(fontsize=7, loc="upper left")
ps.save(fig, "fig_ef_search.png")

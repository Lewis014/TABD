"""
Genera las figuras analíticas del artículo a partir de results.csv.

Salida (carpeta ../Images):
  fig_recall_vs_m.png       Recall@10 frente a M (efecto del factor dominante)
  fig_recall_latency.png    Compromiso Recall@10 vs latencia (los 108 puntos de operación)

Ejecutar:
    python 06_plots.py
"""

import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
import plot_style as ps

ps.apply()
df = pd.read_csv(cfg.results_csv_path())
ref = cfg.load_reference()


# ---------------------------------------------------------------------------
# Recall@10 frente a M, en el ef_construction y ef_search de referencia
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
sub = df[(df["ef_construction"] == ref["ef_construction"]) & (df["ef_search"] == ref["ef_search"])]
for model in cfg.MODELS:
    d = sub[sub["model"] == model].sort_values("M")
    ax.errorbar(d["M"], d["recall@10"], yerr=d["recall@10_sd"], marker=ps.MODEL_MARKER[model],
                color=ps.MODEL_COLOR[model], label=cfg.MODELS[model]["label"], capsize=2)
    last = d.iloc[-1]
    ax.annotate(cfg.MODELS[model]["label"], (last["M"], last["recall@10"]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8)

ax.set_xlabel("$M$ (connections per node)")
ax.set_ylabel("Recall@10")
ax.set_xscale("log", base=2)
ax.set_xticks(cfg.M_VALUES)
ax.set_xticklabels(cfg.M_VALUES)
ax.set_xlim(7, 48)
ax.legend(loc="lower right")
ps.save(fig, "fig_recall_vs_m.png")


# ---------------------------------------------------------------------------
# Compromiso Recall@10 vs latencia (todos los puntos de operación)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for model in cfg.MODELS:
    d = df[df["model"] == model]
    ax.scatter(d["latency_ms"], d["recall@10"], marker=ps.MODEL_MARKER[model], s=26,
               color=ps.MODEL_COLOR[model], label=cfg.MODELS[model]["label"],
               edgecolors="white", linewidths=0.6)

    r = d[(d["M"] == ref["M"]) & (d["ef_construction"] == ref["ef_construction"]) &
          (d["ef_search"] == ref["ef_search"])].iloc[0]
    ax.scatter(r["latency_ms"], r["recall@10"], s=90, facecolors="none",
               edgecolors=ps.INK, linewidths=1.0, zorder=3)

ax.annotate(f"Reference: $M$={ref['M']}, ef_c={ref['ef_construction']}, ef_s={ref['ef_search']}",
            xy=(0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=8, color=ps.INK_SOFT)
ax.set_xscale("log")
ps.plain_ticks(ax.xaxis, [0.1, 0.2, 0.5, 1, 2])
ax.set_xlabel("Latency per query (ms, one thread, log scale)")
ax.set_ylabel("Recall@10")
ax.legend(loc="center right")
ps.save(fig, "fig_recall_latency.png")

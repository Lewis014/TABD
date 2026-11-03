"""
Figura de escalabilidad: latencia por consulta vs. tamaño del corpus.

Lee results/scalability_targets_{modelo}.csv y produce ../Images/fig_scalability.png, en
escala log-log, un panel por modelo con pool de escalabilidad: búsqueda exacta y HNSW a
recall constante, de modo que la pendiente de cada recta se pueda leer directamente.

Ejecutar:
    python 10b_plot_scalability.py
"""

import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
import plot_style as ps

ps.apply()
models = [m for m in cfg.ALL_MODELS
          if (cfg.RESULTS_DIR / f"scalability_targets_{m}.csv").exists()]

frames = {m: pd.read_csv(cfg.RESULTS_DIR / f"scalability_targets_{m}.csv") for m in models}
# El eje y es compartido: los límites salen de todos los paneles, no del último
y_low = min(min(df["hnsw_ms"].min(), df["exact_ms"].min()) for df in frames.values()) * 0.8
y_high = max(max(df["hnsw_ms"].max(), df["exact_ms"].max()) for df in frames.values()) * 1.25

fig, axes = plt.subplots(1, len(models), figsize=(1.2 + 4.0 * len(models), 3.2),
                         sharey=True, squeeze=False)
for ax, model in zip(axes[0], models):
    df = frames[model]

    exact = df.dropna(subset=["exact_ms"]).drop_duplicates("N").sort_values("N")
    ax.plot(exact["N"], exact["exact_ms"], marker="s", color=ps.SERIES[1], label="Exact search")
    for target, color, marker in zip(cfg.RECALL_TARGETS, (ps.SERIES[0], ps.SERIES[2]), ("o", "^")):
        d = df[(df["target"] == target) & df["hnsw_ms"].notna()].sort_values("N")
        ax.plot(d["N"], d["hnsw_ms"], marker=marker, color=color,
                label=f"HNSW at Recall@10 ≥ {target}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ps.plain_ticks(ax.yaxis, [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])
    ax.set_ylim(y_low, y_high)
    ax.set_xticks(exact["N"])
    ax.set_xticklabels([f"{n / 1000:.0f}k" for n in exact["N"]])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Corpus size $N$ (log scale)")
    if len(models) > 1:
        ax.set_title(cfg.ALL_MODELS[model]["label"], fontsize=9)

axes[0][0].set_ylabel("Latency per query (ms, log scale)")
axes[0][0].legend(loc="upper left")
ps.save(fig, "fig_scalability.png")

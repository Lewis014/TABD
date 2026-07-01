"""
Figura de escalabilidad (O9): latencia por consulta vs. tamaño del corpus.

Lee results/scalability.csv y produce ../Images/fig_scalability.png: dos curvas
(búsqueda exacta y HNSW) frente a N. La exacta crece de forma aproximadamente
lineal (O(N·d)); HNSW se mantiene casi plana (O(log N)). Se anota el speedup en
el tamaño máximo.

Ejecutar:
    python 10b_plot_scalability.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path(__file__).parent / "results" / "scalability.csv"
IMAGES = Path(__file__).parent.parent / "Images"
IMAGES.mkdir(exist_ok=True)

df = pd.read_csv(CSV).sort_values("N")

fig, ax = plt.subplots(figsize=(6.2, 4))
ax.plot(df["N"], df["exact_ms"], marker="o", color="#d62728",
        linewidth=1.8, markersize=7, label="Búsqueda exacta (IndexFlatIP)")
ax.plot(df["N"], df["hnsw_ms"], marker="s", color="#1f77b4",
        linewidth=1.8, markersize=7, label="HNSW (M=32, ef\\_search=50)")

ax.set_xlabel("Tamaño del corpus (N documentos)")
ax.set_ylabel("Latencia por consulta (ms)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Escalabilidad: latencia vs. tamaño del corpus", fontsize=10)

# Anotación del speedup en el N máximo
last = df.iloc[-1]
ax.annotate(f"{last['speedup']:.0f}$\\times$ más rápido\\nque la exacta en N={int(last['N']):,}",
            xy=(last["N"], last["hnsw_ms"]),
            xytext=(last["N"] * 0.55, max(df["exact_ms"]) * 0.5),
            fontsize=8, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.7))

fig.tight_layout()
fig.savefig(IMAGES / "fig_scalability.png", dpi=200)
print("Guardado:", IMAGES / "fig_scalability.png")

"""
Figura de sensibilidad de ef_search (observación O1).

Lee results/ef_search_sensitivity.csv y produce ../Images/fig_ef_search.png:
un eje doble con Recall@10 (izquierda) y latencia por consulta (derecha) frente
a ef_search, sobre la mejor configuración (SciBERT, coseno, M=32, ef_c=400).
Se marca ef_search=50 —el valor usado en el estudio— para mostrar que cae en el
"codo" de la curva: por debajo el recall cae, por encima solo crece la latencia.

Ejecutar:
    python 07b_plot_ef_search.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path(__file__).parent / "results" / "ef_search_sensitivity.csv"
IMAGES = Path(__file__).parent.parent / "Images"
IMAGES.mkdir(exist_ok=True)

df = pd.read_csv(CSV).sort_values("ef_search")

C_RECALL = "#1f77b4"   # mismo azul de SciBERT en las otras figuras
C_LAT = "#d62728"      # rojo para latencia

fig, ax1 = plt.subplots(figsize=(6, 4))

# Eje izquierdo: Recall@10
ax1.plot(df["ef_search"], df["recall@10"], marker="o", color=C_RECALL,
         linewidth=1.8, markersize=7, label="Recall@10")
ax1.set_xlabel("ef_search (amplitud de búsqueda en consulta)")
ax1.set_ylabel("Recall@10", color=C_RECALL)
ax1.tick_params(axis="y", labelcolor=C_RECALL)
ax1.set_ylim(0.955, 1.002)
ax1.set_xscale("log")
ax1.set_xticks(df["ef_search"])
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.grid(True, alpha=0.3)

# Eje derecho: latencia
ax2 = ax1.twinx()
ax2.plot(df["ef_search"], df["latency_ms"], marker="s", color=C_LAT,
         linewidth=1.8, markersize=7, linestyle="--", label="Latencia")
ax2.set_ylabel("Latencia por consulta (ms)", color=C_LAT)
ax2.tick_params(axis="y", labelcolor=C_LAT)

# Marca del valor usado en el estudio (ef_search = 50)
ax1.axvline(50, color="gray", linestyle=":", linewidth=1.2)
ax1.annotate("ef_search = 50\n(valor del estudio)", xy=(50, 0.972),
             xytext=(70, 0.965), fontsize=8, color="gray")

ax1.set_title("Sensibilidad de ef_search — SciBERT · coseno · M=32 · ef_c=400",
              fontsize=9)
fig.tight_layout()
fig.savefig(IMAGES / "fig_ef_search.png", dpi=200)
print("Guardado:", IMAGES / "fig_ef_search.png")

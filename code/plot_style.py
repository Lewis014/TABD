"""
Estilo común de las figuras del artículo.

El color sigue a la entidad: cada modelo conserva su color en todas las figuras. Como
el artículo puede leerse impreso, el color nunca va solo: se acompaña de marcador,
estilo de línea o etiqueta directa.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker

import config as cfg

IMAGES = Path(__file__).parent.parent / "Images"
IMAGES.mkdir(exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MODEL_COLOR = dict(zip(cfg.ALL_MODELS, SERIES))
MODEL_MARKER = dict(zip(cfg.ALL_MODELS, ["o", "s", "^", "D", "v", "P"]))


def apply():
    mpl.rcParams.update({
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "lines.markeredgecolor": "white",
        "lines.markeredgewidth": 0.8,
    })


def plain_ticks(axis, values):
    """Marcas con números corrientes en un eje logarítmico: 0.1, 1, 10 en vez de potencias."""
    axis.set_ticks(values)
    axis.set_ticklabels([f"{v:g}" for v in values])
    axis.set_minor_locator(mpl.ticker.NullLocator())


def save(fig, name):
    fig.tight_layout()
    path = IMAGES / name
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    print("Guardado:", path)

"""
Genera los notebooks de Colab, cada uno con el código y los ids que necesita, para no
tener que subir nada más:
  colab_embeddings.ipynb   vectoriza los modelos adicionales
  colab_scale.ipynb        pool grande para la escalabilidad

Ejecutar de nuevo si cambian config.py, encoders.py, los scripts de Colab o splits/:
    python make_colab_notebook.py
"""

import base64
import io
import json
import textwrap
import zipfile

import config as cfg

FILES = ["config.py", "encoders.py", "colab_embed.py", "colab_embed_scale.py",
         "splits/corpus_ids.txt", "splits/query_ids.txt", "splits/known_item_ids.txt"]


def bundle() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in FILES:
            z.write(cfg.ROOT / name, name)
    return base64.b64encode(buffer.getvalue()).decode()


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(True)}


SETUP = """
#@title 1. Preparar el entorno
import base64, glob, io, os, zipfile
import torch
from google.colab import drive

assert torch.cuda.is_available(), "No hay GPU: elige GPU T4 en Entorno de ejecución y vuelve a ejecutar."
print("GPU:", torch.cuda.get_device_name(0))

drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/{folder}"
os.makedirs(OUT, exist_ok=True)

BUNDLE = \"\"\"
{bundle}
\"\"\"
os.makedirs("/content/tabd", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall("/content/tabd")
os.chdir("/content/tabd")
print("Archivos listos en", os.getcwd())
"""

INSTALL = """
#@title 2. Instalar dependencias
!pip install -q datasets sentence-transformers pandas pyarrow
"""

# ---------------------------------------------------------------------------
# Modelos adicionales
# ---------------------------------------------------------------------------
MODELS_INTRO = """
# Embeddings de los modelos adicionales

**Cómo usarlo**

1. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4 → Guardar**.
2. Menú **Entorno de ejecución → Ejecutar todo**, y autoriza el acceso a Google Drive cuando lo pida.
3. Deja la pestaña abierta. Todo se guarda en `tabd_embeddings/` en Google Drive a medida que avanza.

Necesita unos 600 MB libres en Drive. **Si se corta:** vuelve a **Ejecutar todo**; lo que ya está hecho se salta.
"""

MODELS = """
#@title 3. SciNCL, BERT y E5-base
!python colab_embed.py --models scincl bert e5_base --out "{OUT}"
"""

SPECTER = """
#@title 4. SPECTER2 (necesita la librería adapters)
!pip install -q -U adapters
!python colab_embed.py --models specter2 --out "{OUT}"
"""

CHECK = """
#@title 5. Comprobar y terminar de subir a Drive
import numpy as np
expected = {"scincl": 768, "bert": 768, "e5_base": 768, "specter2": 768}
counts = {"corpus": 48018, "query": 1000, "title": 1000}
ok = True
for model, dim in expected.items():
    for kind, n in counts.items():
        path = f"{OUT}/{kind}_{model}.npy"
        if not os.path.exists(path):
            print(f"FALTA  {kind}_{model}.npy")
            ok = False
            continue
        shape = np.load(path, mmap_mode="r").shape
        good = shape == (n, dim)
        ok = ok and good
        print(f"{'OK    ' if good else 'MAL   '} {kind}_{model}.npy {shape}")
print("\\nTodo correcto." if ok else "\\nHay archivos faltantes: copia aquí el error de la celda 3 o 4.")
drive.flush_and_unmount()
print("Drive sincronizado. Revisa en la carpeta que estén los 12 archivos.")
"""

# ---------------------------------------------------------------------------
# Pool de escalabilidad
# ---------------------------------------------------------------------------
SCALE_INTRO = """
# Pool de escalabilidad (E5-base)

Vectoriza abstracts en tramos de 50 000, de unos 150 MB cada uno, hasta el número de tramos elegido en la
celda 3. **Comprueba antes que Drive tenga ese espacio libre**: si se llena, los archivos no se suben.

**Cómo usarlo**

1. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4 → Guardar**.
2. Menú **Entorno de ejecución → Ejecutar todo**, y autoriza el acceso a Google Drive cuando lo pida.
3. Deja la pestaña abierta, con la tapa de la laptop abierta y sin suspender el equipo.

**Si se corta o se acaba la GPU gratis:** vuelve a **Ejecutar todo**, aunque sea otro día. Lo que ya está
hecho se salta.

Cada línea de avance indica cuánto tardó ese tramo. Todo queda en `tabd_scale_e5_base/` en Google Drive.
"""

SCALE_PLAN = """
#@title 3. Elegir qué tramos generar
PRIMER_TRAMO = 7  #@param {type:"integer"}
ULTIMO_TRAMO = 10  #@param {type:"integer"}
CHUNK_VECTORS = 50_000

wanted = range(PRIMER_TRAMO, ULTIMO_TRAMO + 1)
missing = sum(not os.path.exists(f"{OUT}/part_e5_base_{(t - 1) * CHUNK_VECTORS:08d}.npy") for t in wanted)
START_AT = (PRIMER_TRAMO - 1) * CHUNK_VECTORS
STOP_AFTER = ULTIMO_TRAMO * CHUNK_VECTORS
print(f"Tramos {PRIMER_TRAMO} a {ULTIMO_TRAMO}: faltan {missing}.")
print(f"Comprueba que Drive tenga libres al menos {missing * 150 + 300} MB: Colab no puede medirlo aquí.")
"""

SCALE = """
#@title 4. Vectorizar (varias horas; se puede retomar)
!python colab_embed_scale.py --model e5_base --n 1000000 --start-at {START_AT} --stop-after {STOP_AFTER} --batch-size 64 --out "{OUT}"
"""

SCALE_END = """
#@title 5. Terminar de subir a Drive
drive.flush_and_unmount()
print("Drive sincronizado. Revisa la carpeta tabd_scale_e5_base y avísame cuántos tramos quedaron.")
"""


def write_notebook(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "name": name},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = cfg.ROOT / name
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook: {out} ({out.stat().st_size / 1024:.0f} KB)")


def setup_cell(folder, wrapped):
    return code(SETUP.replace("{folder}", folder).replace("{bundle}", wrapped), hidden=True)


def main():
    wrapped = "\n".join(textwrap.wrap(bundle(), 120))
    write_notebook("colab_embeddings.ipynb", [
        markdown(MODELS_INTRO),
        setup_cell("tabd_embeddings", wrapped),
        code(INSTALL),
        code(MODELS),
        code(SPECTER),
        code(CHECK),
    ])
    write_notebook("colab_scale.ipynb", [
        markdown(SCALE_INTRO),
        setup_cell("tabd_scale_e5_base", wrapped),
        code(INSTALL),
        code(SCALE_PLAN),
        code(SCALE),
        code(SCALE_END),
    ])


if __name__ == "__main__":
    main()

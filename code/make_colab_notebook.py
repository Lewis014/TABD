"""
Genera colab_embeddings.ipynb: un único notebook con el código y los ids que necesita,
para no tener que subir nada más a Colab.

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


INTRO = """
# Embeddings para la revisión del paper 54

**Cómo usarlo**

1. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4 → Guardar**.
2. Menú **Entorno de ejecución → Ejecutar todo**, y autoriza el acceso a Google Drive cuando lo pida.
3. Deja la pestaña abierta. Todo se guarda en Google Drive a medida que avanza.

**Si se corta o se acaba la GPU gratis:** vuelve a **Ejecutar todo**, aunque sea otro día. Lo que ya
está hecho se salta y sigue donde quedó.

**Qué genera en Google Drive**

- `tabd_embeddings/`: los 4 modelos nuevos, unos 150 MB por modelo. Es lo imprescindible y corre primero.
- `tabd_scale_e5/`: el pool de 1 millón de abstracts para la escalabilidad, unos 4 GB. Tarda varias
  horas y necesita ese espacio libre en Drive. Para omitirlo, desmarca `ESCALA_1M` en la primera celda.
"""

SETUP = """
#@title 1. Preparar el entorno
ESCALA_1M = True  #@param {type:"boolean"}

import base64, io, os, zipfile
import torch
from google.colab import drive

assert torch.cuda.is_available(), "No hay GPU: elige GPU T4 en Entorno de ejecución y vuelve a ejecutar."
print("GPU:", torch.cuda.get_device_name(0))

drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/tabd_embeddings"
OUT_SCALE = "/content/drive/MyDrive/tabd_scale_e5"
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

MODELS = """
#@title 3. SciNCL, BERT y E5-base
!python colab_embed.py --models scincl bert e5_base --out "{OUT}"
"""

SPECTER = """
#@title 4. SPECTER2 (necesita la librería adapters)
!pip install -q -U adapters
!python colab_embed.py --models specter2 --out "{OUT}"
# adapters puede bajar la versión de transformers: se restaura para la escala
!pip install -q -U transformers sentence-transformers
"""

CHECK = """
#@title 5. Comprobar los modelos
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
"""

SCALE = """
#@title 6. Pool de escalabilidad (varias horas; se puede retomar)
if ESCALA_1M:
    !python colab_embed_scale.py --model e5 --n 1000000 --out "{OUT_SCALE}"
else:
    print("Omitido: ESCALA_1M está desmarcado.")
"""


def main():
    wrapped = "\n".join(textwrap.wrap(bundle(), 120))
    cells = [
        markdown(INTRO),
        code(SETUP.replace("{bundle}", wrapped), hidden=True),
        code(INSTALL),
        code(MODELS),
        code(SPECTER),
        code(CHECK),
        code(SCALE),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "name": "colab_embeddings.ipynb"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = cfg.ROOT / "colab_embeddings.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

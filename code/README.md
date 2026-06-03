# Búsqueda semántica HNSW sobre abstracts científicos — pipeline experimental

Código que respalda las secciones 3–5 del paper. El cómputo está partido en dos:
los **embeddings** se generan en Google Colab (GPU T4) y **todo lo de FAISS** corre
en local (CPU). El puente entre ambas mitades son los archivos `.npy` de vectores.

```
ArXiv (2M) --[Colab]--> 50K muestreados --> 49K corpus + 1K queries
                                |
                         embeddings (.npy)
                                |
                          [descarga + copia]
                                v
        local: ground truth exacto --> 36 índices HNSW --> métricas --> tablas
```

## Estructura

| Archivo | Dónde corre | Qué hace |
|---|---|---|
| `config.py` | ambos | Parámetros compartidos (rutas, 36 configs, K, etc.) |
| `colab_embed.py` | Colab | Ingesta + muestreo estratificado + embeddings |
| `03_groundtruth.py` | local | k-NN exacto por fuerza bruta (la "verdad") |
| `04_benchmark.py` | local | Construye y evalúa las 36 configuraciones HNSW |
| `05_export.py` | local | Genera `results.csv` y las tablas LaTeX |

## Decisiones fijadas

- **50 000** abstracts (`gfissore/arxiv-abstracts-2021`), muestreo estratificado por categoría mayor.
- **1 000** consultas (holdout estratificado) contra **49 000** de corpus.
- Modelos: **SciBERT** (768d) y **E5-large** (1024d). E5 usa prefijos `query:` / `passage:`.
- Motor: **FAISS** (`IndexHNSWFlat`). ChromaDB descartado.
- Rejilla: `M ∈ {8,16,32}` × `ef_construction ∈ {100,200,400}` × `métrica ∈ {coseno, L2}` = **36**.
- `ef_search = 50` fijo (no forma parte de la rejilla).
- Ground truth: fuerza bruta (`IndexFlatL2` / `IndexFlatIP`).
- Métricas: **Recall@{1,5,10}**, **MRR**, **latencia (ms)**.

---

## Fase A — Embeddings (Google Colab, GPU T4)

1. Nuevo notebook en Colab → Entorno de ejecución → cambiar a **GPU (T4)**.
2. Sube `config.py` y `colab_embed.py` al entorno (o clona el repo).
3. Ejecuta:

   ```python
   !pip install -q datasets sentence-transformers pandas pyarrow
   !python colab_embed.py
   from google.colab import files
   files.download("embeddings.zip")
   ```

4. Descomprime `embeddings.zip` dentro de `E:\TABD\code\data\`. Deben quedar:

   ```
   data/corpus_scibert.npy   data/query_scibert.npy
   data/corpus_e5.npy        data/query_e5.npy
   data/corpus_meta.parquet  data/query_meta.parquet
   ```

> Generar 50K × 2 modelos en T4 toma del orden de minutos. En CPU serían horas:
> por eso esta fase va en Colab.

---

## Fase B — FAISS y evaluación (local, Windows)

```powershell
cd E:\TABD\code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_local.txt

python 03_groundtruth.py   # crea groundtruth_*.npy
python 04_benchmark.py     # crea results/results.csv  (las 36 configs)
python 05_export.py        # crea las tablas LaTeX + resumen en consola
```

Salida final en `results/`:

- `results.csv` — datos crudos de las 36 configuraciones.
- `tabla_resultados_best.tex` — tabla principal (4 filas) lista para el paper.
- `tabla_completa.tex` — las 36 filas, para apéndice.

Copia el contenido de `tabla_resultados_best.tex` sobre la tabla `tab:resultados`
del `paper.tex` y ya tienes la Sección 5 con datos reales.

## Reproducibilidad

Todo el azar está fijado con `SEED` en `config.py`. Mismo seed → mismo muestreo,
mismo holdout, mismos resultados.

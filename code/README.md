# Búsqueda semántica HNSW sobre abstracts científicos — pipeline experimental

Código que respalda las secciones 3–5 del paper. El cómputo está partido en dos: los
**embeddings** se generan con GPU y **todo lo de FAISS** corre en local (CPU). El puente
entre ambas mitades son los archivos `.npy` de vectores.

```
ArXiv (2M) --[GPU]--> muestra con ids congelados --> 48 018 de corpus + 1 000 consultas
                                |
                         embeddings (.npy)
                                v
  local: ground truth exacto --> índices HNSW --> métricas --> tablas y figuras
```

## Estructura

| Archivo | Dónde corre | Qué hace |
|---|---|---|
| `config.py` | ambos | Parámetros compartidos (rutas, modelos, rejilla, regla de selección) |
| `encoders.py` | ambos | Carga y uso de cada modelo de embedding |
| `colab_embed.py` | GPU | Vectoriza corpus, consultas y títulos por id |
| `colab_embed_scale.py` | GPU | Pool grande para escalabilidad, guardando por tramos |
| `colab_export_text.py` | Colab | Texto del corpus para la demo y los títulos |
| `02_split.py` | local | Quita las consultas del corpus y congela los ids en `splits/` |
| `03_groundtruth.py` | local | k-NN exacto por fuerza bruta (la "verdad") |
| `04_benchmark.py` | local | 18 índices × 6 `ef_search`, 5 construcciones cada uno |
| `05_export.py` | local | Elige la configuración de referencia, frente de Pareto, tablas |
| `07_ef_search_sensitivity.py` | local | Curvas de `ef_search` |
| `09_natural_queries.py` | local | Known-item título→abstract, exacta y HNSW |
| `08_significance.py` | local | Pruebas pareadas entre modelos, con Holm |
| `10_scalability.py` | local | Latencia frente a N, a recall constante |
| `11_end_to_end_latency.py` | local | Vectorizar + buscar, por modelo |
| `12_index_families.py` | local | HNSW frente a IVF-Flat, IVF-PQ, BallTree y Annoy (si está instalado) |
| `13_factorial_models.py` | local | Dominio frente a objetivo de entrenamiento |
| `14_tables.py` | local | Tablas LaTeX del artículo (`results/tabla_*.tex`) |
| `15_tables_es.py` | local | Las mismas tablas en español para `paper.tex` (`tabla_*_es.tex`) |
| `06_plots.py`, `06b`, `07b`, `09c`, `10b`, `12b` | local | Figuras en `../Images` |
| `app.py` | local | Demo Streamlit |

## Decisiones fijadas

- Muestra: `gfissore/arxiv-abstracts-2021`. Los ids de `splits/` definen el experimento.
- **1 000** consultas fuera de **48 018** documentos indexados.
- Rejilla: `M ∈ {8,16,32}` × `ef_construction ∈ {100,200,400}` × `ef_search ∈ {10,…,400}`.
- Similitud coseno: `IndexHNSWFlat` con producto interno sobre vectores unitarios.
- Cada índice se construye 5 veces con el orden de inserción barajado.
- Latencia con 1 hilo y una consulta a la vez. Construcción con 6 hilos.
- Configuración de referencia: la de menor latencia con Recall@10 ≥ 0.99 en SciBERT y E5-large.

## Repetir los experimentos

```bash
pip install -r requirements_local.txt
bash run_experiments.sh   # 02, 03, 04, 05, 07, 09, 08, 10, 11, 12
```

No correr nada pesado a la vez: se miden latencias. Los archivos originales de los
embeddings van en `data/` (`corpus_{modelo}.npy`, `query_{modelo}.npy` y sus `_meta.parquet`),
y el resto del pipeline trabaja sobre `data/clean/`.

## Modelos adicionales y pool de escalabilidad (GPU)

```bash
python colab_embed.py --models scincl bert e5_base
python colab_embed.py --models specter2    # requiere adapters; mejor en un entorno aparte
python colab_embed_scale.py --model e5_base --n 1000000
```

En Colab, lo más simple son `colab_embeddings.ipynb` y `colab_scale.ipynb`, que traen todo lo necesario
(los genera `make_colab_notebook.py`). Los `.npy` de modelos van en `data/clean/` y los del pool en
`data/scale_e5_base/`. Después:

```bash
bash run_models.sh
python 10_scalability.py --model e5_base
```

Para que el equipo no se suspenda durante una corrida larga: `python keep_awake.py bash run_experiments.sh`.

## Figuras

```bash
python 06_plots.py
python 06b_plot_diagram.py
python 07b_plot_ef_search.py
python 09c_plot_natural.py
python 10b_plot_scalability.py
python 12b_plot_families.py
```

## Reproducibilidad

Todo el azar está fijado con `SEED` en `config.py`, y la muestra por los ids de `splits/`.
La construcción de HNSW con varios hilos no es determinista: por eso se repite y se
informa la desviación entre construcciones. Las latencias dependen del equipo.

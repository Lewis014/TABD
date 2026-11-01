#!/usr/bin/env bash
# Repite en orden los experimentos locales. No correr nada pesado a la vez: mide latencias.
#
#   bash run_experiments.sh
#
# 08 va después de 09 porque también compara los resultados de título->abstract. 13 y 14
# reúnen lo anterior; si ya están los modelos adicionales y el pool de 1M, entran solos.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

python 02_split.py
python 03_groundtruth.py
python 04_benchmark.py
python 05_export.py
python 07_ef_search_sensitivity.py
python 09_natural_queries.py
python 11_end_to_end_latency.py
python 13_factorial_models.py
python 08_significance.py
python 10_scalability.py --model scibert
if [ -d data/scale_e5 ]; then python 10_scalability.py --model e5; fi
python 12_index_families.py
python 14_tables.py

python 06_plots.py
python 06b_plot_diagram.py
python 07b_plot_ef_search.py
python 09c_plot_natural.py
python 10b_plot_scalability.py
python 12b_plot_families.py

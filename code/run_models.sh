#!/usr/bin/env bash
# Evalúa los modelos adicionales, ya vectorizados en data/clean/, con la configuración
# de referencia. No correr nada pesado a la vez: mide latencias.
#
#   bash run_models.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

python 03_groundtruth.py
python 09_natural_queries.py
python 11_end_to_end_latency.py
python 13_factorial_models.py
python 08_significance.py

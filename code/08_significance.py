"""
FASE B.4 — Pruebas de significancia estadística SciBERT vs. E5-large (local, CPU)
================================================================================

Responde a la observación O3 de la revisión por pares: el paper afirma que
SciBERT supera a E5-large (Recall@10 0,998 vs 0,995) pero no verifica si esa
diferencia de 0,003 es estadísticamente significativa o ruido muestral.

Comparamos la MEJOR configuración de cada modelo (ambas coseno, M=32,
ef_construction=400, ef_search=50), reconstruyendo los índices y extrayendo el
resultado CONSULTA A CONSULTA (las 1 000 consultas son las mismas para ambos
modelos, por lo que la comparación es pareada).

Matiz metodológico: cada modelo tiene su propio ground truth exacto (el vecino
"verdadero" depende del espacio de embeddings). Por tanto Recall@10 mide, para
cada modelo, la fidelidad de HNSW respecto a su propia búsqueda exacta. La prueba
compara esas dos fidelidades sobre el mismo conjunto de consultas.

Tests (los sugeridos por el revisor):
  1. McNemar pareado sobre el acierto binario Recall@10 == 1,0 (recall perfecto).
     -> apropiado para tasas de acierto binarias; usamos la versión exacta
        (binomial) porque los pares discordantes pueden ser pocos.
  2. Wilcoxon de rangos con signo sobre el rango recíproco por consulta (base del
     MRR) y sobre el Recall@10 por consulta (fraccional).
Se reporta estadístico, valor p bilateral y tamaño del efecto.

Salida: imprime el informe en consola y guarda results/significance.txt
"""

import numpy as np
import faiss
from scipy.stats import wilcoxon, binomtest

import config as cfg
from importlib import import_module

_bench = import_module("04_benchmark")
normalized = _bench.normalized

# Mejor configuración de cada modelo (idéntica salvo el modelo)
CONFIG = {"metric": "cosine", "M": 32, "ef_construction": 400}
MODELS = ["scibert", "e5"]
K = 10   # evaluamos Recall@10 y el rango del vecino más cercano dentro del top-K_MAX


def retrieve(model):
    """Reconstruye el índice de la mejor config del modelo y devuelve, por
    consulta: el vector de Recall@10 y el rango recíproco del vecino verdadero."""
    corpus = normalized(np.load(cfg.corpus_emb_path(model)))
    queries = normalized(np.load(cfg.query_emb_path(model)))
    truth = np.load(cfg.groundtruth_path(model, CONFIG["metric"]))

    d = corpus.shape[1]
    index = faiss.IndexHNSWFlat(d, CONFIG["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = CONFIG["ef_construction"]
    index.add(corpus)
    index.hnsw.efSearch = cfg.EF_SEARCH
    _, retrieved = index.search(queries, cfg.K_MAX)

    n = len(queries)
    recall10 = np.empty(n)
    rr = np.empty(n)   # reciprocal rank del vecino más cercano verdadero
    for i in range(n):
        r, t = retrieved[i], truth[i]
        recall10[i] = len(set(r[:K]) & set(t[:K])) / K
        true_nn = t[0]
        pos = np.where(r == true_nn)[0]
        rr[i] = 1.0 / (pos[0] + 1) if len(pos) else 0.0
    return recall10, rr


def mcnemar_exact(succ_a, succ_b):
    """McNemar exacto sobre aciertos binarios pareados.
    b = A acierta y B falla; c = A falla y B acierta.
    H0: prob(discordancia a favor de A) = 0,5."""
    b = int(np.sum(succ_a & ~succ_b))
    c = int(np.sum(~succ_a & succ_b))
    res = binomtest(b, b + c, 0.5, alternative="two-sided") if (b + c) > 0 else None
    p = res.pvalue if res is not None else 1.0
    return b, c, p


def rank_biserial(x, y):
    """Tamaño del efecto para Wilcoxon pareado: correlación biserial de rangos."""
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    r_plus = np.sum(ranks[d > 0])
    r_minus = np.sum(ranks[d < 0])
    total = r_plus + r_minus
    return (r_plus - r_minus) / total


def main():
    r10, rr = {}, {}
    for m in MODELS:
        r10[m], rr[m] = retrieve(m)

    a, b = MODELS  # "scibert", "e5"
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 70)
    out("O3 — Significancia estadística: SciBERT vs. E5-large")
    out(f"Config: coseno, M={CONFIG['M']}, ef_c={CONFIG['ef_construction']}, "
        f"ef_search={cfg.EF_SEARCH} | n={len(r10[a])} consultas (pareadas)")
    out("=" * 70)

    out(f"\nRecall@10 medio:  SciBERT={r10[a].mean():.4f}   E5-large={r10[b].mean():.4f}"
        f"   (dif={r10[a].mean()-r10[b].mean():+.4f})")
    out(f"MRR medio:        SciBERT={rr[a].mean():.4f}   E5-large={rr[b].mean():.4f}"
        f"   (dif={rr[a].mean()-rr[b].mean():+.4f})")

    # --- Test 1: McNemar sobre Recall@10 perfecto (binario) ---
    succ_a = r10[a] >= 1.0
    succ_b = r10[b] >= 1.0
    bb, cc, p_mc = mcnemar_exact(succ_a, succ_b)
    out("\n--- McNemar exacto (acierto = Recall@10 perfecto por consulta) ---")
    out(f"Consultas con Recall@10=1,0:  SciBERT={int(succ_a.sum())}   "
        f"E5-large={int(succ_b.sum())}")
    out(f"Pares discordantes: b (SciBERT sí, E5 no)={bb}   "
        f"c (SciBERT no, E5 sí)={cc}")
    out(f"p-valor bilateral (binomial exacto) = {p_mc:.4f}")

    # --- Test 2: Wilcoxon sobre Recall@10 fraccional ---
    diff_r10 = r10[a] - r10[b]
    if np.any(diff_r10 != 0):
        w_r10, p_r10 = wilcoxon(r10[a], r10[b])
        es_r10 = rank_biserial(r10[a], r10[b])
    else:
        w_r10, p_r10, es_r10 = float("nan"), 1.0, 0.0
    out("\n--- Wilcoxon rangos con signo (Recall@10 por consulta) ---")
    out(f"W={w_r10:.1f}   p-valor bilateral={p_r10:.4f}   "
        f"tamaño de efecto (rank-biserial)={es_r10:+.3f}")

    # --- Test 3: Wilcoxon sobre rango recíproco (MRR) ---
    diff_rr = rr[a] - rr[b]
    if np.any(diff_rr != 0):
        w_rr, p_rr = wilcoxon(rr[a], rr[b])
        es_rr = rank_biserial(rr[a], rr[b])
    else:
        w_rr, p_rr, es_rr = float("nan"), 1.0, 0.0
    out("\n--- Wilcoxon rangos con signo (rango recíproco / MRR por consulta) ---")
    out(f"W={w_rr:.1f}   p-valor bilateral={p_rr:.4f}   "
        f"tamaño de efecto (rank-biserial)={es_rr:+.3f}")

    # --- Veredicto ---
    out("\n" + "=" * 70)
    alpha = 0.05
    sig_r10 = p_r10 < alpha
    out(f"Con alpha={alpha}: la diferencia en Recall@10 es "
        f"{'SIGNIFICATIVA' if sig_r10 else 'NO significativa'}.")
    out("=" * 70)

    out_path = cfg.RESULTS_DIR / "significance.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nInforme guardado en {out_path}")


if __name__ == "__main__":
    main()

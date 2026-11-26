"""
FASE B.6 — Pruebas de significancia entre modelos (local)
=========================================================

Compara pares de modelos sobre los mismos textos, por lo que las pruebas son pareadas y
la unidad es la consulta. Corre después de 09 (y de 13 si hay modelos adicionales).

Fidelidad abstract->abstract, en la configuración de referencia. Cada modelo tiene su
propio ground truth, así que se compara cuánto se aleja HNSW de la búsqueda exacta en
el espacio de cada modelo:
  - Wilcoxon de rangos con signo sobre el Recall@10 por consulta, promediado entre las
    construcciones del índice para no confundir la variación entre consultas con la
    variación entre construcciones.
  - McNemar exacto sobre el acierto binario Recall@10 == 1,0, en cada construcción. Se
    informan los pares discordantes y se toma el mayor valor p, el caso conservador.

Recuperación título->abstract, con búsqueda exacta (el límite de cada modelo):
  - McNemar exacto sobre "el abstract relevante está en el top-10".
  - Wilcoxon de rangos con signo sobre el rango recíproco en el top-10.

Wilcoxon descarta los pares sin diferencia (se informa cuántos quedan). El tamaño de
efecto es la correlación biserial de rangos, con rangos promedio en los empates. Además
se da el IC 95 % de la diferencia media por bootstrap pareado sobre consultas, y los
valores p se corrigen por Holm sobre toda la familia de pruebas.

Salida: results/significance.csv y results/significance.txt
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest, rankdata

import config as cfg

# Pares de interés: cada uno aísla un factor cuando los dos modelos están disponibles
PAIRS = [
    ("scibert", "e5"),        # el contraste original
    ("scibert", "scincl"),    # objetivo de entrenamiento: SciNCL parte de SciBERT
    ("scibert", "specter2"),  # objetivo de entrenamiento, con adaptador de consulta
    ("bert", "scibert"),      # dominio, sin entrenamiento para recuperación
    ("scincl", "e5_base"),    # dominio, con entrenamiento para similitud
    ("e5_base", "e5"),        # dimensión, misma familia
]
N_BOOTSTRAP = 10_000
ALPHA = 0.05


def mcnemar_exact(succ_a, succ_b):
    """McNemar exacto. b = A acierta y B falla; c = A falla y B acierta."""
    b = int(np.sum(succ_a & ~succ_b))
    c = int(np.sum(~succ_a & succ_b))
    p = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue if (b + c) > 0 else 1.0
    return b, c, p


def rank_biserial(x, y):
    """Tamaño del efecto para Wilcoxon pareado, con rangos promedio en los empates."""
    d = x - y
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    return (r_plus - r_minus) / (r_plus + r_minus)


def bootstrap_ci(x, y):
    """IC 95 % percentil de la diferencia media, remuestreando consultas."""
    d = x - y
    rng = np.random.default_rng(cfg.SEED)
    idx = rng.integers(0, len(d), size=(N_BOOTSTRAP, len(d)))
    means = d[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def wilcoxon_row(scenario, metric, a, b, x, y):
    n_nonzero = int(np.sum(x != y))
    stat, p = wilcoxon(x, y) if n_nonzero else (np.nan, 1.0)
    low, high = bootstrap_ci(x, y)
    return {"scenario": scenario, "metric": metric, "test": "wilcoxon", "model_a": a, "model_b": b,
            "mean_a": x.mean(), "mean_b": y.mean(), "diff": x.mean() - y.mean(),
            "ci95_low": low, "ci95_high": high, "n_nonzero": n_nonzero, "statistic": stat,
            "effect_rank_biserial": rank_biserial(x, y), "p": p}


def mcnemar_row(scenario, metric, a, b, succ_a, succ_b):
    """succ_* con forma (construcciones, consultas) o (consultas,)."""
    succ_a, succ_b = np.atleast_2d(succ_a), np.atleast_2d(succ_b)
    runs = [mcnemar_exact(sa, sb) for sa, sb in zip(succ_a, succ_b)]
    worst = max(runs, key=lambda r: r[2])
    return {"scenario": scenario, "metric": metric, "test": "mcnemar", "model_a": a, "model_b": b,
            "mean_a": succ_a.mean(), "mean_b": succ_b.mean(), "diff": succ_a.mean() - succ_b.mean(),
            "discordant_ab": worst[0], "discordant_ba": worst[1],
            "discordant_ab_range": f"{min(r[0] for r in runs)}-{max(r[0] for r in runs)}",
            "discordant_ba_range": f"{min(r[1] for r in runs)}-{max(r[1] for r in runs)}",
            "p": worst[2]}


def holm(pvalues):
    """Valores p ajustados por Holm."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def load_fidelity():
    data = dict(np.load(cfg.perquery_path()))
    if cfg.perquery_extra_path().exists():
        data.update(np.load(cfg.perquery_extra_path()))
    return data


def main():
    ref = cfg.load_reference()
    tag = f"{ref['M']}_{ref['ef_construction']}_{ref['ef_search']}"
    fidelity = load_fidelity()

    rows = []
    for a, b in PAIRS:
        ka, kb = f"{a}_{tag}_r10", f"{b}_{tag}_r10"
        if ka in fidelity and kb in fidelity:
            ha, hb = fidelity[ka], fidelity[kb]
            rows.append(wilcoxon_row("abstract-abstract", "recall@10", a, b,
                                     ha.mean(axis=0) / cfg.K_MAX, hb.mean(axis=0) / cfg.K_MAX))
            rows.append(mcnemar_row("abstract-abstract", "recall@10=1", a, b,
                                    ha == cfg.K_MAX, hb == cfg.K_MAX))

        if cfg.natural_ranks_path(a).exists() and cfg.natural_ranks_path(b).exists():
            ra = np.load(cfg.natural_ranks_path(a))["exact"]
            rb = np.load(cfg.natural_ranks_path(b))["exact"]
            rows.append(mcnemar_row("title-abstract", "hit@10", a, b, ra <= cfg.K_MAX, rb <= cfg.K_MAX))
            rra = np.where(ra <= cfg.K_MAX, 1.0 / ra, 0.0)
            rrb = np.where(rb <= cfg.K_MAX, 1.0 / rb, 0.0)
            rows.append(wilcoxon_row("title-abstract", "rr@10", a, b, rra, rrb))

    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"])
    df["significant"] = df["p_holm"] < ALPHA
    # Los valores p van sin redondear: muchos están por debajo de 1e-6
    pvalues = ["p", "p_holm"]
    df.drop(columns=pvalues).round(6).join(df[pvalues]).to_csv(cfg.RESULTS_DIR / "significance.csv", index=False)

    lines = [f"Pruebas pareadas en la configuración de referencia {ref} "
             f"(alpha={ALPHA}, Holm sobre {len(df)} pruebas)", ""]
    for _, r in df.iterrows():
        head = f"{r['scenario']:18s} {r['metric']:12s} {r['model_a']} vs {r['model_b']}: " \
               f"{r['mean_a']:.4f} vs {r['mean_b']:.4f}"
        if r["test"] == "wilcoxon":
            body = f"Wilcoxon W={r['statistic']:.1f} n={int(r['n_nonzero'])} " \
                   f"r={r['effect_rank_biserial']:+.3f} IC95=[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
        else:
            body = f"McNemar b={r['discordant_ab_range']} c={r['discordant_ba_range']}"
        lines.append(f"{head} | {body} | p={r['p']:.2e} p_holm={r['p_holm']:.2e}"
                     f"{' *' if r['significant'] else ''}")
    report = "\n".join(lines)
    print(report)
    (cfg.RESULTS_DIR / "significance.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

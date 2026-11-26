"""
Tablas LaTeX del artículo que no genera 05_export.py, para incluirlas con \\input.

Lee los CSV de results/ y escribe en la misma carpeta:
  tabla_ef_search.tex      ef_search en la M y ef_construction de referencia
  tabla_naturales.tex      known-item título->abstract por modelo
  tabla_significancia.tex  pruebas pareadas
  tabla_escalabilidad.tex  latencia y ventaja frente a la exacta, por N y objetivo de recall
  tabla_e2e.tex            vectorizar + buscar
  tabla_familias.tex       HNSW frente a otras familias
  tabla_modelos.tex        dominio frente a objetivo de entrenamiento

Una tabla cuyos datos aún no existen se omite.

Ejecutar:
    python 14_tables.py
"""

import pandas as pd

import config as cfg

BATCH_THREADS = 12   # el mismo valor que 10_scalability.py


def label(model: str) -> str:
    return cfg.ALL_MODELS[model]["label"]


# Qué objetivo de similitud tiene cada modelo, para la tabla del cuadro dominio x objetivo
TRAINING = {"scibert": "None", "bert": "None", "scincl": "Citations", "specter2": "Citations",
            "e5_base": "Query--passage", "e5": "Query--passage"}


def latex_table(caption, tag, spec, header, rows, size=r"\small", group_header=None) -> str:
    """Tabla para LNCS (122 mm de ancho; llncs ya deja 1.4 pt entre columnas). Una fila que es
    una cadena se copia tal cual: sirve para subtítulos que ocupan toda la fila."""
    body = [row if isinstance(row, str) else " & ".join(str(c) for c in row) + r" \\" for row in rows]
    return "\n".join([
        r"\begin{table}[tbp]",
        rf"\caption{{{caption}}}\label{{tab:{tag}}}",
        r"\centering",
        size,
        rf"\begin{{tabular}}{{{spec}}}",
        r"\hline",
        *([group_header] if group_header else []),
        " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\",
        r"\hline",
        *body,
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])


def ef_search(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "ef_search_sensitivity.csv")
    df = df[df["M"] == ref["M"]]
    rows = [(label(r["model"]), int(r["ef_search"]), f"{r['recall@1']:.3f}", f"{r['recall@5']:.3f}",
             f"{r['recall@10']:.3f}$\\pm${r['recall@10_sd']:.3f}", f"{r['latency_ms']:.2f}",
             f"{r['latency_p95_ms']:.2f}")
            for _, r in df.sort_values(["model", "ef_search"], ascending=[False, True]).iterrows()]
    return latex_table(
        rf"Sensitivity to \textit{{ef\_search}} with $M={ref['M']}$ and "
        rf"\textit{{ef\_construction}}$={ref['ef_construction']}$ over {cfg.N_BUILDS} builds "
        r"(latency per query, one thread).",
        "efsearch", "|l|c|c|c|c|c|c|",
        ["Model", r"\textit{ef\_s}", "R@1", "R@5", "R@10", "Lat. (ms)", "p95 (ms)"], rows)


def naturales(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "natural_queries.csv")
    rows = []
    for model in [m for m in cfg.ALL_MODELS if m in set(df["model"])]:
        e = df[(df["model"] == model) & (df["search"] == "exact")].iloc[0]
        h = df[(df["model"] == model) & (df["search"] == "hnsw")].iloc[0]
        rows.append((label(model), f"{e['recall@1']:.3f}", f"{e['recall@5']:.3f}", f"{e['recall@10']:.3f}",
                     f"{e['mrr@10']:.3f}", f"{h['recall@10']:.3f}", f"{e['sim_own']:.3f}",
                     f"{e['sim_top1']:.3f}", f"{e['median_rank']:.0f}"))
    return latex_table(
        r"Known-item retrieval title$\rightarrow$abstract over 1\,000 titles, with exact search and "
        rf"with HNSW ($M={ref['M']}$, \textit{{ef\_c}}$={ref['ef_construction']}$, "
        rf"\textit{{ef\_s}}$={ref['ef_search']}$). Similarities are between the title and its own "
        r"abstract, and between the title and the top-ranked document.",
        "naturales", "|l|c|c|c|c|c|c|c|c|",
        ["Model", "R@1", "R@5", "R@10", "MRR@10", "R@10", "Own", "Top-1", "rank"], rows,
        group_header=r" & \multicolumn{4}{c|}{\textbf{Exact search}} & \textbf{HNSW} & "
                     r"\multicolumn{2}{c|}{\textbf{Similarity}} & \textbf{Median} \\ \cline{2-8}")


def p_value(p: float) -> str:
    """Valor p en notación de LaTeX; por debajo de 1e-10 basta con la cota."""
    if p < 1e-10:
        return r"$<10^{-10}$"
    if p >= 0.01:
        return f"{p:.2f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def significancia(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "significance.csv")
    comparison = {("abstract-abstract", "recall@10"): r"Fidelity, R@10",
                  ("abstract-abstract", "recall@10=1"): r"Fidelity, R@10$=1$",
                  ("title-abstract", "hit@10"): r"Title, hit@10",
                  ("title-abstract", "rr@10"): r"Title, RR@10"}
    rows = []
    for (a, b), pair in df.groupby(["model_a", "model_b"], sort=False):
        rows.append(rf"\multicolumn{{5}}{{|l|}}{{\textit{{{label(a)} vs {label(b)}}}}} \\")
        for _, r in pair.iterrows():
            if r["test"] == "wilcoxon":
                detail = f"$n$={int(r['n_nonzero'])}, $r$={r['effect_rank_biserial']:+.3f}, " \
                         f"CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
            else:
                detail = f"$b$={r['discordant_ab_range']}, $c$={r['discordant_ba_range']}"
            rows.append((comparison[(r["scenario"], r["metric"])],
                         {"wilcoxon": "Wilcoxon", "mcnemar": "McNemar"}[r["test"]],
                         f"{r['mean_a']:.3f} / {r['mean_b']:.3f}", detail, p_value(r["p_holm"])))
    return latex_table(
        rf"Paired tests at the reference configuration, Holm-adjusted over {len(df)} tests. Fidelity: "
        r"abstract$\rightarrow$abstract against each model's exact search; Title: title$\rightarrow$abstract "
        r"with exact search. $n$: queries with a difference; $r$: rank-biserial correlation; CI: 95\,\% paired "
        r"bootstrap of the mean difference; $b$, $c$: discordant queries (range over the builds).",
        "significancia", "|l|l|c|l|c|", ["Comparison", "Test", "Means", "Detail", r"$p_{\mathrm{Holm}}$"],
        rows, size=r"\scriptsize")


def escalabilidad(ref):
    tables = []
    for model in cfg.ALL_MODELS:
        path = cfg.RESULTS_DIR / f"scalability_targets_{model}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        low, high = cfg.RECALL_TARGETS
        rows = []
        for N, g in df.groupby("N"):
            a, b = g[g["target"] == low].iloc[0], g[g["target"] == high].iloc[0]
            cells = [f"{int(N):,}"]
            cells.append(f"{b['exact_ms']:.2f}" if pd.notna(b["exact_ms"]) else f"{a['exact_ms']:.2f}")
            for r in (a, b):
                cells += ["--"] * 4 if pd.isna(r["hnsw_ms"]) else \
                    [int(r["ef_search"]), f"{r['hnsw_ms']:.2f}", f"{r['speedup']:.0f}$\\times$",
                     f"{r['speedup_batch']:.1f}$\\times$"]
            ok = b if pd.notna(b["build_time_s"]) else a
            cells += [f"{ok['build_time_s']:.0f}", f"{ok['index_mb']:.0f}"]
            rows.append(cells)
        tables.append(latex_table(
            rf"Latency against corpus size ({label(model)}, $M={ref['M']}$, "
            rf"\textit{{ef\_c}}$={ref['ef_construction']}$). Exact and HNSW latency in ms, one thread and one query "
            rf"at a time; HNSW uses the smallest \textit{{ef\_search}} reaching each Recall@10 target. Gain: speedup "
            rf"over exact search; Batch: the same with all queries together on {BATCH_THREADS} threads. Build time "
            rf"(s, {cfg.BUILD_THREADS} threads) and memory (MB) of the index.",
            f"escalabilidad_{model}", "|r|c|c|c|c|c|c|c|c|c|c|c|",
            ["$N$", "Exact", r"\textit{ef\_s}", "ms", "Gain", "Batch", r"\textit{ef\_s}", "ms", "Gain", "Batch",
             "Build", "MB"], rows,
            group_header=rf" & & \multicolumn{{4}}{{c|}}{{\textbf{{R@10 $\geq$ {low}}}}} & "
                         rf"\multicolumn{{4}}{{c|}}{{\textbf{{R@10 $\geq$ {high}}}}} & & \\ \cline{{3-10}}"))
    return "\n\n".join(tables)


def e2e(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "end_to_end_latency.csv")
    df = df[df["device"] == "cpu"]
    rows = []
    for model in [m for m in cfg.ALL_MODELS if m in set(df["model"])]:
        cells = [label(model)]
        for text in ("title", "abstract"):
            r = df[(df["model"] == model) & (df["text"] == text)].iloc[0]
            cells += [f"{r['encode_ms']:.0f}", f"{r['encode_p95_ms']:.0f}", f"{r['search_ms']:.2f}",
                      f"{r['total_ms']:.0f}"]
        rows.append(cells)
    share = 100 * df["encode_share"]
    return latex_table(
        rf"End-to-end latency per query on the CPU, in ms, over {int(df['n'].iloc[0])} texts encoded one at a "
        rf"time with {int(df['cpu_threads'].iloc[0])} threads, followed by HNSW search on one thread "
        rf"($M={ref['M']}$, \textit{{ef\_c}}$={ref['ef_construction']}$, \textit{{ef\_s}}$={ref['ef_search']}$). "
        rf"Encoding takes {share.min():.1f}--{share.max():.1f}\,\% of the total.",
        "e2e", "|l|c|c|c|c|c|c|c|c|",
        ["Model", "Encode", "p95", "Search", "Total", "Encode", "p95", "Search", "Total"], rows,
        group_header=r" & \multicolumn{4}{c|}{\textbf{Title}} & \multicolumn{4}{c|}{\textbf{Abstract}} \\ \cline{2-9}")


def familias(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "index_families_targets.csv")
    points = pd.read_csv(cfg.RESULTS_DIR / "index_families.csv")
    low, high = cfg.RECALL_TARGETS
    rows = []
    for model, group in df.groupby("model", sort=False):
        rows.append(rf"\multicolumn{{6}}{{|l|}}{{\textit{{{label(model)}}}}} \\")
        for family, g in group.groupby("family", sort=False):
            # La memoria y la construcción no dependen del parámetro de búsqueda
            p = points[(points["model"] == model) & (points["family"] == family)].iloc[0]
            cells = [family, f"{g['max_recall@10'].iloc[0]:.3f}"]
            for target in (low, high):
                r = g[g["target"] == target].iloc[0]
                cells.append("--" if pd.isna(r["latency_ms"]) else f"{r['latency_ms']:.2f}")
            cells += [f"{p['build_time_s']:.1f}", f"{p['index_mb']:.0f}"]
            rows.append(cells)
    return latex_table(
        rf"Index families on the same queries and ground truth (one thread, one query at a time). Best R@10: "
        rf"highest Recall@10 over the search parameter; then the latency of the fastest setting reaching each "
        rf"target (--: not reached). Build time includes training.",
        "familias", "|l|c|c|c|c|c|",
        ["Family", "Best R@10", rf"ms, R@10$\geq${low}", rf"ms, R@10$\geq${high}", "Build (s)", "MB"], rows)


def modelos(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "model_comparison.csv")
    rows = [(r["label"], r["domain"].capitalize(), TRAINING[r["model"]], int(r["dim"]),
             f"{r['fidelity_r10']:.3f}", f"{r['search_ms']:.2f}", f"{r['title_r1']:.3f}", f"{r['title_r10']:.3f}",
             f"{r['title_mrr10']:.3f}")
            for _, r in df.iterrows()]
    return latex_table(
        rf"Domain against training objective at the reference configuration ($M={ref['M']}$, "
        rf"\textit{{ef\_c}}$={ref['ef_construction']}$, \textit{{ef\_s}}$={ref['ef_search']}$). Fidelity: "
        r"abstract$\rightarrow$abstract Recall@10 of HNSW against exact search, and search latency on one thread. "
        r"Retrieval: title$\rightarrow$abstract with exact search.",
        "modelos", "|l|l|l|c|c|c|c|c|c|",
        ["Model", "Domain", "Training", "Dim.", "R@10", "ms", "R@1", "R@10", "MRR@10"], rows,
        group_header=r"\multicolumn{4}{|c|}{} & \multicolumn{2}{c|}{\textbf{Fidelity}} & "
                     r"\multicolumn{3}{c|}{\textbf{Retrieval}} \\ \cline{5-9}")


def main():
    ref = cfg.load_reference()
    for name, build in (("tabla_ef_search", ef_search), ("tabla_naturales", naturales),
                        ("tabla_significancia", significancia), ("tabla_escalabilidad", escalabilidad),
                        ("tabla_e2e", e2e), ("tabla_familias", familias), ("tabla_modelos", modelos)):
        try:
            text = build(ref)
        except FileNotFoundError as ex:
            print(f"{name}: se omite, falta {ex.filename}")
            continue
        (cfg.RESULTS_DIR / f"{name}.tex").write_text(text, encoding="utf-8")
        print(f"{name}.tex")


if __name__ == "__main__":
    main()

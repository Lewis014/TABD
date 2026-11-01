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


def label(model: str) -> str:
    return cfg.ALL_MODELS[model]["label"]


def latex_table(caption, tag, spec, header, rows) -> str:
    return "\n".join([
        r"\begin{table}[tbp]",
        rf"\caption{{{caption}}}\label{{tab:{tag}}}",
        r"\centering",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\hline",
        " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\",
        r"\hline",
        *[" & ".join(str(c) for c in row) + r" \\" for row in rows],
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
        ["Model", "R@1", "R@5", "R@10", "MRR@10", r"R@10$_{\mathrm{HNSW}}$", "Sim. own", "Sim. top", "Med. rank"], rows)


def significancia(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "significance.csv")
    rows = []
    for _, r in df.iterrows():
        scenario = "Abs$\\rightarrow$abs" if r["scenario"] == "abstract-abstract" else "Title$\\rightarrow$abs"
        metric = {"recall@10": "R@10", "recall@10=1": "R@10$=1$", "hit@10": "Hit@10", "rr@10": "RR@10"}[r["metric"]]
        if r["test"] == "wilcoxon":
            detail = f"$n$={int(r['n_nonzero'])}, $r$={r['effect_rank_biserial']:+.3f}, " \
                     f"CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
        else:
            detail = f"$b$={r['discordant_ab_range']}, $c$={r['discordant_ba_range']}"
        rows.append((scenario, f"{label(r['model_a'])} vs {label(r['model_b'])}", metric,
                     r["test"].capitalize(), f"{r['mean_a']:.3f} / {r['mean_b']:.3f}", detail,
                     f"{r['p_holm']:.1e}"))
    return latex_table(
        r"Paired tests over the same queries, Holm-adjusted. $n$: pairs with a difference; $r$: rank-biserial "
        r"correlation; CI: 95\,\% paired bootstrap of the mean difference; $b$, $c$: discordant pairs over the builds.",
        "significancia", "|l|l|l|l|c|l|c|",
        ["Scenario", "Pair", "Metric", "Test", "Means", "Detail", r"$p_{\mathrm{Holm}}$"], rows)


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
                cells += ["--"] * 3 if pd.isna(r["hnsw_ms"]) else \
                    [int(r["ef_search"]), f"{r['hnsw_ms']:.2f}", f"{r['speedup']:.0f}$\\times$"]
            ok = b if pd.notna(b["build_time_s"]) else a
            cells += [f"{ok['speedup_batch']:.1f}$\\times$", f"{ok['build_time_s']:.1f}", f"{ok['index_mb']:.0f}"]
            rows.append(cells)
        tables.append(latex_table(
            rf"Latency per query against corpus size ({label(model)}, $M={ref['M']}$, "
            rf"\textit{{ef\_c}}$={ref['ef_construction']}$; one thread, one query at a time). HNSW uses the smallest "
            rf"\textit{{ef\_search}} reaching each Recall@10 target. Batch: speedup when all queries run together on all threads, "
            rf"at the {high} target.",
            f"escalabilidad_{model}", "|r|c|c|c|c|c|c|c|c|c|c|",
            ["$N$", "Exact (ms)", rf"\textit{{ef\_s}}$_{{{low}}}$", "ms", "Gain",
             rf"\textit{{ef\_s}}$_{{{high}}}$", "ms", "Gain", "Batch", "Build (s)", "MB"], rows))
    return "\n\n".join(tables)


def e2e(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "end_to_end_latency.csv")
    rows = [(label(r["model"]), r["device"].upper(), r["text"].capitalize(), f"{r['encode_ms']:.1f}",
             f"{r['encode_p95_ms']:.1f}", f"{r['search_ms']:.2f}", f"{r['total_ms']:.1f}",
             f"{100 * r['encode_share']:.1f}\\,\\%") for _, r in df.iterrows()]
    return latex_table(
        rf"End-to-end latency per query over {int(df['n'].iloc[0])} texts: query encoding plus HNSW search "
        rf"($M={ref['M']}$, \textit{{ef\_c}}$={ref['ef_construction']}$, \textit{{ef\_s}}$={ref['ef_search']}$).",
        "e2e", "|l|c|l|c|c|c|c|c|",
        ["Model", "Device", "Text", "Encode (ms)", "p95 (ms)", "Search (ms)", "Total (ms)", "Encode share"], rows)


def familias(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "index_families_targets.csv")
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("latency_ms")):
            rows.append((label(r["model"]), r["family"], r["target"],
                         rf"\multicolumn{{4}}{{c|}}{{not reached (best {r['max_recall@10']:.3f})}}"))
        else:
            rows.append((label(r["model"]), r["family"], r["target"], f"{r['recall@10']:.3f}",
                         f"{r['latency_ms']:.2f}", f"{r['build_time_s']:.1f}", f"{r['index_mb']:.0f}"))
    return latex_table(
        r"Fastest option per index family and Recall@10 target (latency per query, one thread).",
        "familias", "|l|l|c|c|c|c|c|",
        ["Model", "Family", "Target", "R@10", "Lat. (ms)", "Build (s)", "MB"], rows)


def modelos(ref):
    df = pd.read_csv(cfg.RESULTS_DIR / "model_comparison.csv")
    rows = [(r["label"], r["domain"].capitalize(), "Yes" if r["retrieval_trained"] else "No", int(r["dim"]),
             f"{r['fidelity_r10']:.3f}", f"{r['search_ms']:.2f}", f"{r['title_r1']:.3f}", f"{r['title_r10']:.3f}",
             f"{r['title_mrr10']:.3f}", "--" if pd.isna(r["encode_title_cpu_ms"]) else f"{r['encode_title_cpu_ms']:.0f}")
            for _, r in df.iterrows()]
    return latex_table(
        rf"Effect of domain and training objective at the reference configuration ($M={ref['M']}$, "
        rf"\textit{{ef\_c}}$={ref['ef_construction']}$, \textit{{ef\_s}}$={ref['ef_search']}$). Fidelity: abstract$\rightarrow$abstract "
        r"Recall@10 against exact search. Retrieval: title$\rightarrow$abstract with exact search.",
        "modelos", "|l|l|c|c|c|c|c|c|c|c|",
        ["Model", "Domain", "Retrieval training", "Dim.", "Fid. R@10", "Search (ms)", "R@1", "R@10", "MRR@10",
         "Encode (ms)"], rows)


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

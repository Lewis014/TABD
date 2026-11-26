"""
FASE B.3 — Selección de la configuración de referencia y tablas (local)
=======================================================================

Lee results/results.csv y aplica la regla fijada en config.py: entre los puntos de
operación que alcanzan cfg.REFERENCE_TARGET de Recall@10 en todos los modelos de la
rejilla, el de menor latencia. Esa configuración es la que usan 07 a 13.

No hay una configuración "óptima" sin más: depende de qué se quiera minimizar. Por eso
también se marca el frente de Pareto sobre recall, latencia, construcción y memoria, y
se lista para cada objetivo de recall la opción más barata en cada coste.

Salida en results/:
  reference_config.json
  pareto.csv            todos los puntos de operación, con la marca de Pareto
  selection.csv         opción más barata por modelo, objetivo de recall y coste
  tabla_configs.tex     los 18 índices con el ef_search de referencia
  tabla_seleccion.tex   selection.csv en LaTeX
"""

import json
import numpy as np
import pandas as pd

import config as cfg

COSTS = {"latency_ms": "latency", "build_time_s": "build time", "index_mb": "memory"}


def label(model: str) -> str:
    return cfg.ALL_MODELS[model]["label"]


def reference_config(df: pd.DataFrame) -> dict:
    keys = ["M", "ef_construction", "ef_search"]
    wide = df.pivot_table(index=keys, columns="model", values=["recall@10", "latency_ms"])
    feasible = wide[(wide["recall@10"] >= cfg.REFERENCE_TARGET).all(axis=1)]
    if feasible.empty:
        raise SystemExit(f"Ningún punto alcanza Recall@10 >= {cfg.REFERENCE_TARGET} en todos los modelos.")
    best = feasible["latency_ms"].sum(axis=1).idxmin()
    return dict(zip(keys, (int(v) for v in best)))


def pareto_mask(df: pd.DataFrame) -> np.ndarray:
    """True si ningún otro punto es igual o mejor en todo y estrictamente mejor en algo."""
    # Todo como coste: el recall se invierte para minimizarlo igual que el resto
    v = np.column_stack([-df["recall@10"], df["latency_ms"], df["build_time_s"], df["index_mb"]])
    keep = np.ones(len(v), dtype=bool)
    for i in range(len(v)):
        dominated = (v <= v[i]).all(axis=1) & (v < v[i]).any(axis=1)
        keep[i] = not dominated.any()
    return keep


def selection(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in df["model"].unique():
        for target in cfg.RECALL_TARGETS:
            ok = df[(df["model"] == model) & (df["recall@10"] >= target)]
            for cost, name in COSTS.items():
                if ok.empty:
                    rows.append({"model": model, "target": target, "criterion": name})
                else:
                    best = ok.sort_values([cost, "latency_ms"]).iloc[0]
                    rows.append({"model": model, "target": target, "criterion": name, **best.drop("model")})
    return pd.DataFrame(rows)


def table_configs(df: pd.DataFrame, ref: dict) -> str:
    sub = df[df["ef_search"] == ref["ef_search"]].sort_values(["model", "M", "ef_construction"],
                                                              key=lambda c: c if c.name != "model" else c.map(list(cfg.MODELS).index))
    lines = [
        r"\begin{table}[tbp]",
        rf"\caption{{Fidelity and cost of the 18 indexes at \textit{{ef\_search}}$={ref['ef_search']}$ "
        rf"over {cfg.N_BUILDS} builds (recall: mean $\pm$ s.d.; times: median; latency per query, "
        r"one thread).}\label{tab:configs}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{|l|c|c|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Model} & \textbf{$M$} & \textbf{\textit{ef\_c}} & \textbf{R@1} & \textbf{R@5} & "
        r"\textbf{R@10} & \textbf{Lat. (ms)} & \textbf{Build (s)} & \textbf{MB} \\",
        r"\hline",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"{label(r['model'])} & {int(r['M'])} & {int(r['ef_construction'])} & "
            f"{r['recall@1']:.3f} & {r['recall@5']:.3f} & "
            f"{r['recall@10']:.3f}$\\pm${r['recall@10_sd']:.3f} & {r['latency_ms']:.2f} & "
            f"{r['build_time_s']:.1f} & {r['index_mb']:.0f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_selection(sel: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[tbp]",
        r"\caption{Cheapest operating point per model and Recall@10 target, for each cost "
        r"(latency per query, one thread).}\label{tab:seleccion}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{|l|c|l|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Model} & \textbf{Target} & \textbf{Minimizes} & \textbf{$M$} & \textbf{\textit{ef\_c}} & "
        r"\textbf{\textit{ef\_s}} & \textbf{Lat. (ms)} & \textbf{Build (s)} & \textbf{MB} \\",
        r"\hline",
    ]
    for _, r in sel.iterrows():
        if pd.isna(r.get("M")):
            lines.append(f"{label(r['model'])} & {r['target']} & {r['criterion']} & "
                         r"\multicolumn{6}{c|}{not reached} \\")
            continue
        lines.append(
            f"{label(r['model'])} & {r['target']} & {r['criterion']} & {int(r['M'])} & "
            f"{int(r['ef_construction'])} & {int(r['ef_search'])} & {r['latency_ms']:.2f} & "
            f"{r['build_time_s']:.1f} & {r['index_mb']:.0f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    df = pd.read_csv(cfg.results_csv_path())

    ref = reference_config(df)
    cfg.reference_path().write_text(json.dumps(ref, indent=2), encoding="utf-8")

    df["pareto"] = pareto_mask(df)
    df.to_csv(cfg.RESULTS_DIR / "pareto.csv", index=False)

    sel = selection(df)
    sel.to_csv(cfg.RESULTS_DIR / "selection.csv", index=False)

    (cfg.RESULTS_DIR / "tabla_configs.tex").write_text(table_configs(df, ref), encoding="utf-8")
    (cfg.RESULTS_DIR / "tabla_seleccion.tex").write_text(table_selection(sel), encoding="utf-8")

    print(f"Configuración de referencia (Recall@10 >= {cfg.REFERENCE_TARGET}): {ref}\n")
    row = df[(df["M"] == ref["M"]) & (df["ef_construction"] == ref["ef_construction"]) &
             (df["ef_search"] == ref["ef_search"])]
    cols = ["model", "recall@1", "recall@5", "recall@10", "recall@10_sd", "mrr",
            "latency_ms", "latency_p95_ms", "build_time_s", "index_mb"]
    print(row[cols].to_string(index=False))
    print(f"\nPuntos en el frente de Pareto: {int(df['pareto'].sum())} de {len(df)}")
    print(f"Filas con MRR igual a Recall@1: {int((df['mrr'] - df['recall@1']).abs().lt(1e-3).sum())} de {len(df)}")
    print("\n=== Opción más barata por objetivo ===")
    print(sel[["model", "target", "criterion", "M", "ef_construction", "ef_search",
               "recall@10", "latency_ms", "build_time_s", "index_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()

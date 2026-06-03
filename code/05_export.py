"""
FASE B.3 — Exportación de resultados (local)
=============================================

Lee results/results.csv y produce:

  results/tabla_resultados_best.tex
      4 filas (modelo x métrica) con la MEJOR configuración por Recall@10.
      Coincide con la tabla principal del paper (tab:resultados).

  results/tabla_completa.tex
      Las 36 filas, para un apéndice.

Además imprime en consola la mejor configuración global y un pequeño resumen.
"""

import pandas as pd
import config as cfg


def best_per_group(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["model", "metric"])["recall@10"].idxmax()
    return df.loc[idx].sort_values(["model", "metric"]).reset_index(drop=True)


def fmt_model(m: str) -> str:
    return {"scibert": "SciBERT", "e5": "E5-large"}.get(m, m)


def fmt_metric(m: str) -> str:
    return {"cosine": "Coseno", "l2": "L2"}.get(m, m)


def table_best(df: pd.DataFrame) -> str:
    best = best_per_group(df)
    lines = [
        r"\begin{table}[h]",
        r"\caption{Recall@K, MRR y latencia según modelo de embedding y métrica de "
        r"similitud sobre FAISS-HNSW (mejor configuración de $M$ y "
        r"\textit{ef\_construction}).}\label{tab:resultados}",
        r"\centering",
        r"\begin{tabular}{|l|l|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Modelo} & \textbf{Métrica} & \textbf{R@1} & \textbf{R@5} & "
        r"\textbf{R@10} & \textbf{MRR} & \textbf{Lat. (ms)} \\",
        r"\hline",
    ]
    for _, r in best.iterrows():
        lines.append(
            f"{fmt_model(r['model']):8s} & {fmt_metric(r['metric']):6s} & "
            f"{r['recall@1']:.3f} & {r['recall@5']:.3f} & {r['recall@10']:.3f} & "
            f"{r['mrr']:.3f} & {r['latency_ms']:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_full(df: pd.DataFrame) -> str:
    df = df.sort_values(["model", "metric", "M", "ef_construction"])
    lines = [
        r"\begin{table}[h]",
        r"\caption{Resultados completos de las 36 configuraciones.}\label{tab:completa}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{|l|l|c|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Modelo} & \textbf{Métrica} & \textbf{M} & \textbf{ef\_c} & "
        r"\textbf{R@1} & \textbf{R@5} & \textbf{R@10} & \textbf{MRR} & "
        r"\textbf{Lat.(ms)} \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{fmt_model(r['model'])} & {fmt_metric(r['metric'])} & "
            f"{int(r['M'])} & {int(r['ef_construction'])} & "
            f"{r['recall@1']:.3f} & {r['recall@5']:.3f} & {r['recall@10']:.3f} & "
            f"{r['mrr']:.3f} & {r['latency_ms']:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    df = pd.read_csv(cfg.results_csv_path())

    (cfg.RESULTS_DIR / "tabla_resultados_best.tex").write_text(
        table_best(df), encoding="utf-8"
    )
    (cfg.RESULTS_DIR / "tabla_completa.tex").write_text(
        table_full(df), encoding="utf-8"
    )

    print("Tablas LaTeX escritas en results/.\n")
    print("=== Mejor configuración por (modelo, métrica), según Recall@10 ===")
    cols = ["model", "metric", "M", "ef_construction",
            "recall@1", "recall@5", "recall@10", "mrr", "latency_ms"]
    print(best_per_group(df)[cols].to_string(index=False))

    top = df.loc[df["recall@10"].idxmax()]
    print(f"\nMejor global: {fmt_model(top['model'])} / {fmt_metric(top['metric'])} "
          f"M={int(top['M'])} ef_c={int(top['ef_construction'])} "
          f"-> R@10={top['recall@10']:.3f}, MRR={top['mrr']:.3f}, "
          f"latencia={top['latency_ms']:.3f} ms")


if __name__ == "__main__":
    main()

"""
Versión en español de las tablas del artículo, para paper.tex.

Lee cada results/tabla_*.tex (en inglés, de 05_export.py y 14_tables.py) y escribe
results/tabla_*_es.tex traduciendo títulos, encabezados y celdas de texto. Las cifras no
se tocan, salvo el separador de miles, que pasa a espacio fino (5,000 -> 5\\,000).

Cada reemplazo tiene que encontrar su texto: si cambia el inglés de una tabla, el script
se detiene y dice cuál, en vez de dejar una tabla a medio traducir.

Ejecutar después de 05_export.py y 14_tables.py:
    python 15_tables_es.py
"""

import re

import config as cfg

# Reemplazos comunes a todas las tablas: encabezados y abreviaturas
COMMON = [
    (r"\\textbf\{Model\}", r"\\textbf{Modelo}"),
    (r"\\textbf\{Build \(s\)\}", r"\\textbf{Constr. (s)}"),
]

TABLES = {
    "tabla_configs": [
        (r"Fidelity and cost of the 18 indexes at ", "Fidelidad y costo de los 18 índices con "),
        (r" over (\d+) builds \(recall: mean \$\\pm\$ s\.d\.; times: median; latency per query, one thread\)\.",
         r" en \1 construcciones (recall: media $\\pm$ d.e.; tiempos: mediana; latencia por consulta, un hilo)."),
    ],
    "tabla_seleccion": [
        (r"Cheapest operating point per model and Recall@10 target, for each cost \(latency per query, one thread\)\.",
         "Punto de operación más barato por modelo y objetivo de Recall@10, para cada costo (latencia por "
         "consulta, un hilo)."),
        (r"\\textbf\{Target\}", r"\\textbf{Objetivo}"),
        (r"\\textbf\{Minimizes\}", r"\\textbf{Minimiza}"),
        (r"& latency &", "& latencia &"),
        (r"& build time &", "& construcción &"),
        (r"& memory &", "& memoria &"),
    ],
    "tabla_ef_search": [
        (r"Sensitivity to ", "Sensibilidad a "),
        (r" over (\d+) builds \(latency per query, one thread\)\.",
         r" en \1 construcciones (latencia por consulta, un hilo)."),
    ],
    "tabla_naturales": [
        (r"Known-item retrieval title\$\\rightarrow\$abstract over 1\\,000 titles, with exact search and with HNSW",
         r"Recuperación \\textit{known-item} título$\\rightarrow$abstract con 1\\,000 títulos, con búsqueda exacta "
         "y con HNSW"),
        (r"Similarities are between the title and its own abstract, and between the title and the top-ranked "
         r"document\.",
         "Similitudes entre el título y su propio abstract, y entre el título y el primer documento recuperado."),
        (r"\\textbf\{Exact search\}", r"\\textbf{Búsqueda exacta}"),
        (r"\\textbf\{Similarity\}", r"\\textbf{Similitud}"),
        (r"\\textbf\{Median\}", r"\\textbf{Rango}"),
        (r"\\textbf\{Own\}", r"\\textbf{Propio}"),
        (r"\\textbf\{rank\}", r"\\textbf{mediano}"),
    ],
    "tabla_modelos": [
        (r"Domain against training objective at the reference configuration",
         "Dominio frente a objetivo de entrenamiento en la configuración de referencia"),
        (r"Fidelity: abstract\$\\rightarrow\$abstract Recall@10 of HNSW against exact search, and search latency "
         r"on one thread\. Retrieval: title\$\\rightarrow\$abstract with exact search\.",
         r"Fidelidad: Recall@10 abstract$\\rightarrow$abstract de HNSW frente a la búsqueda exacta, y latencia de "
         r"búsqueda con un hilo. Recuperación: título$\\rightarrow$abstract con búsqueda exacta."),
        (r"\\textbf\{Fidelity\}", r"\\textbf{Fidelidad}"),
        (r"\\textbf\{Retrieval\}", r"\\textbf{Recuperación}"),
        (r"\\textbf\{Domain\}", r"\\textbf{Dominio}"),
        (r"\\textbf\{Training\}", r"\\textbf{Entrenamiento}"),
        (r"& Scientific &", "& Científico &"),
        (r"& General &", "& General &"),
        (r"& None &", "& Ninguno &"),
        (r"& Citations &", "& Citas &"),
        (r"& Query--passage &", "& Consulta--pasaje &"),
    ],
    "tabla_significancia": [
        (r"Paired tests at the reference configuration, Holm-adjusted over (\d+) tests\. Fidelity: "
         r"abstract\$\\rightarrow\$abstract against each model's exact search; Title: title\$\\rightarrow\$abstract "
         r"with exact search\. \$n\$: queries with a difference; \$r\$: rank-biserial correlation; CI: 95\\,\\% "
         r"paired bootstrap of the mean difference; \$b\$, \$c\$: discordant queries \(range over the builds\)\.",
         r"Pruebas pareadas en la configuración de referencia, con corrección de Holm sobre \1 pruebas. Fidelidad: "
         r"abstract$\\rightarrow$abstract frente a la búsqueda exacta de cada modelo; Título: "
         r"título$\\rightarrow$abstract con búsqueda exacta. $n$: consultas con diferencia; $r$: correlación "
         r"biserial de rangos; IC: \\textit{bootstrap} pareado al 95\\,\\% de la diferencia media; $b$, $c$: "
         r"consultas discordantes (rango entre construcciones)."),
        (r"\\textbf\{Comparison\}", r"\\textbf{Comparación}"),
        (r"\\textbf\{Test\}", r"\\textbf{Prueba}"),
        (r"\\textbf\{Means\}", r"\\textbf{Medias}"),
        (r"\\textbf\{Detail\}", r"\\textbf{Detalle}"),
        (r"Fidelity, ", "Fidelidad "),
        (r"Title, ", "Título "),
        (r"CI \[", "IC ["),
    ],
    "tabla_e2e": [
        (r"End-to-end latency per query on the CPU, in ms, over (\d+) texts encoded one at a time with (\d+) "
         r"threads, followed by HNSW search on one thread",
         r"Latencia de extremo a extremo por consulta en la CPU, en ms, sobre \1 textos vectorizados uno a uno "
         r"con \2 hilos, seguida de la búsqueda HNSW con un hilo"),
        (r"Encoding takes ([\d.]+--[\d.]+)\\,\\% of the total\.", r"Vectorizar ocupa el \1\\,\\% del total."),
        (r"\\textbf\{Title\}", r"\\textbf{Título}"),
        (r"\\textbf\{Encode\}", r"\\textbf{Vect.}"),
        (r"\\textbf\{Search\}", r"\\textbf{Búsq.}"),
    ],
    "tabla_familias": [
        (r"Index families on the same queries and ground truth \(one thread, one query at a time\)\. Best R@10: "
         r"highest Recall@10 over the search parameter; then the latency of the fastest setting reaching each "
         r"target \(--: not reached\)\. Build time includes training\.",
         "Familias de índices con las mismas consultas y la misma verdad de referencia (un hilo, una consulta a "
         "la vez). Mejor R@10: el mayor Recall@10 al barrer el parámetro de búsqueda; después, la latencia del "
         "ajuste más rápido que alcanza cada objetivo (--: no se alcanza). La construcción, en segundos, "
         "incluye el entrenamiento."),
        (r"\\textbf\{Family\}", r"\\textbf{Familia}"),
        (r"\\textbf\{Build \(s\)\}", r"\\textbf{Constr.}"),
        (r"\\textbf\{Best R@10\}", r"\\textbf{Mejor R@10}"),
        (r"^Exact &", "Exacta &"),
    ],
    "tabla_escalabilidad": [
        (r"Latency against corpus size \(", "Latencia según el tamaño del corpus ("),
        (r"Exact and HNSW latency in ms, one thread and one query at a time; HNSW uses the smallest "
         r"\\textit\{ef\\_search\} reaching each Recall@10 target\. Gain: speedup over exact search; Batch: the "
         r"same with all queries together on (\d+) threads\. Build time \(s, (\d+) threads\) and memory \(MB\) of "
         r"the index\.",
         r"Latencia exacta y de HNSW en ms, un hilo y una consulta a la vez; HNSW usa el menor "
         r"\\textit{ef\\_search} que alcanza cada objetivo de Recall@10. Ventaja: aceleración frente a la búsqueda "
         r"exacta; Lote: lo mismo con todas las consultas juntas en \1 hilos. Tiempo de construcción (s, \2 "
         r"hilos) y memoria (MB) del índice."),
        (r"\\textbf\{Exact\}", r"\\textbf{Exacta}"),
        (r"\\textbf\{Gain\}", r"\\textbf{Ventaja}"),
        (r"\\textbf\{Batch\}", r"\\textbf{Lote}"),
        (r"\\textbf\{Build\}", r"\\textbf{Constr.}"),
    ],
}

THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def translate(name: str, text: str) -> str:
    for pattern, repl in TABLES[name]:
        text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if n == 0:
            raise SystemExit(f"{name}: no aparece el texto de «{pattern[:60]}…»; actualizar 15_tables_es.py")
    for pattern, repl in COMMON:
        text = re.sub(pattern, repl, text)
    return THOUSANDS.sub(r"\\,", text)


def main():
    for name in TABLES:
        source = cfg.RESULTS_DIR / f"{name}.tex"
        if not source.exists():
            print(f"{name}: se omite, no existe")
            continue
        target = cfg.RESULTS_DIR / f"{name}_es.tex"
        target.write_text(translate(name, source.read_text(encoding="utf-8")), encoding="utf-8")
        print(target.name)


if __name__ == "__main__":
    main()

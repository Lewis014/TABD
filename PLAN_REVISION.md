# Plan de actualización del paper 54

**Semantic Search of Scientific Abstracts through HNSW Indexing in Vector Databases**
Ambos revisores: *weak accept*. Archivo base: `paper_en.tex` (idéntico a `Ingles-article.tex`).
Aún no se ha modificado nada del paper ni del código. Nada de este plan usa APIs de pago: el cálculo es local y en Colab.

Las líneas `L…` se refieren a `paper_en.tex`.

---

## 0. Estado (31 de octubre de 2026)

**Hecho:** código corregido en la rama `revision-paper-54`, experimentos locales repetidos, figuras en inglés y notebook de Colab listo.

**Falta, en orden:**
1. Correr `code/colab_embeddings.ipynb` en Colab y copiar los resultados a la PC.
2. Evaluar los modelos nuevos y la escala a 1M.
3. **Corrida final limpia** con `bash code/run_experiments.sh`, sin usar el equipo mientras corre. En la corrida de hoy, otro proceso duplicó la latencia de dos construcciones.
4. Reescribir el artículo y la carta a los revisores.

Estas cifras son las de hoy. El recall será casi igual en la corrida final; las latencias pueden moverse un poco. Los tiempos del benchmark son la mediana de 5 construcciones. Los de escalabilidad son la media de 3 y los de familias de índices salen de una sola construcción; la corrida final deja los de escalabilidad también en mediana.

| Afirmación | Paper enviado | Medición corregida |
|---|---|---|
| Consultas dentro del corpus | 982 de 1 000 (sin declarar) | 0, con 48 018 documentos indexados |
| Peor Recall@10 con ef_search=50 | 0.946 | 0.884 |
| M=32, ef_c=400, ef_search=50, SciBERT / E5 | 0.998 / 0.995 | 0.991 / 0.977 |
| Configuración de referencia | "Óptima": M=32, ef_c=400, ef_search=50 | Menor latencia con R@10 ≥ 0.99 en ambos modelos: M=32, ef_c=100, ef_search=100 |
| En esa configuración, SciBERT / E5 | — | R@10 0.997 / 0.992; 0.66 / 0.98 ms; 11 / 22 s; 153 / 200 MB |
| Coste de subir ef_c de 100 a 400 | "Casi duplica" | ×2.7 a ×3.6 en construcción, por +0.001 a +0.007 de R@10 de media |
| Memoria según M (de 8 a 32) | Sin medir | Solo +5 % a +6 %: dominan los vectores |
| MRR frente a R@1 en fidelidad | "Confirma desde otro ángulo" | Iguales en 108 de 108 puntos |
| Fidelidad SciBERT vs E5 | +0.003, en el nivel del ruido | +0.005, IC 95 % [+0.003, +0.007]; p Holm 7e-9; se distingue del ruido (d.e. 0.0004) |
| Título→abstract R@10, búsqueda exacta | 0.194 vs 0.953 (con HNSW) | 0.173 vs 0.949; rango mediano 259 vs 1; 1 vs 777 pares discordantes |
| Título→abstract R@10, HNSW | Igual que arriba | 0.163 vs 0.944 |
| HNSW frente a exacta, una consulta y 1 hilo | Sin medir | HNSW gana en todo el rango: ×5 a ×28 con R@10 ≥ 0.99 |
| HNSW frente a exacta, en lote con 12 hilos | Cruce cerca de 60 000 | Casi empate hasta 100k: ×0.5 a ×1.1. Explica el "cruce" |
| Pendiente log-log hasta 100k | "Sublineal" y "exponencial" | Exacta 0.95 [0.88, 1.02]; HNSW a R@10 ≥ 0.99, 0.43 [0.27, 0.58] |
| Vectorizar más buscar, título en CPU, SciBERT / E5 | Sin medir | 46 / 202 ms; vectorizar es más del 98 % del tiempo |

**Otras familias de índices, con R@10 ≥ 0.99 (SciBERT / E5):**
- **HNSW:** 0.64 / 1.06 ms.
- **IVF-Flat:** 2.22 / 4.28 ms, pero se construye entre 4 y 6 veces más rápido.
- **IVF-PQ con OPQ:** no llega; se queda en 0.737 / 0.611, a cambio de ocupar muy poca memoria. PQ con producto interno y sin OPQ no pasaba de 0.235 / 0.366, así que esa configuración se descartó por injusta.
- **IVF-PQ con OPQ y reordenamiento:** 1.21 ms en SciBERT; E5 no llega y se queda en 0.983. Ocupa casi lo mismo que HNSW, porque guarda los vectores, y OPQ tarda más de 8 minutos en entrenar.
- **BallTree (árbol exacto):** 78 / 79 ms, más lento que la búsqueda exacta (9 / 17 ms) y con el doble de memoria.
- **Annoy (árbol aproximado):** no evaluado; se declara como limitación.

La construcción de 3 366 s de E5 en `index_families.csv` no vale: incluye 44 minutos de una suspensión del equipo. Se corrige en la corrida final.

---

## 1. Errores del experimento que los revisores no vieron

Salieron al revisar el código para responder a las observaciones. Cambian cifras publicadas, así que van antes que todo lo demás.

| # | Hallazgo | Evidencia | Efecto en el paper |
|---|---|---|---|
| 1 | **Las consultas no están separadas del corpus.** `stratified_sample` termina en `reset_index`, y `subset.drop(index=queries.index)` borra las filas 0–999 en vez de las consultas ([colab_embed.py:92-93](code/colab_embed.py:92)) | 982 de las 1 000 consultas están en el corpus. En el 98.2 % de los casos el vecino exacto top-1 es la propia consulta | Es falso que sean *held-out* (L169-170, L232). R@1 y MRR de 1.000 son un artefacto y R@10 está inflado |
| 2 | **Coseno y L2 corrieron sobre los mismos vectores.** `normalized()` modifica el array original ([04_benchmark.py:29](code/04_benchmark.py:29), [03_groundtruth.py:23](code/03_groundtruth.py:23)). Coseno va primero, así que L2 recibe vectores ya unitarios, aunque el comentario diga "crudos" | Los ground truth de coseno y L2 coinciden al 100 %. Con L2 realmente crudo en SciBERT coincidirían al 82 % | Lo que el revisor 1 llamó redundante es, de hecho, la misma búsqueda. La diferencia entre ambas columnas es solo ruido de construcción |
| 3 | **El ruido es tan grande como las diferencias que se interpretan** | Mismos vectores y misma configuración: \|ΔR@10\| hasta 0.0029 y \|Δlatencia\| hasta 0.23 ms | La brecha SciBERT–E5 publicada (0.003 y 0.07 ms) no se distingue de ese ruido. Las pruebas pareadas solo cubren la variación entre consultas, no entre construcciones |
| 4 | **MRR es igual a R@1** en la evaluación de fidelidad | 36 de 36 filas de `results.csv`. Si HNSW encuentra al vecino verdadero, sale primero; si no, no aparece | "The MRR confirms this from another angle" (L283) es falso: es el mismo número |
| 5 | **Las latencias son de lote con 12 hilos**, no por consulta. Ningún script fija hilos, y el paper no declara hardware, hilos, pooling ni versiones | Ver tabla de latencias abajo | "Under 1 ms" no está medido en condiciones de uso |
| 6 | **Los estratos están mal formados** ([colab_embed.py:58](code/colab_embed.py:58)) | 1 447 estratos, 1 224 con un solo documento. Consultas frente a corpus: math 9.4 % vs 20.0 %, hep-ph 12.0 % vs 6.2 % | L126 ("proportions similar") no se cumple en las consultas |
| 7 | **La estadística tiene fallos** | `rank_biserial` no promedia empates ([08_significance.py:89](code/08_significance.py:89)). La prueba sobre RR se apoya en 1 o 2 pares distintos. Title→abstract no tiene ninguna prueba. Las similitudes 0.733 y 0.878 salen de solo 30 documentos ([09b_verify.py:22](code/09b_verify.py:22)) | "Statistically equivalent" (L293-295) no se sostiene: no significativo no es equivalente |
| 8 | **La escalabilidad no sostiene sus conclusiones** | Una sola corrida y no monótona (exacta: 0.177 ms en 20k, 0.159 ms en 35k). 303 consultas están dentro del pool. Recall cae de 0.999 a 0.987, así que no se compara a recall constante. Solo SciBERT, aunque se recomienda E5 | El cruce en "60 000" (L442) sale de un solo cruce ruidoso |

**Se sostiene:**
- **Title→abstract (0.194 vs 0.953):** los títulos nunca se indexaron, así que la fuga no lo afecta. Es el resultado más fuerte y el que más le interesó al revisor 1. Se vuelve a correr solo porque cambia el corpus.
- **El salto de M=8 a M=16:** se mantiene en la medición provisional.

**No se sostiene:**
- **"Recall@10 > 0.94 en la peor configuración":** baja a cerca de 0.88. El revisor 2 lo elogió (S2), así que hay que explicarlo en la carta.
- **"ef_search = 50 es el óptimo":** probablemente no vale para E5, que se queda en 0.977, y el barrido nunca se hizo con E5.

### Cifras provisionales

Son una sola construcción, sin las consultas en el corpus (48 018 documentos), con ef_search=50. Sirven para dimensionar el cambio; **no son los resultados nuevos**.

| Configuración | SciBERT R@10: publicado → provisional | E5-large R@10: publicado → provisional |
|---|---|---|
| M=8, ef_c=100 | 0.952 → 0.917 | 0.947 → 0.879 |
| M=16, ef_c=200 | 0.991 → 0.975 | 0.986 → 0.952 |
| M=32, ef_c=400 | 0.998 → 0.991 | 0.995 → 0.977 |
| R@1 en M=32, ef_c=400 | 1.000 → 0.998 | 0.999 → 0.992 |

| Latencia en CPU (i5-11400H) | SciBERT | E5-large |
|---|---|---|
| Búsqueda en lote, 12 hilos (como se publicó) | 0.27 ms | 0.44 ms |
| Búsqueda una a una, 1 hilo: media / p95 | 0.42 / 0.56 ms | 0.76 / 1.07 ms |
| **Vectorizar un título** (30 textos) | **45 ms** | **203 ms** |
| **Vectorizar un abstract** (30 textos) | **154 ms** | **703 ms** |

Vectorizar la consulta cuesta entre 100 y 900 veces más que buscarla. Eso responde a la observación del revisor 1 sobre los tiempos y da un resultado nuevo para la discusión: en el uso real, el cuello de botella es el modelo y no el índice.

---

## 2. Decisiones

Salvo que se indique otra cosa, sigo la recomendación de cada una.

| # | Decisión | Opciones | Recomendación |
|---|---|---|---|
| D1 | **Corregir la fuga** | **A.** Quitar las 982 consultas del corpus con los embeddings actuales: quedan 48 018 documentos, sin Colab. **B.** Regenerar todo en Colab con el muestreo corregido: consultas proporcionales, pero más trabajo y cambia la muestra de title→abstract | **A.** Se corrige `colab_embed.py` y se congelan las listas de ids en el repo para que sea reproducible. El paper describe la composición real de las consultas en lugar de decir que son proporcionales |
| D2 | **Coseno y L2** | **A.** Solo coseno: 18 configuraciones y una frase con la equivalencia (\|\|a−b\|\|² = 2 − 2a·b en vectores unitarios). **B.** L2 real sobre vectores crudos: solo cambia SciBERT, porque E5 ya viene normalizado | **A.** Es lo que pide el revisor 1 y el paper no necesita afirmar nada más sobre la métrica. El ruido se mide con construcciones repetidas |
| D3 | **Modelos nuevos** | Añadir los del cuadro de abajo solo con la configuración de referencia, en una tabla. Alternativa mínima: no añadir ninguno y declarar la confusión como limitación | **Añadirlos.** Es la crítica de fondo del revisor 1 y también responde a la poca novedad que señala el revisor 2 |
| D4 | **Tamaño de la escalabilidad** | **A.** Hasta 100k, bien medido, limitando lo que se afirma. **B.** Hasta 1M con E5-large | **A es obligatoria. B si el plazo lo permite** (ver F3) |

Cuadro de modelos para D3. Separa el dominio del objetivo de entrenamiento:

| | Sin entrenamiento para recuperación | Con entrenamiento para similitud o recuperación |
|---|---|---|
| **Científico** | SciBERT (actual) | SciNCL (`malteos/scincl`), SPECTER2 (`allenai/specter2_base` con adaptadores) |
| **General** | BERT (`bert-base-uncased`) | E5-base (`intfloat/e5-base-v2`, 768d), E5-large (actual, 1024d) |

Cada contraste aísla un factor:
- **SciBERT vs SciNCL:** el objetivo de entrenamiento, porque SciNCL parte de los pesos de SciBERT.
- **SciBERT vs BERT:** el dominio.
- **E5-base vs E5-large:** la dimensión.
- **SPECTER2:** tiene un adaptador propio para consultas cortas, el análogo directo del prefijo `query:` de E5.

Cuidados en este experimento:
- **Fuga por título:** SciNCL y SPECTER2 esperan `título [SEP] abstract`, pero indexar el título filtraría la respuesta en title→abstract. Todos se indexan **solo con el abstract**, y se declara esa desviación.
- **Dimensión de SciNCL y SPECTER2:** las fichas no la indican. Se confirma al cargarlos.
- **Librería `adapters`:** puede chocar con la versión de `transformers`. Si no instala, queda SciNCL, y el revisor pidió "SPECTER2 **o** SciNCL".

**Límite de páginas:** 20; el paper tiene 16 hoy. **Orden acordado:** primero el código, los experimentos y las figuras; el artículo se actualiza al final. **Dato que falta:** fecha límite de la entrega.

---

## 3. Fases

### F0. Preparar
- Crear la rama `revision-paper-54` y guardar la línea base: `paper_en.tex` y el cambio pendiente de `app.py`.
- `paper_en.tex` pasa a ser el único original. `paper.tex` deja de mantenerse.

### F1. Corregir el código y repetir (CPU local)
Correcciones:
- `normalized()` devuelve una copia.
- `major_category` y el `drop` de `colab_embed.py`.
- `rank_biserial` con rangos promediados.
- Ids congelados en `code/splits/`.

Nuevo `04_benchmark.py`:
- 18 configuraciones, solo coseno.
- **5 construcciones por configuración** con orden de inserción barajado, reportando media y desviación.
- Latencia de una consulta a la vez con 1 hilo, con media y p95.
- Tiempo de construcción con hilos fijos.
- **Tamaño del índice en bytes**.

Regla de selección, que sustituye a "la mejor por R@10":
- Frente de Pareto sobre recall, latencia, construcción y memoria.
- Configuración de referencia: la de menor latencia que alcance Recall@10 ≥ 0.99. Es un objetivo convencional, fijado antes de correr la rejilla completa, aunque ya se habían visto las cifras provisionales de la sección 1.

Se ejecuta en este orden, porque 07–10 tienen la mejor configuración escrita a mano: `03 → 04 → 05 → elegir referencia → 07 → 08 → 09 → 10`.
- **07:** barrido de ef_search en ambos modelos y todas las M.
- **08:** informar W, p, efecto, número de pares discordantes e IC por bootstrap, con corrección de Holm. Añadir **la prueba de title→abstract**.
- **09:** guardar resultados por consulta, añadir búsqueda exacta para separar el error del índice del error del modelo, y calcular las similitudes sobre las 1 000 muestras.
- **10:** consultas fuera de todos los subconjuntos, 1 hilo, repeticiones, y comparación a recall constante (ajustando ef_search).

Todas las tablas se generan desde CSV y entran con `\input`, para no copiar cifras a mano.

### F2. Mediciones nuevas (CPU local)
- `11_end_to_end_latency.py`: vectorizar más buscar, por modelo, para títulos y abstracts, en CPU y en GPU si hay.
- `12_index_families.py`: IVF-Flat e IVF-PQ en FAISS, y Annoy como índice de árbol si instala en Windows. Curva recall-latencia, memoria y construcción con las mismas consultas.

### F3. Colab (GPU T4)
- Vectorizar los modelos nuevos de D3 por id. El texto de las 18 consultas que faltan en `corpus_text.parquet` se trae del dataset.
- Si se elige D4-B: piloto de 10k para medir el rendimiento y luego 1M con E5-large, guardando por tramos por si Colab se corta. Tamaños: 10k, 30k, 100k, 300k y 1M, con ajuste log-log e IC de la pendiente. Las 15.8 GB de RAM alcanzan: unos 4 GB de vectores más el grafo.

### F4. Figuras (todas tienen texto en español)
- **Fig. 1:** redibujar en inglés (solo existe el PNG). Añadir la rama de consulta por título y los modelos nuevos.
- **Fig. 2:** está desactualizada: muestra SciBERT, ef_search=6040 y la interfaz en español, mientras el texto dice E5. Pasar `app.py` a inglés y volver a capturar.
- **Fig. 3 a 7:** traducir `06_plots.py`, `07b`, `09c` y `10b`. Quitar el `ef\_search` mal escapado y la anotación "1.4× más rápido". Figuras 3 y 4 sin las series de L2.

### F5. Reescribir el texto
Editar `paper_en.tex` directamente, **sin volver a traducir**: los errores vienen del paso por traducción automática de `8c41bb0`. La última versión en español escrita a mano es `ce4fa6e:paper.tex`, y sirve de referencia para lo que se quiso decir.

Orden: Métodos → Resultados → **Discusión → Conclusiones** → Introducción → Resumen. Aplicar las secciones 4 y 5 de este plan.

Presupuesto de páginas:
- **Entra:** tabla de modelos, tabla de costo y memoria, figura de familias de índices.
- **Sale:** filas y series de L2, la subsección *Data Flow* (repite lo anterior), y la doble descripción de title→abstract (L206 y L361-365).

### F6. Carta y cierre
- **Carta a los revisores:** cada observación, qué se cambió y dónde. **Declara la fuga y sus consecuencias**: el texto enviado afirmaba consultas separadas y las cifras cambian.
- **Revisión final:** cotejar cada cifra del texto con los CSV, confirmar que no queda español en figuras ni tablas, compilar sin referencias sin definir y comprobar el límite de páginas.

---

## 4. Observaciones de los revisores

| Rev. | Observación | Acción | Fase |
|---|---|---|---|
| R1 | Coseno vs L2 es redundante | D2; explicar la equivalencia en una frase | F1, F5 |
| R1 | SciBERT vs E5 mezcla dominio y entrenamiento | D3; reescribir L136 y L388-395 sin atribuir la causa más allá de lo medido | F3, F5 |
| R1 | Pruebas estadísticas mal descritas | Definir criterio binario, unidad pareada, manejo de ceros y corrección; informar todos los valores | F1, F5 |
| R1 | Escalabilidad insuficiente | D4; quitar el cruce en 60 000 salvo que resista la nueva medición | F1, F3, F5 |
| R1 | Tiempos sin vectorizar la consulta | Latencia de extremo a extremo | F2, F5 |
| R1 | "ANN" como red neuronal | L86 y L99: *approximate nearest neighbor* | F5 |
| R1 | "HNSW database" | L93: *HNSW index* | F5 |
| R1 | "the images" | L226: *embeddings* | F5 |
| R1 | "construction of the response element" | L285: `ef_construction` | F5 |
| R1 | "exponentially" | L463: eliminar y decir lo que se midió | F5 |
| R1 | Marco completo "without additional parameterizations" | L480-482: eliminar | F5 |
| R1 | `ef_construction` medio sin función de optimización | L476-480: derivarlo de la regla de selección o quitarlo | F1, F5 |
| R1 | "Óptima" elegida solo por Recall@10 | Frente de Pareto; quitar "optimal" en L244, L344, L468 y L474 | F1, F5 |
| R1 | Discusión antes de conclusiones | Intercambiar L452 y L485 | F5 |
| R2 | Poca novedad (W1) | Plantear como aporte fidelidad vs recuperación y el cuadro 2×2, sin exagerar | F5 |
| R2 | Solo HNSW (W2) | Comparar con IVF, PQ y, si se puede, árboles | F2, F5 |
| R2 | Resumen de parámetros en la Sección 3 | Ampliar la Tabla 1 con fase, función y efecto de M, `ef_construction` y `ef_search` | F5 |
| R2 | "not static" | L86: *not constant* | F5 |
| R2 | "This depends on similarity measures" | L88: usar la redacción que propone | F5 |
| R2 | "The true weakness of HNSW…" | L105: el original decía "no está en consultar, sino en construir". Pasa a *index construction* | F5 |
| R2 | Fig. 1 en español | Redibujar, junto con las demás figuras | F4 |
| R1/R2 | Relevancia para América Latina | Ninguna acción: no es una corrección pedida | — |

---

## 5. Otros errores de texto que nadie señaló

| Línea | Problema | Corrección |
|---|---|---|
| L53-60 | Cifras antiguas; "despite its lower dimensionality" no tiene lógica | Reescribir el resumen al final |
| L93 | "minimizes overestimation"; `$K$` y `$MRR$` mal formados | Reformular según `ce4fa6e` |
| L99 | "input and output rates" | *in-degree and out-degree* |
| L102 | "criticized the development cost of LSH-APG" | Zhao et al. **atacan** el costo de construcción **con** LSH-APG |
| L105 | "representation" dos veces; `[9]`, `[15]`, `[7]` sin `\cite` | *construction*; usar `\cite` |
| L105 | Cifras de H-DMax mezcladas: 1.27× no corresponde a λ=1.10, y SIFT1M es el dataset | 1.27× con λ=1.20 y pérdida de 1.03 %; 1.10× con λ=1.10 y pérdida de 0.27 % |
| L105 | El 3.6× se atribuye a PQ [15] de 2011, pero está en [9] (HNSW-PQ). "85 %" no aparece literal en el PDF de [9] | Citar [9] para las medidas y [15] para el método. Localizar el 85 % o quitarlo |
| L107 | "unnormalized LRR was 0.116" | L2 sin normalizar: R@10 = 0.373 y MRR = 0.116 |
| L128 | `descriptor` | `title` (y `category` es en realidad `categories`) |
| L134 | "representations of ideas", "ease of memorization" | Reescribir |
| L136 | "this study answers it" | Es justo lo que el revisor 1 objeta |
| Métodos | No se indican ids de modelos, pooling, hardware, hilos ni versiones | Añadirlos |
| L171-178 | Descripción confusa del índice exacto y de "two stages" | Reescribir, indicando que las consultas se excluyen del corpus |
| L204 | "signed Wilcoxon rank-sum" mezcla dos pruebas distintas | *Wilcoxon signed-rank* |
| L213 | "distributed computing infrastructure" | Son dos entornos: Colab y una CPU local |
| L226 | "T4 block", "link vectors"; falta el punto final | *T4 GPU*, *embedding vectors* |
| L262 | "Let's first analyze the most impressive results" | Tono formal |
| L283 | MRR entre 0.997 y 1.000 (el mínimo era 0.996) | Se elimina con el hallazgo 4 |
| L285 | "nearly doubles" (el dato es ×2.9; el resumen dice "triples"); "a few percentage points" (son 0.5) | Corregir con las cifras nuevas |
| L287 | "cosine transform" dos veces; "10-point recall rate"; "thousandths of the noise" | Se reduce a la frase de equivalencia |
| L364 | "the abstract is derived from its title" | Al revés: el documento relevante es el abstract del artículo al que pertenece el título |
| L458 | "probabilistic analysis of extreme scenarios" | No corresponde a nada de lo que se hizo; eliminar |
| L466 | "extensively documented" contradice el vacío que plantea la introducción | Eliminar |
| L488 | "compares HNSW against itself" | Lo compara con la búsqueda exacta en el mismo espacio |
| L490 | BEIR, TREC-COVID, NFCorpus y SciFact sin citar | Citar o generalizar |
| Bibliografía | [4], [9] y [11] incompletas; faltan SciNCL, SPECTER2 y las referencias nuevas | Completar y verificar cada entrada |

El título se queda como está, aunque FAISS sea una librería y no una base de datos, porque el cambio de título no se pidió. En el texto se le llama librería.

---

## 6. Si el plazo es corto

| Prioridad | Qué entra |
|---|---|
| **Obligatorio** | F0, F1, la latencia de extremo a extremo y la memoria, F4, F5, F6. Sin esto, el paper conserva cifras que sabemos incorrectas o deja sin responder observaciones directas |
| **Muy recomendable** | Modelos nuevos (D3). Si no entran, se declara la confusión como limitación y las conclusiones no atribuyen la causa |
| **Si hay tiempo** | Familias de índices y escala a 1M. Si no entran, W2 se trata como limitación y solo se afirma lo medido hasta 100k |

La corrección de la fuga no es recortable.

---

## 7. Entregables

- `paper_en.tex` y su PDF actualizados.
- Código corregido, resultados regenerados e ids congelados.
- Las 7 figuras en inglés.
- Carta de respuesta a los revisores.

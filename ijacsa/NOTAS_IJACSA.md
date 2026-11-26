# Paper 54 → IJACSA (SAI): notas de adaptación

## Archivos

| Archivo | Qué es |
|---|---|
| `54_ijacsa.tex` | Fuente única. Por defecto compila la **versión anónima** para revisión doble ciego. |
| `54_ijacsa_final.tex` | Envoltorio de 3 líneas: define `\FINALVERSION` y carga `54_ijacsa.tex` → **versión final** con autores. |
| `IEEEtran.cls` | Clase incluida en la plantilla oficial de SAI (`SAI_Paper_Format_Latex`, descargada de thesai.org). |
| `54_ijacsa.pdf` / `54_ijacsa_final.pdf` | PDF compilados (10 págs. cada uno; 1,1 MB). |

Las figuras se leen de `../Images/` (las mismas que generan los scripts de `code/`).

Compilar (tres pasadas, desde `ijacsa/`):

```
pdflatex 54_ijacsa.tex        # versión anónima (la que se sube)
pdflatex 54_ijacsa_final.tex  # versión final (NO subir durante la revisión)
```

## Formato aplicado (plantilla oficial SAI)

- `IEEEtran` conference, carta, dos columnas; encabezado "(IJACSA) … Vol. XXX, No. XXX, 2026" y pie "www.ijacsa.thesai.org | n | P a g e" tal como vienen en la plantilla (la revista rellena volumen y número).
- Secciones con números romanos y subsecciones con letras: I Introduction, II Related Work, III Methodology, IV Results, V Discussion, VI Conclusion and Future Work. Después, sin numerar: Supplementary Material, Declarations y References.
- Abstract de **258 palabras**, sin fórmulas, símbolos ni caracteres especiales; tampoco hay `@`, `%`, `≥` ni guiones bajos. La plantilla pide "aprox. 150–250" y el formulario de envío "250–300 recomendadas"; 258 cumple ambas.
- **8 keywords**, separadas por punto y coma como en la plantilla.
- Título sin abreviaturas: "HNSW" se escribe completo, como pide la plantilla para títulos y encabezados.
- Referencias en estilo IEEE, numeradas [1]…[38] en orden de primera aparición (comprobado con un script). Los títulos van en *sentence case* (regla de la plantilla) y llevan DOI cuando existe.
- Tablas con `booktabs` y título en versalitas encima; leyendas de figura debajo.
- Declaración de uso de IA generativa, exigida por IJACSA (*Declarations*).

Hay tres desviaciones menores respecto a la plantilla, todas comentadas en el `.tex`:
- Se quitó `flushend`: junto con `\balance` rompía la última página.
- El equilibrado de columnas se hace con `\IEEEtriggeratref{36}`. **Hay que revisarlo si cambia el texto.**
- Se ajustaron los parámetros de colocación de flotantes. Márgenes, columnas y fuentes no cambian.

El aviso `\headheight is too small` viene de la propia plantilla (20 pt para un encabezado de dos líneas) y se deja igual.

## Extensión

- Cuerpo (Introducción → Conclusión): páginas 1–8. El límite de IJACSA es de 10 páginas para el cuerpo, **sin contar** referencias, tablas ni figuras.
- Total con declaraciones y referencias: 10 páginas.

## Qué se sintetizó respecto a `54.tex` (LNCS, 19 págs.)

Se conservan todos los hallazgos y todas las cifras; ninguna se recalculó.

| En 54.tex | En la versión IJACSA |
|---|---|
| Tabla "naturales" + tabla "modelos" | Una sola Tabla IV (fidelidad, recuperación título→abstract, HNSW, similitudes, rango mediano). |
| Tabla de significancia (24 filas, 1 test por fila) | Tabla V a doble columna: una fila por par y panel, con los 24 tests, *n*, *r*, IC 95 %, *b*/*c* y *p* Holm. |
| Dos tablas de escalabilidad | Una sola Tabla VIII con paneles SciBERT / E5-base. |
| Tabla de selección (12 filas) | Resumida en el texto de IV-B (puntos más rápidos, más baratos y más pequeños por objetivo). Puede ir como material suplementario si se quiere. |
| Figura de la demo en Streamlit | Eliminada; la demo se describe en una frase (III-G). |
| Tabla configs (R@1, R@5, R@10 de 18 índices) | Se mantiene completa (Tabla III). |

Se añadió: las ecuaciones (2) Recall@K y (3) MRR@10 (antes solo descritas en prosa), el párrafo de organización del artículo, el apartado "Limitations" dentro de Discussion y la conexión de los resultados con los trabajos de IJACSA en la Discusión.

## Referencias de IJACSA añadidas (9, verificadas)

Cada una se comprobó en su página oficial de thesai.org, en el PDF (páginas y contenido citado) y en Crossref (el DOI resuelve con los mismos metadatos).

| # | Referencia | DOI | Dónde se cita y para qué |
|---|---|---|---|
| [3] | Rastogi, Verma, Kumar (2021), *Query expansion based on word embeddings and ontologies…*, 12(11), 367–373 | 10.14569/IJACSA.2021.0121142 | Intro: el desajuste de vocabulario de la búsqueda por palabras clave. |
| [5] | Ajallouda, Saissi, Zellou (2025), *Embedding models: A comprehensive review…*, 16(10), 539–548 | 10.14569/IJACSA.2025.0161056 | Intro (embeddings), II-C y V-A: BERT/RoBERTa rinden poco en recuperación. |
| [12] | Muludi et al. (2024), *Retrieval-augmented generation approach…*, 15(3), 776–785 | 10.14569/IJACSA.2024.0150379 | Intro, II-B y V-A: RAG con FAISS. |
| [13] | Huan, Zhou (2024), *Integrating advanced language models and vector database…*, 15(6), 1–6 | 10.14569/IJACSA.2024.0150601 | Intro, II-B y V-A: almacén vectorial con FAISS. |
| [14] | Ibrihich et al. (2026), *A hybrid system … RAG for Moroccan court rulings*, 17(9), 200–209 | 10.14569/IJACSA.2026.0170920 | Intro, II-B y V-B: E5 + `IndexFlatIP` exacto + BM25 (justifica el índice exacto en colecciones pequeñas). |
| [15] | Shuo, Affendey, Sidi (2024), *Content-based image retrieval using transfer learning and vector database*, 15(9), 836–844 | 10.14569/IJACSA.2024.0150985 | Intro, II-B y V-A: Milvus IVF_FLAT; reconocen la pérdida de exactitud de la búsqueda aproximada. |
| [26] | Saouabe et al. (2024), *Large-scale image indexing and retrieval methods: A PRISMA-based review*, 15(7), 325–337 | 10.14569/IJACSA.2024.0150732 | II-A: IVF, PQ y *hashing*; escalabilidad como reto abierto. |
| [28] | Alsuhaibani (2023), *Deep learning-based sentence embeddings using BERT…*, 14(8), 997–1004 | 10.14569/IJACSA.2023.01408108 | II-C: la calidad de los vectores de BERT depende de la capa. |
| [31] | Abdelazim, Tharwat, Mohamed (2023), *Semantic embeddings for Arabic RAG (ARAG)*, 14(11), 1328–1334 | 10.14569/IJACSA.2023.01411135 | II-C y V-A: E5 es el mejor de 10 modelos (Recall@10 > 0,90). |

Se descartaron otros candidatos porque su contenido no respaldaba la cita: SEARCHX (índice invertido TF-IDF, sin vectores), *Mining trend analytics* (publicaciones de redes sociales, no literatura científica) y *AI Lawyer* (no detalla índice ni modelo de embeddings).

Las 29 referencias originales se conservan. Sus DOI se validaron en Crossref; las páginas de SBERT y SciBERT siguen las de ACL Anthology, que difieren en 2 de las de Crossref.

## Checklist de anonimato (versión `54_ijacsa.pdf`)

Comprobado automáticamente en el texto extraído y en los metadatos del PDF; no aparece ninguno de estos datos:

- [x] Nombres de autores, afiliación (UNSA, Arequipa, Perú), correos y ORCID.
- [x] Enlace al repositorio (`github.com/Lewis014/TABD`). El texto dice que el material se publicará tras la aceptación.
- [x] Metadatos del PDF: Author, Subject, Keywords y Creator vacíos.
- [x] Agradecimientos y financiación retirados ("withheld for double-blind review").
- [x] Sin autocitas ni referencias al envío anterior del artículo.
- [x] Las figuras PNG solo llevan la etiqueta "Software: Matplotlib".

## Qué se restaura en la versión final (`54_ijacsa_final.tex`)

Ya incluido y activado automáticamente con `\FINALVERSION`:

1. Autores: Santiago Callocondo-Garay, Piero Mejia-Ramos y Antonio Arroyo-Paz.
2. Afiliación: Escuela Profesional de Ingeniería de Sistemas, Universidad Nacional de San Agustín, Arequipa, Perú.
3. Nota de autor de correspondencia: puesta en el primer autor (scallocondo@unsa.edu.pe). **Confirmar quién es.**
4. Enlace al repositorio en *Supplementary Material* y en *Data availability*.
5. Metadatos de autoría del PDF.

Lo que deben completar los autores antes de la versión final:

6. **Funding**: hoy dice `[To be completed by the authors.]`.
7. **Acknowledgment**: hay un bloque comentado en el `.tex`; activarlo solo si hace falta.
8. **ORCID**: la plantilla SAI no los pone en el artículo; van en el formulario de envío (0009-0007-2794-5097, 0009-0006-3011-0640 y 0000-0003-1838-4527).
9. Vol./No. del encabezado: lo asigna la revista.

## Decisiones pendientes de los autores (antes de enviar)

- **Envío previo.** Según `PLAN_REVISION.md`, este trabajo pasó una revisión en otro venue (formato LNCS, dos *weak accept*). IJACSA exige declarar que el manuscrito no está enviado ni publicado en otro lugar, y no admite más de un 25 % de material ya publicado por los mismos autores. **Retiren formalmente el envío anterior** (o confirmen que no sigue en curso) antes de subirlo a IJACSA.
- **Declaración de IA generativa.** El borrador menciona el uso de Claude para adaptar a la plantilla, condensar y verificar referencias. Si se usaron otras herramientas (código, redacción en inglés, traducción), hay que añadirlas.
- **Conflicto de intereses.** El texto dice "The authors declare no conflict of interest"; confirmarlo.
- **Material para revisores (opcional).** Si quieren que los revisores vean el código ya, pueden crear una copia anónima en https://anonymous.4open.science a partir del repo y cambiar la frase de *Supplementary Material* de la versión ciega por ese enlace. Antes, revisen que el repo no tenga nombres en commits, README ni rutas.
- **Envío.** El portal solo acepta PDF (máx. 5 MB): subir `54_ijacsa.pdf`. Hay que elegir área temática principal y secundaria, y el tipo de artículo es "Research Paper".

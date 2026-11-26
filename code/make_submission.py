"""
Arma el paquete de envío del paper 54 en ../envio_54/:
  54.tex   paper_en.tex con las tablas de results/ incrustadas y las figuras sin carpeta
  *.png    las figuras que usa
  llncs.cls
  54.pdf   compilado desde esa carpeta con pdflatex, para comprobar que compila sola
  54.zip   las fuentes (54.tex, llncs.cls y las figuras)

Ejecutar de nuevo si cambia paper_en.tex, una tabla o una figura:
    python make_submission.py
"""

import re
import shutil
import subprocess
import zipfile

import config as cfg

PAPER = cfg.ROOT.parent / "paper_en.tex"
OUT = cfg.ROOT.parent / "envio_54"
NAME = "54"

# Los comentarios del original están en español; el envío va en inglés
COMMENTS = {
    "% Plantilla LLNCS para Springer Computer Science Proceedings":
        "% LLNCS template for Springer Computer Science Proceedings",
    "% Parámetros de flotado más permisivos para empaquetar figuras/tablas y evitar huecos":
        "% More permissive float parameters, to pack figures and tables without gaps",
}


def inline_tables(text: str) -> str:
    def table(match):
        return (cfg.ROOT.parent / match.group(1)).read_text(encoding="utf-8").strip()
    return re.sub(r"\\input\{(code/results/[^}]+\.tex)\}", table, text)


def main():
    OUT.mkdir(exist_ok=True)
    text = inline_tables(PAPER.read_text(encoding="utf-8"))
    for old, new in COMMENTS.items():
        assert old in text, f"No aparece el comentario «{old}»"
        text = text.replace(old, new)

    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{Images/([^}]+)\}", text)
    text = text.replace("{Images/", "{")
    assert "\\input{" not in text and "Images/" not in text
    (OUT / f"{NAME}.tex").write_text(text, encoding="utf-8")

    for fig in figures:
        shutil.copy2(cfg.ROOT.parent / "Images" / fig, OUT / fig)
    cls = subprocess.run(["kpsewhich", "llncs.cls"], capture_output=True, text=True, check=True).stdout.strip()
    shutil.copy2(cls, OUT / "llncs.cls")

    # Tres pasadas: referencias cruzadas y números de página estables
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{NAME}.tex"],
                       cwd=OUT, capture_output=True, check=True)
    log = (OUT / f"{NAME}.log").read_text(encoding="latin-1")
    problems = [line for line in log.splitlines()
                if re.search(r"Overfull|undefined|Warning: (Reference|Citation)|^! ", line)]
    for ext in ("aux", "log", "out"):
        (OUT / f"{NAME}.{ext}").unlink(missing_ok=True)

    sources = [f"{NAME}.tex", "llncs.cls", *figures]
    with zipfile.ZipFile(OUT / f"{NAME}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in sources:
            z.write(OUT / name, name)

    print(f"Paquete en {OUT}: {len(figures)} figuras, {len(sources)} archivos en {NAME}.zip")
    print("Avisos de LaTeX:" if problems else "LaTeX sin avisos de desborde ni referencias sin definir.")
    for line in problems:
        print("  ", line)


if __name__ == "__main__":
    main()

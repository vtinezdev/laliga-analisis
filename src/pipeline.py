"""Reproduce el proyecto de principio a fin.

1. Descarga los datos brutos que falten (src/download.py).
2. Ejecuta los notebooks 01 a 05 en orden, guardando sus resultados en el propio notebook.
   Cada uno lee las tablas que exporta el anterior en data/processed/.
3. Ejecuta los tests (tests/).

Uso (desde la raíz del repositorio, con el entorno virtual activado):
    python src/pipeline.py                 # todo
    python src/pipeline.py --desde 04      # solo desde el notebook 04 (los anteriores ya ejecutados)
    python src/pipeline.py --sin-descarga --sin-tests
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("0*.ipynb"))


def ejecutar_notebook(ruta: Path) -> None:
    inicio = time.time()
    nb = nbformat.read(ruta, as_version=4)
    # timeout=None: sin límite de tiempo por celda (el notebook 04 reajusta decenas de modelos)
    NotebookClient(nb, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(ruta.parent)}}).execute()
    nbformat.write(nb, ruta)
    print(f"  ✓ {ruta.name} ({time.time() - inicio:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--desde", default="01", help="prefijo del primer notebook a ejecutar (01-05)")
    parser.add_argument("--sin-descarga", action="store_true")
    parser.add_argument("--sin-tests", action="store_true")
    args = parser.parse_args()
    # En Windows, la salida redirigida a un fichero usa cp1252 y no admite "✓": se fuerza UTF-8
    sys.stdout.reconfigure(encoding="utf-8")

    if not args.sin_descarga:
        print("1. Descarga de datos brutos", flush=True)
        sys.path.insert(0, str(ROOT / "src"))
        import download
        download.main()

    print("2. Notebooks", flush=True)
    for ruta in NOTEBOOKS:
        if ruta.name[:2] >= args.desde:
            ejecutar_notebook(ruta)

    if not args.sin_tests:
        print("3. Tests", flush=True)
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT))


if __name__ == "__main__":
    main()

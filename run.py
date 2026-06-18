"""Lanzador unico de SmartPark Madrid.

Uso:
    py run.py            -> lanza la app (genera datos si faltan)
    py run.py --prepare  -> solo regenera los Parquet
    py run.py --check    -> solo valida que el entorno este listo

Es la forma mas comoda de arrancar el prototipo: comprueba dependencias,
genera los datos preprocesados si no existen y abre la app de Streamlit.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline import load_config


# --------------------------------------------------------------------------- #
# Helpers visuales
# --------------------------------------------------------------------------- #

def info(msg: str) -> None:
    print(f"  ℹ  {msg}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def header(msg: str) -> None:
    print(f"\n{'─' * 60}\n  {msg}\n{'─' * 60}")


# --------------------------------------------------------------------------- #
# Comprobaciones
# --------------------------------------------------------------------------- #

REQUIRED_PACKAGES = [
    "pandas", "numpy", "pyarrow", "openpyxl",
    "folium", "streamlit", "streamlit_folium", "matplotlib", "yaml",
]


def check_dependencies() -> bool:
    """Verifica que todas las dependencias esten instaladas."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        warn(f"Faltan dependencias: {', '.join(missing)}")
        info("Instala con:  py -m pip install -r requirements.txt")
        return False
    ok("Dependencias instaladas")
    return True


def check_raw_data(cfg) -> bool:
    """Verifica que existan los archivos crudos (CSV + Excel)."""
    csv_path = cfg.csv_tickets
    xlsx_path = cfg.xlsx_parquimetros
    missing = []
    if not csv_path.exists():
        missing.append(csv_path.name)
    if not xlsx_path.exists():
        missing.append(xlsx_path.name)
    if missing:
        warn("Faltan archivos crudos (descargalos a la raiz del proyecto):")
        for m in missing:
            print(f"      - {m}")
        return False
    ok("Archivos crudos presentes")
    return True


def check_processed_data(cfg) -> bool:
    """Verifica que existan los Parquet generados."""
    tickets = cfg.output_path("tickets_m30")
    parqs = cfg.output_path("parquimetros")
    if tickets.exists() and parqs.exists():
        size_mb = tickets.stat().st_size / 1024 / 1024
        ok(f"Datos procesados listos ({size_mb:.0f} MB)")
        return True
    info("Datos procesados no encontrados (se generaran)")
    return False


# --------------------------------------------------------------------------- #
# Acciones
# --------------------------------------------------------------------------- #

def run_prepare() -> int:
    """Ejecuta prepare_data.py."""
    header("Generando datos procesados (≈ 2 min)")
    return subprocess.call([sys.executable, "prepare_data.py"])


def run_streamlit() -> int:
    """Lanza la app Streamlit."""
    header("Lanzando SmartPark Madrid")
    info("La app se abrira en tu navegador (http://localhost:8501)")
    info("Para detenerla pulsa Ctrl+C en esta terminal")
    print()
    try:
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "app.py"]
        )
    except KeyboardInterrupt:
        print()
        ok("App detenida.")
        return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Lanzador de SmartPark Madrid")
    parser.add_argument(
        "--prepare", action="store_true",
        help="Solo regenera los Parquet (no arranca la app)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Solo verifica que el entorno este listo",
    )
    parser.add_argument(
        "--force-prepare", action="store_true",
        help="Fuerza la regeneracion aunque los Parquet ya existan",
    )
    args = parser.parse_args()

    header("SmartPark Madrid · Verificacion del entorno")

    cfg = load_config()
    info(f"Raiz del proyecto: {cfg.project_root}")

    deps_ok = check_dependencies()
    if not deps_ok:
        return 1

    raw_ok = check_raw_data(cfg)
    processed_ok = check_processed_data(cfg)

    if args.check:
        print()
        return 0 if (deps_ok and (raw_ok or processed_ok)) else 1

    # Decidir si hace falta preparar datos
    needs_prepare = args.prepare or args.force_prepare or not processed_ok
    if needs_prepare:
        if not raw_ok:
            warn("No se pueden generar los Parquet sin los archivos crudos.")
            return 1
        rc = run_prepare()
        if rc != 0:
            warn("La preparacion de datos fallo.")
            return rc

    if args.prepare:
        ok("Preparacion completada.")
        return 0

    return run_streamlit()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)

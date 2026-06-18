"""Orquestador del pipeline de preprocesado.

Encadena las tres primeras etapas del pipeline (ingest -> clean ->
transform) y persiste los datasets que consumen el motor y la app.

Uso:
    py prepare_data.py

Este script no contiene logica de negocio: es un mero "driver" que llama
a las funciones del paquete `pipeline`. Toda la configuracion (rutas,
filtro M-30, columnas requeridas, etc.) se lee de `config.yaml`.
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from pipeline import load_config
from pipeline.clean import (
    clean_parquimetros,
    filter_tickets_m30,
    normalize_ticket_types,
)
from pipeline.ingest import iter_tickets_chunks, read_parquimetros
from pipeline.transform import (
    add_derived_columns,
    build_aggregates,
    write_parquet,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_parquimetros() -> None:
    cfg = load_config()
    log("Etapa: parquimetros")
    raw = read_parquimetros(cfg)
    log(f"  Crudos: {len(raw):,}")
    clean = clean_parquimetros(raw, cfg)
    log(f"  Activos en M-30: {len(clean):,}")
    write_parquet(clean, cfg, "parquimetros")
    log(f"  -> {cfg.output_path('parquimetros')}")


def stage_tickets() -> pd.DataFrame:
    cfg = load_config()
    log(f"Etapa: tickets ({cfg.csv_tickets.name})")

    if not cfg.csv_tickets.exists():
        log(f"ERROR: no encuentro {cfg.csv_tickets}")
        sys.exit(1)

    chunks: list[pd.DataFrame] = []
    total_in = total_out = 0
    for i, chunk in enumerate(iter_tickets_chunks(cfg)):
        total_in += len(chunk)
        chunk = filter_tickets_m30(chunk, cfg)
        if chunk.empty:
            continue
        chunk = normalize_ticket_types(chunk)
        chunk = add_derived_columns(chunk)
        chunks.append(chunk)
        total_out += len(chunk)
        log(f"  chunk {i + 1}: leidas={total_in:,} | retenidas M-30={total_out:,}")

    if not chunks:
        log("ERROR: ningun ticket retenido. Revisa el filtro o el CSV.")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    log(f"Total tickets M-30: {len(df):,}")
    write_parquet(df, cfg, "tickets_m30")
    log(f"  -> {cfg.output_path('tickets_m30')}")
    return df


def stage_aggregates(df: pd.DataFrame) -> None:
    cfg = load_config()
    log("Etapa: agregados")
    write_parquet(build_aggregates(df), cfg, "agregados")
    log(f"  -> {cfg.output_path('agregados')}")


def main() -> None:
    t0 = time.time()
    stage_parquimetros()
    df = stage_tickets()
    stage_aggregates(df)
    log(f"Pipeline completo en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

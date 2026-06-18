"""Etapa 1 del pipeline: ingesta de datos crudos.

Funciones puras que se limitan a leer las fuentes originales tal como las
publica el Ayuntamiento de Madrid, sin filtrar ni transformar. La separacion
con `clean.py` y `transform.py` permite cambiar la fuente sin tocar el resto
del pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from pipeline.config import Config


def read_parquimetros(cfg: Config) -> pd.DataFrame:
    """Lee el inventario completo de parquimetros (Excel)."""
    return pd.read_excel(cfg.xlsx_parquimetros)


def iter_tickets_chunks(cfg: Config) -> Iterator[pd.DataFrame]:
    """Itera el CSV de tickets (~1.4 GB) en chunks del tamano configurado.

    Carga unicamente las columnas declaradas en `config.yaml` y aplica dtypes
    string para minimizar el coste de parseo. Las conversiones a datetime y
    float se realizan en `clean.py` despues del filtrado geografico.
    """
    if not cfg.csv_tickets.exists():
        raise FileNotFoundError(
            f"No encuentro el CSV de tickets en {cfg.csv_tickets}.\n"
            "Descargalo de datos.madrid.es y colocalo en la raiz del proyecto."
        )

    str_cols = [
        "matricula_parquimetro", "distrito", "barrio", "tipo_zona",
        "distintivo", "cod_distrito", "cod_barrio",
    ]
    dtypes = {c: "string" for c in str_cols}

    yield from pd.read_csv(
        cfg.csv_tickets,
        sep=cfg.csv_separator,
        usecols=cfg.required_columns,
        dtype=dtypes,
        chunksize=cfg.chunk_size,
        low_memory=False,
    )

"""Etapa 3 del pipeline: variables derivadas y serializacion.

Genera las columnas que el modelo de scoring y la app necesitan (`hora`,
`fecha`, `dia_semana`) y persiste el resultado en formato Parquet, mas
eficiente que CSV para consultas analiticas (~10x reduccion de tamano y
lectura columnar).
"""

from __future__ import annotations

import pandas as pd

from pipeline.config import Config


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Anade hora del dia, fecha (str) y dia de la semana en ingles."""
    out = df.copy()
    out["hora"] = out["fecha_inicio"].dt.hour.astype("Int8")
    out["fecha"] = out["fecha_inicio"].dt.date.astype("string")
    out["dia_semana"] = out["fecha_inicio"].dt.day_name().astype("string")
    return out


def build_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen academico de cardinalidades del dataset preprocesado."""
    return pd.DataFrame(
        {
            "metric": ["filas", "distritos", "barrios", "fechas", "parquimetros"],
            "value": [
                len(df),
                df["distrito"].nunique(),
                df["barrio"].nunique(),
                df["fecha"].nunique(),
                df["matricula_parquimetro"].nunique(),
            ],
        }
    )


def write_parquet(df: pd.DataFrame, cfg: Config, output_key: str) -> None:
    """Persiste un DataFrame en `data/<output_key>` segun config.yaml."""
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_path(output_key)
    df.to_parquet(path, index=False)

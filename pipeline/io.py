"""Capa de I/O sobre los Parquet ya procesados.

Funciones de solo lectura usadas por la app y el motor para consumir los
datos preprocesados. Se separan de `transform.py` (escritura) para
respetar el principio de read/write segregation.
"""

from __future__ import annotations

import pandas as pd

from pipeline.config import Config, load_config


def cargar_parquimetros(cfg: Config | None = None) -> pd.DataFrame:
    """Carga el inventario de parquimetros ya filtrado a M-30."""
    cfg = cfg or load_config()
    df = pd.read_parquet(cfg.output_path("parquimetros"))
    # Saneamiento de codificacion (Ñ rota en dataset original)
    if "calle" in df.columns and cfg.text_fixes:
        s = df["calle"].astype("string")
        for bad, good in cfg.text_fixes.items():
            s = s.str.replace(bad, good, regex=False)
        df["calle"] = s
    return df


def cargar_historico(cfg: Config | None = None) -> pd.DataFrame:
    """Carga el historico completo de tickets M-30 (1T 2025)."""
    cfg = cfg or load_config()
    return pd.read_parquet(cfg.output_path("tickets_m30"))


def cargar_tickets_fecha(
    fecha: pd.Timestamp,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Carga los tickets cuyo `fecha_inicio` cae en `fecha`.

    Usa filtros pushdown de Parquet para evitar leer todo el historico en
    memoria; esto permite a la app cambiar de dia sin penalizacion.
    """
    cfg = cfg or load_config()
    fecha = pd.Timestamp(fecha).normalize()
    siguiente = fecha + pd.Timedelta(days=1)
    return pd.read_parquet(
        cfg.output_path("tickets_m30"),
        filters=[
            ("fecha_inicio", ">=", fecha),
            ("fecha_inicio", "<", siguiente),
        ],
    )


def fechas_disponibles(cfg: Config | None = None) -> list[pd.Timestamp]:
    """Devuelve la lista ordenada de dias presentes en el historico."""
    cfg = cfg or load_config()
    s = pd.read_parquet(cfg.output_path("tickets_m30"), columns=["fecha"])["fecha"]
    return sorted(pd.to_datetime(s.dropna().unique()))

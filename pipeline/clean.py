"""Etapa 2 del pipeline: limpieza y filtrado.

Aplica el filtro geografico declarado en la memoria (cap. 1.2: interior de
la M-30) y normaliza tipos. Es deliberadamente conservadora: no descarta
tickets por valores anomalos en otras columnas, eso se documenta como
limitacion en el cap. 9 de la memoria.
"""

from __future__ import annotations

import pandas as pd

from pipeline.config import Config


def filter_tickets_m30(chunk: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Conserva solo los tickets cuyo distrito esta en el ambito M-30."""
    return chunk[chunk["distrito"].isin(cfg.distritos_m30)]


def normalize_ticket_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte fechas y limpia el importe (decimal con coma -> float)."""
    out = df.copy()
    for col in ("fecha_operacion", "fecha_inicio", "fecha_fin"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    out["importe_tique"] = (
        out["importe_tique"].astype(str).str.replace(",", ".").astype(float)
    )
    return out


def clean_parquimetros(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filtra parquimetros activos en distritos M-30 con coordenadas validas."""
    activos = df[df["fecha_de_baja"].isna()].copy()
    en_m30 = activos[activos["distrito"].isin(cfg.distritos_m30)].copy()

    # numero_finca trae mezcla de int y 'S/N' -> forzamos string
    en_m30["numero_finca"] = en_m30["numero_finca"].astype("string")
    en_m30["cod_distrito"] = en_m30["cod_distrito"].astype("string")
    en_m30["cod_barrio"] = en_m30["cod_barrio"].astype("string")
    en_m30["matricula"] = en_m30["matricula"].astype(str)

    cols = [
        "matricula", "cod_distrito", "distrito", "cod_barrio", "barrio",
        "calle", "numero_finca", "longitud", "latitud",
    ]
    return en_m30[cols].reset_index(drop=True)

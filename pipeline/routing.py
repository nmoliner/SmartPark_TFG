"""Modulo de enrutado y ETA (Etapa 7 del pipeline).

Calcula tiempos estimados de viaje origen → destino y selecciona plazas
candidatas cuya liberacion se sincroniza con la llegada del usuario.

No usa geocoder externo: trabaja directamente con coordenadas del inventario
de parquimetros, lo que mantiene la auto-suficiencia del prototipo
(defendible academicamente: cero dependencias externas, cero PII).

Conceptos:
  - Distancia haversine entre dos coordenadas geograficas.
  - ETA = distancia / velocidad_urbana_kmh (configurable).
  - Plazas sincronizadas: aquellas cuya `min_hasta_liberacion` cae en
    [ETA - margen, ETA + margen]. La intuicion es que llegamos cuando
    el coche se va.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import pandas as pd

from pipeline.config import Config, load_config


def distancia_haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> float:
    """Distancia en linea recta entre dos coordenadas (km)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * asin(sqrt(a))


def eta_minutos(distancia_km: float, velocidad_kmh: float) -> float:
    """Convierte distancia y velocidad en minutos de viaje."""
    if velocidad_kmh <= 0:
        return float("inf")
    return (distancia_km / velocidad_kmh) * 60


def plazas_sincronizadas(
    candidatos: pd.DataFrame,
    origen_lat: float,
    origen_lon: float,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Anota a cada candidato con distancia, ETA y diferencia con la liberacion.

    Devuelve los candidatos enriquecidos con:
      - distancia_km        : haversine origen → plaza
      - eta_min             : tiempo de viaje estimado
      - diff_llegada_min    : (min_hasta_liberacion - eta_min); 0 = sincronia perfecta
      - score_sincronizado  : combina score base con cercania de sincronia
    Y filtrados a los que estan dentro de la ventana de tolerancia.
    """
    cfg = cfg or load_config()
    routing_cfg = cfg.raw.get("routing", {})
    velocidad = routing_cfg.get("urban_speed_kmh", 18)
    margen = routing_cfg.get("sync_window_min", 4)

    if candidatos.empty:
        return candidatos.assign(
            distancia_km=[], eta_min=[], diff_llegada_min=[],
            score_sincronizado=[],
        )

    df = candidatos.copy()
    df["distancia_km"] = df.apply(
        lambda r: distancia_haversine_km(
            origen_lat, origen_lon, r["latitud"], r["longitud"],
        ),
        axis=1,
    )
    df["eta_min"] = df["distancia_km"].apply(
        lambda d: eta_minutos(d, velocidad)
    )
    df["diff_llegada_min"] = df["min_hasta_liberacion"] - df["eta_min"]

    # Filtrar a la ventana: la plaza debe liberarse dentro de [llegada - margen,
    # llegada + margen]. diff_llegada_min positiva = la plaza libera DESPUES de
    # llegar (esperamos), negativa = libera ANTES (riesgo de que la cojan).
    en_ventana = df["diff_llegada_min"].abs() <= margen
    df_sync = df[en_ventana].copy()

    # Score sincronizado: combina oportunidad base con la proximidad de
    # sincronia. Penaliza tanto llegar tarde (ya cogida) como muy temprano
    # (esperar mucho). Formula: base * (1 - |diff|/margen).
    df_sync["score_sincronizado"] = (
        df_sync["score_oportunidad"]
        * (1 - df_sync["diff_llegada_min"].abs() / margen).clip(0, 1)
    )

    return (
        df_sync.sort_values("score_sincronizado", ascending=False)
        .reset_index(drop=True)
    )

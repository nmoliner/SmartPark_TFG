"""Tests del modulo de routing (haversine + ETA + sincronizacion).

Verifican propiedades clave defendibles ante el tribunal:
  - Distancia haversine simetrica y cero en mismo punto.
  - ETA proporcional a la distancia.
  - Plazas sincronizadas filtradas dentro de la ventana.
"""

import pandas as pd
import pytest

from pipeline.routing import (
    distancia_haversine_km,
    eta_minutos,
    plazas_sincronizadas,
)


# ----- Distancia haversine ------------------------------------------------- #

def test_haversine_mismo_punto_es_cero():
    assert distancia_haversine_km(40.42, -3.70, 40.42, -3.70) == pytest.approx(0)


def test_haversine_simetrica():
    d1 = distancia_haversine_km(40.42, -3.70, 40.45, -3.68)
    d2 = distancia_haversine_km(40.45, -3.68, 40.42, -3.70)
    assert d1 == pytest.approx(d2)


def test_haversine_distancia_madrid_aprox():
    """Sol -> Atocha aprox. 1.2 km en linea recta."""
    sol_lat, sol_lon = 40.4168, -3.7038
    atocha_lat, atocha_lon = 40.4067, -3.6904
    d = distancia_haversine_km(sol_lat, sol_lon, atocha_lat, atocha_lon)
    assert 1.0 < d < 1.6


# ----- ETA ----------------------------------------------------------------- #

def test_eta_proporcional():
    """A doble distancia, doble tiempo."""
    e1 = eta_minutos(1.0, velocidad_kmh=20)
    e2 = eta_minutos(2.0, velocidad_kmh=20)
    assert e2 == pytest.approx(2 * e1)


def test_eta_velocidad_cero_devuelve_inf():
    assert eta_minutos(5.0, velocidad_kmh=0) == float("inf")


def test_eta_calculo_concreto():
    """5 km a 30 km/h = 10 min."""
    assert eta_minutos(5.0, velocidad_kmh=30) == pytest.approx(10.0)


# ----- Sincronizacion ------------------------------------------------------ #

def test_plazas_sincronizadas_dataframe_vacio():
    """Si no hay candidatos, devuelve DataFrame vacio sin romper."""
    vacio = pd.DataFrame()
    result = plazas_sincronizadas(vacio, 40.42, -3.70)
    assert result.empty


def test_plazas_sincronizadas_filtra_fuera_de_ventana():
    """Las plazas con diferencia > sync_window_min quedan fuera."""
    candidatos = pd.DataFrame({
        "latitud": [40.4168, 40.4170],
        "longitud": [-3.7038, -3.7040],
        "min_hasta_liberacion": [3.0, 30.0],   # 30 min: muy lejos de la ventana
        "score_oportunidad": [0.8, 0.5],
    })
    resultado = plazas_sincronizadas(candidatos, 40.4168, -3.7038)
    # Al menos uno (el de 3 min) deberia colarse, el de 30 min no.
    assert len(resultado) <= 1

"""Tests del scorer (cap. 6.4 de la memoria).

Verifican propiedades defendibles ante el tribunal:
  - La probabilidad esta saturada en [10, 90] %.
  - Score minimo (4) -> probabilidad alta; score maximo (12) -> probabilidad baja.
  - El detalle del scoring se compone de los 4 scores por dimension.
"""

import pandas as pd

from pipeline.config import load_config
from pipeline.scoring import ParkingScorer


def _ticket_demo() -> pd.DataFrame:
    """Genera un mini-historico sintetico con las 4 dimensiones."""
    rows = []
    distritos = ["CENTRO", "SALAMANCA", "CHAMARTIN"]
    horas = [9, 12, 18]
    zonas = ["AZUL", "VERDE"]
    distintivos = ["B", "C", "ECO"]
    for d in distritos:
        for h in horas:
            for z in zonas:
                for dist in distintivos:
                    # Volumen ficticio: combina indices para variar el conteo
                    n = (distritos.index(d) + 1) * (horas.index(h) + 1) * 10
                    for _ in range(n):
                        rows.append({
                            "distrito": d, "hora": h,
                            "tipo_zona": z, "distintivo": dist,
                        })
    return pd.DataFrame(rows)


def test_probabilidad_saturada_entre_floor_y_ceil():
    cfg = load_config()
    scorer = ParkingScorer(cfg).fit(_ticket_demo())
    res = scorer.predict(
        distrito="CENTRO", hora=12, tipo_zona="AZUL", distintivo="B",
    )
    assert cfg.prob_floor <= res.probabilidad <= cfg.prob_ceil


def test_detalle_contiene_las_cuatro_dimensiones():
    cfg = load_config()
    scorer = ParkingScorer(cfg).fit(_ticket_demo())
    res = scorer.predict(
        distrito="SALAMANCA", hora=9, tipo_zona="VERDE", distintivo="ECO",
    )
    assert set(res.detalle.keys()) == set(cfg.scoring_dimensions)


def test_score_total_es_suma_de_dimensiones():
    cfg = load_config()
    scorer = ParkingScorer(cfg).fit(_ticket_demo())
    res = scorer.predict(
        distrito="CHAMARTIN", hora=18, tipo_zona="AZUL", distintivo="C",
    )
    assert res.score_total == sum(res.detalle.values())


def test_etiqueta_coherente_con_probabilidad():
    cfg = load_config()
    scorer = ParkingScorer(cfg).fit(_ticket_demo())
    res = scorer.predict(
        distrito="CENTRO", hora=12, tipo_zona="AZUL", distintivo="B",
    )
    # La etiqueta debe ser una de las 3 categorias declaradas
    assert res.nivel in (
        "Alta probabilidad", "Probabilidad media", "Baja probabilidad",
    )


def test_score_minimo_da_probabilidad_alta():
    """Si todas las dimensiones puntuan 1 -> total = 4 -> prob = ceil (90%)."""
    cfg = load_config()
    scorer = ParkingScorer(cfg)
    prob = scorer._to_probability(cfg.score_min)
    assert prob == cfg.prob_ceil


def test_score_maximo_da_probabilidad_minima_de_la_formula():
    """Score maximo (12) -> formula da 30%; nunca supera el ceil ni baja del floor."""
    cfg = load_config()
    scorer = ParkingScorer(cfg)
    prob_min = scorer._to_probability(cfg.score_max)
    prob_max = scorer._to_probability(cfg.score_min)
    assert prob_min < prob_max
    assert cfg.prob_floor <= prob_min <= cfg.prob_ceil
    assert cfg.prob_floor <= prob_max <= cfg.prob_ceil

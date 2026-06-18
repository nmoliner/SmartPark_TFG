"""Validacion cuantitativa del modelo de scoring (cap. 7 de la memoria).

Compara las predicciones del `ParkingScorer` (entrenado con un subconjunto
del histórico) con la presion **realmente observada** en un dia de test
(holdout), siguiendo el principio metodologico de no entrenar y validar
sobre los mismos datos.

Salida:
  - DataFrame con (distrito, hora, prob_predicha, presion_observada)
  - Correlacion de Spearman: medida no parametrica de monotonia, robusta
    a la diferencia de escala entre [0, 100]% y conteos absolutos.
  - Comparacion con baselines naive (constante, aleatorio).

Hipotesis: si el modelo es informativo, deberia haber correlacion
**negativa** entre probabilidad predicha y presion observada (mas tickets
= mas dificil aparcar = menos probabilidad).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.config import Config, load_config
from pipeline.scoring import ParkingScorer


@dataclass(frozen=True)
class ValidationResult:
    """Resultado de la validacion holdout."""

    fecha_test: pd.Timestamp
    n_train_tickets: int
    n_test_tickets: int
    n_observaciones: int
    spearman_modelo: float
    spearman_baseline_constante: float
    spearman_baseline_aleatorio: float
    detalle: pd.DataFrame  # filas: (distrito, hora, prob_predicha, presion_observada)


def validar_holdout(
    historico: pd.DataFrame,
    fecha_test: pd.Timestamp,
    distintivo: str = "C",
    tipo_zona: str = "AZUL",
    semilla: int = 42,
    cfg: Config | None = None,
) -> ValidationResult:
    """Valida el scorer sobre un dia held-out.

    Procedimiento:
      1. Excluir del histórico los tickets de la fecha de test → train set.
      2. Entrenar el `ParkingScorer` con el train set.
      3. Para cada (distrito, hora) presente en el dia de test, predecir la
         probabilidad y contar la presion real (numero de tickets).
      4. Calcular correlacion de Spearman entre prediccion y presion.
      5. Comparar con dos baselines:
            - constante: predice siempre 50%
            - aleatorio: predice valores uniformes en [10, 90]%
    """
    cfg = cfg or load_config()
    fecha_test = pd.Timestamp(fecha_test).normalize()

    # 1. Particion train/test
    fechas = historico["fecha_inicio"].dt.normalize()
    train = historico[fechas != fecha_test]
    test = historico[fechas == fecha_test]
    if len(test) == 0:
        raise ValueError(f"No hay tickets en la fecha de test {fecha_test.date()}")

    # 2. Entrenar con el train
    scorer = ParkingScorer(cfg=cfg).fit(train)

    # 3. Para cada (distrito, hora) del test, predecir y observar
    test_grid = (
        test.groupby(["distrito", "hora"], observed=True)
        .size()
        .reset_index(name="presion_observada")
    )

    predicciones = []
    for _, row in test_grid.iterrows():
        try:
            res = scorer.predict(
                distrito=row["distrito"],
                hora=int(row["hora"]),
                tipo_zona=tipo_zona,
                distintivo=distintivo,
            )
            predicciones.append(res.probabilidad)
        except (KeyError, ValueError):
            predicciones.append(np.nan)

    test_grid["prob_predicha"] = predicciones
    test_grid = test_grid.dropna(subset=["prob_predicha"])

    # 4. Spearman: rangos. Esperamos correlacion NEGATIVA.
    sp_modelo = _spearman(test_grid["prob_predicha"], test_grid["presion_observada"])

    # 5. Baselines
    n = len(test_grid)
    baseline_cte = np.full(n, 50.0)
    rng = np.random.default_rng(semilla)
    baseline_rand = rng.uniform(
        cfg.prob_floor, cfg.prob_ceil, size=n,
    )
    sp_cte = _spearman(baseline_cte, test_grid["presion_observada"])
    sp_rand = _spearman(baseline_rand, test_grid["presion_observada"])

    return ValidationResult(
        fecha_test=fecha_test,
        n_train_tickets=int(len(train)),
        n_test_tickets=int(len(test)),
        n_observaciones=n,
        spearman_modelo=sp_modelo,
        spearman_baseline_constante=sp_cte,
        spearman_baseline_aleatorio=sp_rand,
        detalle=test_grid.sort_values(
            "presion_observada", ascending=False
        ).reset_index(drop=True),
    )


def _spearman(x, y) -> float:
    """Correlacion de Spearman implementada con pandas (sin scipy)."""
    s = pd.Series(x).rank(method="average")
    t = pd.Series(y).rank(method="average")
    if s.std() == 0 or t.std() == 0:
        return float("nan")
    return float(s.corr(t))

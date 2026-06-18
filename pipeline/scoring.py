"""Etapa 4 del pipeline: modelo de probabilidad estimada.

Implementa el `ParkingScorer` descrito en el cap. 6.4 de la memoria.

Decisiones metodologicas (todas configurables via config.yaml):
  - Segmentacion en terciles (Baja/Media/Alta) sobre cuatro dimensiones.
  - Scoring lineal aditivo (1, 2, 3) -> rango teorico [4, 12].
  - Mapeo lineal saturado a probabilidad [10, 90] %.
  - Etiqueta cualitativa con umbrales 65 % / 40 %.

Justificacion: ante la ausencia de ground truth de ocupacion real, se opta
por un modelo no supervisado, interpretable y conservador, frente a un
clasificador supervisado que generaria un razonamiento circular si se
entrenara sobre el propio numero de tickets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.config import Config, load_config


@dataclass
class ScoringResult:
    """Resultado de una consulta al scorer."""

    probabilidad: float
    nivel: str
    score_total: int
    detalle: dict[str, int]


class ParkingScorer:
    """Modelo de probabilidad estimada de encontrar aparcamiento."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        # DataFrames ajustados, uno por dimension
        self._fits: dict[str, pd.DataFrame] = {}

    # ----- Ajuste -------------------------------------------------------- #

    def _aggregate_by(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Cuenta tickets por valor de `col` y asigna nivel por terciles."""
        s = df.groupby(col, observed=True).size().reset_index(name="num_tickets")
        try:
            s["nivel"] = pd.qcut(
                s["num_tickets"], q=3,
                labels=list(self.cfg.score_map.keys()),
                duplicates="drop",
            )
        except ValueError:
            # Si hay menos de 3 categorias o todos los valores son iguales
            s["nivel"] = "Media"
        s["nivel"] = s["nivel"].astype(str)
        s["score"] = s["nivel"].map(self.cfg.score_map).astype(int)
        return s

    def fit(self, df_tickets: pd.DataFrame) -> "ParkingScorer":
        """Calcula los terciles de presion sobre el historico."""
        for dim in self.cfg.scoring_dimensions:
            self._fits[dim] = self._aggregate_by(df_tickets, dim)
        return self

    # ----- Consulta ------------------------------------------------------ #

    def _score_for(self, dim: str, value) -> int:
        """Devuelve el score 1/2/3. Valor desconocido -> nivel medio."""
        df_dim = self._fits.get(dim)
        if df_dim is None:
            return self.cfg.score_map["Media"]
        match = df_dim.loc[df_dim[dim] == value, "score"]
        return int(match.iloc[0]) if not match.empty else self.cfg.score_map["Media"]

    def _to_probability(self, score_total: int) -> float:
        """Mapeo lineal saturado de [score_min, score_max] a [floor, ceil]."""
        rng = self.cfg.score_max - self.cfg.score_min
        prob = 100 - ((score_total - self.cfg.score_min) / rng) * 70
        return float(np.clip(prob, self.cfg.prob_floor, self.cfg.prob_ceil))

    def _label(self, prob: float) -> str:
        if prob >= self.cfg.prob_high:
            return "Alta probabilidad"
        if prob >= self.cfg.prob_medium:
            return "Probabilidad media"
        return "Baja probabilidad"

    def predict(self, **dimension_values) -> ScoringResult:
        """Estima la probabilidad para un escenario.

        Args:
            **dimension_values: Una entrada por cada dimension declarada en
                cfg.scoring_dimensions (p. ej. distrito, hora, tipo_zona,
                distintivo).
        """
        detalle = {
            dim: self._score_for(dim, dimension_values.get(dim))
            for dim in self.cfg.scoring_dimensions
        }
        total = sum(detalle.values())
        prob = self._to_probability(total)
        return ScoringResult(
            probabilidad=prob,
            nivel=self._label(prob),
            score_total=total,
            detalle=detalle,
        )

    def ranking(
        self, ranking_dim: str = "distrito", **fixed_values
    ) -> pd.DataFrame:
        """Devuelve un ranking de valores de `ranking_dim` para el resto fijo."""
        if ranking_dim not in self._fits:
            raise ValueError(f"Dimension '{ranking_dim}' no esta ajustada.")
        rows = []
        for v in self._fits[ranking_dim][ranking_dim]:
            r = self.predict(**{**fixed_values, ranking_dim: v})
            rows.append({
                ranking_dim: v,
                "probabilidad": r.probabilidad,
                "nivel": r.nivel,
                "score_total": r.score_total,
            })
        return (
            pd.DataFrame(rows)
            .sort_values("probabilidad", ascending=False)
            .reset_index(drop=True)
        )

    # ----- Acceso a tablas ajustadas (para la UI) ----------------------- #

    @property
    def por_distrito(self) -> pd.DataFrame:
        return self._fits.get("distrito")

    @property
    def por_hora(self) -> pd.DataFrame:
        return self._fits.get("hora")

    @property
    def por_zona(self) -> pd.DataFrame:
        return self._fits.get("tipo_zona")

    @property
    def por_distintivo(self) -> pd.DataFrame:
        return self._fits.get("distintivo")

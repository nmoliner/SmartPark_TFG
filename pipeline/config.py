"""Carga y validacion de la configuracion del pipeline.

Lee `config.yaml` en la raiz del proyecto y lo expone como un objeto
inmutable accesible desde cualquier modulo del pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Wrapper inmutable sobre el dict de configuracion."""

    raw: dict[str, Any] = field(default_factory=dict)

    # ---- Accesos tipados a las secciones mas usadas -------------------- #

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["processed_dir"]

    @property
    def csv_tickets(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["raw"]["tickets_csv"]

    @property
    def xlsx_parquimetros(self) -> Path:
        return PROJECT_ROOT / self.raw["paths"]["raw"]["parquimetros_xlsx"]

    @property
    def distritos_m30(self) -> set[str]:
        return set(self.raw["ambito"]["distritos_m30"])

    @property
    def chunk_size(self) -> int:
        return int(self.raw["ingest"]["chunk_size"])

    @property
    def csv_separator(self) -> str:
        return self.raw["ingest"]["csv_separator"]

    @property
    def required_columns(self) -> list[str]:
        return list(self.raw["ingest"]["required_columns"])

    # ---- Scoring ------------------------------------------------------- #

    @property
    def score_map(self) -> dict[str, int]:
        return dict(self.raw["scoring"]["level_to_score"])

    @property
    def score_min(self) -> int:
        return int(self.raw["scoring"]["score_min"])

    @property
    def score_max(self) -> int:
        return int(self.raw["scoring"]["score_max"])

    @property
    def prob_floor(self) -> float:
        return float(self.raw["scoring"]["prob_floor"])

    @property
    def prob_ceil(self) -> float:
        return float(self.raw["scoring"]["prob_ceil"])

    @property
    def prob_high(self) -> float:
        return float(self.raw["scoring"]["prob_thresholds"]["high"])

    @property
    def prob_medium(self) -> float:
        return float(self.raw["scoring"]["prob_thresholds"]["medium"])

    @property
    def scoring_dimensions(self) -> list[str]:
        return list(self.raw["scoring"]["dimensions"])

    # ---- Reserva ------------------------------------------------------- #

    @property
    def liberation_window_min(self) -> int:
        return int(self.raw["reservation"]["liberation_window_min"])

    @property
    def renewal_window_min(self) -> int:
        return int(self.raw["reservation"]["renewal_window_min"])

    @property
    def reservation_hold_min(self) -> int:
        return int(self.raw["reservation"]["reservation_hold_min"])

    @property
    def physical_meter_regex(self) -> str:
        return self.raw["reservation"]["physical_meter_regex"]

    # ---- App ----------------------------------------------------------- #

    @property
    def app_options(self) -> dict[str, list[str]]:
        return self.raw["app"]["options"]

    @property
    def app_map(self) -> dict[str, Any]:
        return self.raw["app"]["map"]

    @property
    def text_fixes(self) -> dict[str, str]:
        return self.raw["app"]["text_fixes"]

    # ---- Salidas ------------------------------------------------------- #

    def output_path(self, key: str) -> Path:
        return self.processed_dir / self.raw["paths"]["outputs"][key]


@lru_cache(maxsize=1)
def load_config(path: Path | str | None = None) -> Config:
    """Carga `config.yaml` y lo cachea durante la vida del proceso."""
    p = Path(path) if path else CONFIG_PATH
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(raw=data)

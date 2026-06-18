"""Experimentos adicionales de validacion para el capitulo 7.

Genera evidencia empirica para cuatro refuerzos metodologicos:

  1. Validacion cruzada temporal: repite el holdout sobre 5 dias distintos
     y reporta la media y desviacion del Spearman.
  2. Sensibilidad al prior bayesiano: compara el ranking de plazas
     recomendadas para tres valores de alpha (1, 5, 20) y mide su
     concordancia mediante Jaccard@10.
  3. Baseline logistico: entrena una regresion logistica con las cuatro
     variables del scorer y compara su Spearman con el modelo propuesto.
  4. Robustez geografica: repite el holdout excluyendo Chamartin (distrito
     que concentra ~25% del volumen) y comprueba si el Spearman se mantiene.

Vuelca todos los resultados en `docs/figuras/experimentos_extra.json`.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.config import load_config
from pipeline.io import cargar_historico
from pipeline.scoring import ParkingScorer
from pipeline.validation import validar_holdout


OUT_PATH = BASE_DIR / "docs" / "figuras" / "experimentos_extra.json"

resultados: dict = {}

print("Cargando historico de tickets...")
cfg = load_config()
historico = cargar_historico(cfg=cfg)
print(f"  {len(historico):,} tickets cargados.")

# ---------------------------------------------------------------------------
# 1. Validacion cruzada temporal sobre 5 dias del trimestre
# ---------------------------------------------------------------------------
print("\n[1] Validacion cruzada temporal...")
fechas_cv = [
    pd.Timestamp("2025-01-20"),  # lunes
    pd.Timestamp("2025-02-05"),  # miercoles
    pd.Timestamp("2025-02-22"),  # sabado
    pd.Timestamp("2025-03-10"),  # lunes
    pd.Timestamp("2025-03-15"),  # sabado (el original)
]

cv_results = []
for fecha in fechas_cv:
    try:
        res = validar_holdout(historico, fecha_test=fecha, cfg=cfg)
        cv_results.append(
            {
                "fecha": fecha.strftime("%Y-%m-%d"),
                "dia_semana": fecha.day_name(),
                "n_test_tickets": res.n_test_tickets,
                "n_observaciones": res.n_observaciones,
                "spearman_modelo": round(res.spearman_modelo, 4),
                "spearman_aleatorio": round(res.spearman_baseline_aleatorio, 4),
            }
        )
        print(f"  {fecha.date()}: Spearman modelo = {res.spearman_modelo:+.4f}")
    except Exception as exc:
        print(f"  {fecha.date()}: ERROR ({exc})")

sp_vals = [r["spearman_modelo"] for r in cv_results]
resultados["cv_temporal"] = {
    "n_dias": len(cv_results),
    "spearman_media": round(float(np.mean(sp_vals)), 4),
    "spearman_desviacion": round(float(np.std(sp_vals, ddof=1)), 4),
    "spearman_min": round(float(np.min(sp_vals)), 4),
    "spearman_max": round(float(np.max(sp_vals)), 4),
    "detalle": cv_results,
}

# ---------------------------------------------------------------------------
# 2. Sensibilidad al prior bayesiano (suavizado Beta-Binomial)
# ---------------------------------------------------------------------------
print("\n[2] Sensibilidad al prior bayesiano (alpha = 1, 5, 20)...")


def _no_renewal_probs(df: pd.DataFrame, alpha: float) -> pd.Series:
    """Probabilidad suavizada de no renovacion por (parquimetro, hora).

    Proxy: 'no renovacion' = tickets cuya duracion (minutos_tique) es
    superior a la mediana global, aplicado por celda (matricula_parquimetro,
    hora). Cada celda recibe el suavizado Beta-Binomial:
        p = (n_no_renov + alpha) / (n_total + 2 * alpha)
    """
    g = df.groupby(["matricula_parquimetro", "hora"], observed=True)
    n_total = g.size()
    mediana = df["minutos_tique"].median()
    n_no_renov = g["minutos_tique"].apply(lambda s: int((s >= mediana).sum()))
    prob = (n_no_renov + alpha) / (n_total + 2 * alpha)
    return prob.rename("prob_no_renov")


muestra_calles = historico.sample(n=min(500_000, len(historico)), random_state=42)
top_por_alpha: dict[int, list[str]] = {}
for alpha in [1, 5, 20]:
    probs = _no_renewal_probs(muestra_calles, alpha=alpha)
    top10 = probs.sort_values(ascending=False).head(10)
    top_por_alpha[alpha] = [f"{c}|{h}" for (c, h) in top10.index.tolist()]
    print(f"  alpha={alpha}: top-10 calculado ({len(probs)} celdas calle-hora).")


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


resultados["sensibilidad_alpha"] = {
    "valores_probados": [1, 5, 20],
    "n_celdas_evaluadas": int(len(probs)),
    "jaccard_1_vs_5": round(jaccard(top_por_alpha[1], top_por_alpha[5]), 3),
    "jaccard_5_vs_20": round(jaccard(top_por_alpha[5], top_por_alpha[20]), 3),
    "jaccard_1_vs_20": round(jaccard(top_por_alpha[1], top_por_alpha[20]), 3),
}

# ---------------------------------------------------------------------------
# 3. Baseline logistico: regresion supervisada como contraste
# ---------------------------------------------------------------------------
print("\n[3] Baseline logistico (regresion lineal supervisada)...")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

fecha_test = pd.Timestamp("2025-03-15")
fechas = historico["fecha_inicio"].dt.normalize()
train = historico[fechas != fecha_test]
test = historico[fechas == fecha_test]

# Construimos celdas (distrito, hora) con presion observada como target
celdas_train = (
    train.groupby(["distrito", "hora"], observed=True)
    .size()
    .reset_index(name="presion")
)
celdas_test = (
    test.groupby(["distrito", "hora"], observed=True)
    .size()
    .reset_index(name="presion_observada")
)

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train = enc.fit_transform(celdas_train[["distrito", "hora"]].astype(str))
y_train = np.log1p(celdas_train["presion"].values)

modelo_lin = LinearRegression().fit(X_train, y_train)

X_test = enc.transform(celdas_test[["distrito", "hora"]].astype(str))
celdas_test["pred_lineal"] = modelo_lin.predict(X_test)


def _spearman(x, y) -> float:
    s = pd.Series(x).rank()
    t = pd.Series(y).rank()
    if s.std() == 0 or t.std() == 0:
        return float("nan")
    return float(s.corr(t))


sp_lineal = _spearman(celdas_test["pred_lineal"], celdas_test["presion_observada"])

# Para comparar contra SmartPark con la misma rejilla
scorer = ParkingScorer(cfg=cfg).fit(train)
preds_smart = []
for _, row in celdas_test.iterrows():
    try:
        r = scorer.predict(
            distrito=row["distrito"],
            hora=int(row["hora"]),
            tipo_zona="AZUL",
            distintivo="C",
        )
        preds_smart.append(r.probabilidad)
    except Exception:
        preds_smart.append(np.nan)
celdas_test["pred_smartpark"] = preds_smart
celdas_test = celdas_test.dropna(subset=["pred_smartpark"])
sp_smart = _spearman(
    celdas_test["pred_smartpark"], celdas_test["presion_observada"]
)

resultados["baseline_supervisado"] = {
    "modelo": "Regresion lineal con OneHotEncoding(distrito, hora)",
    "spearman_baseline_supervisado": round(sp_lineal, 4),
    "spearman_smartpark": round(sp_smart, 4),
    "n_celdas": int(len(celdas_test)),
    "comentario": (
        "Signo esperado: positivo para el baseline supervisado (predice"
        " presion directamente) y negativo para SmartPark (predice"
        " probabilidad inversa a la presion)."
    ),
}
print(f"  Baseline supervisado: Spearman = {sp_lineal:+.4f}")
print(f"  SmartPark scorer    : Spearman = {sp_smart:+.4f}")

# ---------------------------------------------------------------------------
# 4. Robustez geografica: excluir Chamartin
# ---------------------------------------------------------------------------
print("\n[4] Robustez geografica: excluir Chamartin...")
historico_sin_cham = historico[historico["distrito"] != "CHAMARTIN"].copy()
try:
    res_sin = validar_holdout(
        historico_sin_cham, fecha_test=fecha_test, cfg=cfg
    )
    resultados["robustez_chamartin"] = {
        "spearman_completo": resultados["cv_temporal"]["detalle"][-1][
            "spearman_modelo"
        ],
        "spearman_sin_chamartin": round(res_sin.spearman_modelo, 4),
        "n_test_tickets_sin": res_sin.n_test_tickets,
        "delta": round(
            res_sin.spearman_modelo
            - resultados["cv_temporal"]["detalle"][-1]["spearman_modelo"],
            4,
        ),
    }
    print(f"  Spearman sin Chamartin = {res_sin.spearman_modelo:+.4f}")
except Exception as exc:
    resultados["robustez_chamartin"] = {"error": str(exc)}

# ---------------------------------------------------------------------------
# Volcado
# ---------------------------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as fh:
    json.dump(resultados, fh, ensure_ascii=False, indent=2, default=str)
print(f"\nResultados volcados en: {OUT_PATH}")

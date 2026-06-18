r"""Exporta a PNG la figura de validación del capítulo 7.

Genera el gráfico de dispersión (scatter) que enfrenta la probabilidad
predicha por el modelo con los tickets realmente vendidos, sobre el día de
holdout principal (15-03-2025), junto con la recta de tendencia. Es la misma
idea que muestra la pestaña Validación de la app, pero en estático para la
memoria.

Uso:
    .\.venv\Scripts\python.exe scripts\exportar_figura_cap7.py

Requisitos: los datasets crudos deben estar en la raíz del proyecto (los mismos
que usa el pipeline).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline import cargar_historico, validar_holdout  # noqa: E402

OUT_DIR = BASE_DIR / "docs" / "figuras"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FECHA_TEST = pd.Timestamp("2025-03-15")

print("Cargando histórico y ejecutando la validación holdout...")
resultado = validar_holdout(cargar_historico(), FECHA_TEST)
detalle = resultado.detalle.copy()

x = detalle["prob_predicha"].to_numpy(dtype=float)
y = detalle["presion_observada"].to_numpy(dtype=float)

print(f"  Spearman modelo: {resultado.spearman_modelo:.3f}")
print(f"  Observaciones:   {resultado.n_observaciones}")

# Recta de tendencia (ajuste lineal).
pendiente, interseccion = np.polyfit(x, y, 1)
x_linea = np.array([x.min(), x.max()])
y_linea = pendiente * x_linea + interseccion

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(x, y, s=70, alpha=0.7, color="#2563eb", edgecolor="white", linewidth=0.5)
ax.plot(x_linea, y_linea, color="#dc2626", linewidth=2.5, label="Tendencia")
ax.set_title("Lo que predijo el modelo frente a los tickets reales (15-03-2025)")
ax.set_xlabel("Probabilidad predicha por el modelo (%)")
ax.set_ylabel("Tickets reales vendidos")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right")

# Anotacion con el Spearman del dia.
ax.text(
    0.03,
    0.04,
    f"Spearman = {resultado.spearman_modelo:.2f}",
    transform=ax.transAxes,
    fontsize=11,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f1f5f9", edgecolor="#94a3b8"),
)

fig.tight_layout()
ruta = OUT_DIR / "cap_07_fig_01_validacion_scatter.png"
fig.savefig(ruta, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  -> {ruta}")

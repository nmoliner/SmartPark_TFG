"""Genera la figura de evolución del tráfico (2021 vs 2025) para el capítulo 6.

Reproduce el gráfico del notebook ``intensidad días laborables 2021-2025.ipynb``
usando los valores reales de intensidad media del interior de la M-30:
    - 2021: 242.420,86
    - 2025: 248.388,89  (+2,46 %)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Valores reales calculados en el notebook de intensidad de tráfico
media_2021 = 242420.86
media_2025 = 248388.89
variacion = (media_2025 - media_2021) / media_2021 * 100  # +2,46 %

anios = ["2021", "2025"]
valores = [media_2021, media_2025]

fig, ax = plt.subplots(figsize=(7, 5))
barras = ax.bar(anios, valores, color=["#9ecae1", "#3182bd"], width=0.55)

ax.set_title(
    "Intensidad media de tráfico en días laborables\nInterior de la M-30",
    fontsize=12,
)
ax.set_ylabel("Intensidad media (vehículos)")
ax.set_ylim(0, max(valores) * 1.12)

# Separador de miles con punto (formato español)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "."))
)

# Etiqueta encima de cada barra
for barra, valor in zip(barras, valores):
    ax.text(
        barra.get_x() + barra.get_width() / 2,
        valor,
        f"{valor:,.0f}".replace(",", "."),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Anotación de la variación porcentual
ax.text(
    0.5,
    0.92,
    f"+{variacion:.2f} %",
    transform=ax.transAxes,
    ha="center",
    fontsize=12,
    color="#08519c",
    fontweight="bold",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

salida = Path(__file__).resolve().parent.parent / "docs" / "figuras"
salida.mkdir(parents=True, exist_ok=True)
destino = salida / "cap_06_fig_00_trafico_2021_2025.png"
fig.savefig(destino, dpi=150, bbox_inches="tight")
print(f"Figura guardada en: {destino}")
print(f"Variacion 2021-2025: +{variacion:.2f} %")

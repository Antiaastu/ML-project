"""Step 2 - Data Collection.

Downloads the 'Energy Efficiency' dataset (UCI Machine Learning Repository, id=242) and saves it as a CSV file,
so every notebook works from exactly the same data. The original columns are called X1..X8 (inputs) and
Y1, Y2 (targets); we give them readable names.
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

dataset = fetch_ucirepo(id=242)
df = dataset.data.features.join(dataset.data.targets)

# X1..X8 / Y1, Y2 are the original codes; these names say what each column means
df = df.rename(columns={
    "X1": "relative_compactness", "X2": "surface_area", "X3": "wall_area", "X4": "roof_area",
    "X5": "overall_height", "X6": "orientation", "X7": "glazing_area", "X8": "glazing_area_distribution",
    "Y1": "heating_load", "Y2": "cooling_load"})

df.to_csv(OUT / "energy_efficiency.csv", index=False)
dataset.variables.to_csv(OUT / "variable_descriptions.csv", index=False)

print("Saved:", OUT / "energy_efficiency.csv")
print("Shape (rows, columns):", df.shape)
print(df.describe().round(2).T)

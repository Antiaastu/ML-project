"""Step 2 - Data Collection.

Downloads the 'Predict Students' Dropout and Academic Success' dataset
(UCI Machine Learning Repository, id=697) and saves it as a CSV file so the
rest of the project can work offline and always uses the same data.
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

dataset = fetch_ucirepo(id=697)

# 'features' = the input columns (X); 'targets' = the thing we want to predict (y)
df = dataset.data.features.copy()
df["Target"] = dataset.data.targets["Target"]

df.to_csv(OUT / "students_dropout.csv", index=False)
dataset.variables.to_csv(OUT / "variable_descriptions.csv", index=False)

print("Saved:", OUT / "students_dropout.csv")
print("Shape (rows, columns):", df.shape)
print(df["Target"].value_counts())

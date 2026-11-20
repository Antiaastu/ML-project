"""Inference helpers shared by Notebook 3 and the Streamlit app.

The app receives the 8 *raw* building inputs. The models were trained on those 8 plus 3 engineered features
(see Notebook 1). `add_engineered` rebuilds them with exactly the same formulas, so training and deployment
can never disagree ("training/serving skew").
"""
import numpy as np
import pandas as pd

TARGETS = ["heating_load", "cooling_load"]
RAW = ["relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height",
       "orientation", "glazing_area", "glazing_area_distribution"]
GEOMETRY = RAW[:5]
ENGINEERED = ["volume", "surface_to_volume", "glazing_m2"]


def add_engineered(df):
    d = df.copy()
    d["volume"] = d["roof_area"] * d["overall_height"]
    d["surface_to_volume"] = d["surface_area"] / d["volume"]
    d["glazing_m2"] = d["glazing_area"] * d["roof_area"]
    return d


def predict_loads(bundle, raw_df, fallback=False):
    """Return a DataFrame with the predicted heating and cooling load and the 90% error band of each.

    fallback=False: the accurate main model, valid for the 12 known building shapes.
    fallback=True : the simple linear model, used for custom geometries (its error band is much wider)."""
    X = add_engineered(raw_df[RAW])[bundle["features"]]
    out = pd.DataFrame(index=raw_df.index)
    for t in TARGETS:
        m = bundle[t]["fallback"] if fallback else bundle[t]
        out[t] = m["pipeline"].predict(X)
        out[f"{t}_band"] = m["error_band_90"]
    return out


def nearest_shape(shapes, geometry):
    """Distance (in standard-deviation units) from a geometry to the closest of the 12 known shapes.
    0 means 'exactly one of the 12 building shapes the models were trained on'."""
    g = np.array([geometry[c] for c in GEOMETRY], dtype=float)
    known = shapes[GEOMETRY].to_numpy(float)
    scale = known.std(axis=0)
    d = np.sqrt((((known - g) / scale) ** 2).sum(axis=1))
    i = int(d.argmin())
    return int(shapes.iloc[i]["shape_id"]), float(d[i])

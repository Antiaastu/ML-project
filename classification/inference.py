"""Inference helpers shared by Notebook 3 and the Streamlit app.

The app receives the 36 *raw* facts about a student. The model was trained on
those 36 plus 6 engineered features (see Notebook 1). `add_engineered` rebuilds
the 6 engineered features with exactly the same formulas, so training and
deployment can never disagree.
"""
import numpy as np
import pandas as pd

CLASSES = ["Dropout", "Enrolled", "Graduate"]          # encoded as 0, 1, 2
ENGINEERED = ["sem1_approval_rate", "sem2_approval_rate", "total_units_approved",
              "total_units_enrolled", "overall_approval_rate", "grade_change"]


def _ratio(num, den):
    """num / den, but 0 when den is 0 (same rule as in Notebook 1)."""
    num, den = pd.Series(num).astype(float), pd.Series(den).astype(float)
    return np.where(den > 0, num / den.where(den > 0, np.nan), 0.0)


def add_engineered(df):
    d = df.copy()
    s1e, s1a = d["curricular_units_1st_sem_enrolled"], d["curricular_units_1st_sem_approved"]
    s2e, s2a = d["curricular_units_2nd_sem_enrolled"], d["curricular_units_2nd_sem_approved"]
    d["sem1_approval_rate"] = _ratio(s1a, s1e)
    d["sem2_approval_rate"] = _ratio(s2a, s2e)
    d["total_units_approved"] = s1a + s2a
    d["total_units_enrolled"] = s1e + s2e
    d["overall_approval_rate"] = _ratio(d["total_units_approved"], d["total_units_enrolled"])
    d["grade_change"] = d["curricular_units_2nd_sem_grade"] - d["curricular_units_1st_sem_grade"]
    return d


def predict_with_threshold(proba, t):
    """Dropout if P(Dropout) >= t, otherwise the more likely of the other two classes."""
    proba = np.asarray(proba)
    pred = proba[:, 1:].argmax(axis=1) + 1
    pred[proba[:, 0] >= t] = 0
    return pred


def predict_students(entry, raw_df):
    """entry: one scenario of the deploy bundle; raw_df: DataFrame with the 36 raw columns.
    Returns (predicted class names, probability matrix)."""
    X = add_engineered(raw_df)[entry["features"]]
    proba = entry["pipeline"].predict_proba(X)
    pred = predict_with_threshold(proba, entry["threshold"])
    return [CLASSES[i] for i in pred], proba

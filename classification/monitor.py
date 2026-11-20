"""Monitoring helpers (ML process step 10): prediction logs, latency, data drift.

Data drift = the students the model sees in real life start to look different
from the students it was trained on. If that happens, the model's accuracy can
silently fall, so we compare live inputs against the training data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

LOG_COLUMNS = ["timestamp", "scenario", "latency_ms", "predicted", "p_dropout", "p_enrolled",
               "p_graduate", "inputs_json"]


def psi(reference, new, bins=10):
    """Population Stability Index. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift."""
    reference, new = np.asarray(reference, float), np.asarray(new, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (almost) constant column - nothing to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


MIN_ROWS_FOR_DRIFT = 100      # PSI is very noisy on fewer new rows


def drift_report(reference_df, new_df, columns):
    """One row per feature: PSI, KS test p-value and a plain-language status.

    A shift is only reported when BOTH measures agree: a high PSI alone is not enough, because with a small
    number of new rows PSI is inflated by pure chance (found by testing the app with 40 replayed students).
    """
    rows = []
    for c in columns:
        if c not in new_df or new_df[c].nunique() < 1:
            continue
        p = psi(reference_df[c], new_df[c])
        pv = float(ks_2samp(reference_df[c], new_df[c]).pvalue)
        if p >= 0.25 and pv < 0.01:
            status = "LARGE shift"
        elif p >= 0.10 and pv < 0.05:
            status = "moderate shift"
        else:
            status = "stable"
        rows.append({"feature": c, "psi": round(p, 3), "ks_p_value": round(pv, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def load_logs(log_dir):
    f = Path(log_dir) / "predictions.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def logged_inputs(logs):
    """Expand the JSON column of the log into one column per input feature."""
    if logs.empty:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(s) for s in logs["inputs_json"]])


def latency_summary(logs):
    if logs.empty:
        return {}
    lat = logs["latency_ms"].astype(float)
    return {"requests": int(len(lat)), "mean_ms": float(lat.mean()),
            "p95_ms": float(lat.quantile(0.95)), "max_ms": float(lat.max())}

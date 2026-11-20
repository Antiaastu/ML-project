"""Monitoring helpers (ML process step 10): prediction logs, latency, data drift, live error.

Data drift = the buildings the model sees in real life start to look different from the buildings it was trained on.
Our inputs are mostly *discrete* (e.g. only 2 heights, 4 orientations), so categorical-style inputs are compared by the
share of each value, and genuinely continuous inputs by their distribution.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

LOG_COLUMNS = ["timestamp", "request_id", "latency_ms", "heating_pred", "cooling_pred", "in_domain", "inputs_json"]
MIN_ROWS_FOR_DRIFT = 100          # drift scores are very noisy on fewer new rows
CATEGORICAL_MAX_VALUES = 12       # inputs with at most this many distinct values are treated as categories


def psi_numeric(reference, new, bins=10):
    """Population Stability Index for a continuous feature (quantile bins of the reference data)."""
    reference, new = np.asarray(reference, float), np.asarray(new, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def psi_categorical(reference, new):
    """PSI computed on the share of each distinct value. Also returns a chi-square p-value for the same comparison."""
    cats = sorted(set(pd.unique(reference)) | set(pd.unique(new)))
    r = np.array([(reference == c).mean() for c in cats]); n = np.array([(new == c).mean() for c in cats])
    r_c, n_c = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    psi = float(np.sum((n_c - r_c) * np.log(n_c / r_c)))
    counts = np.array([(new == c).sum() for c in cats], float)
    expected = r_c / r_c.sum() * counts.sum()
    return psi, float(chisquare(counts, expected).pvalue)


def drift_report(reference_df, new_df, columns):
    """One row per input: PSI, a significance p-value and a plain status.
    A shift is only reported when BOTH the PSI and the significance test agree."""
    rows = []
    for c in columns:
        if c not in new_df or len(new_df) == 0:
            continue
        if reference_df[c].nunique() <= CATEGORICAL_MAX_VALUES:
            p, pv = psi_categorical(reference_df[c].to_numpy(), new_df[c].to_numpy()); kind = "categorical"
        else:
            p, pv = psi_numeric(reference_df[c], new_df[c]), float(ks_2samp(reference_df[c], new_df[c]).pvalue); kind = "numeric"
        status = "LARGE shift" if (p >= 0.25 and pv < 0.01) else ("moderate shift" if (p >= 0.10 and pv < 0.05) else "stable")
        rows.append({"input": c, "type": kind, "psi": round(p, 3), "p_value": round(pv, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def load_logs(log_dir):
    f = Path(log_dir) / "predictions.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame(columns=LOG_COLUMNS)


def logged_inputs(logs):
    return pd.DataFrame([json.loads(s) for s in logs["inputs_json"]]) if len(logs) else pd.DataFrame()


def latency_summary(logs):
    if logs.empty:
        return {}
    lat = logs["latency_ms"].astype(float)
    return {"requests": int(len(lat)), "mean_ms": float(lat.mean()), "p95_ms": float(lat.quantile(0.95)), "max_ms": float(lat.max())}


def live_error(logs, feedback):
    """Compare logged predictions with the actual loads that users reported in the feedback form."""
    if feedback is None or feedback.empty or logs.empty:
        return None
    m = feedback.merge(logs[["request_id", "heating_pred", "cooling_pred"]], on="request_id", how="inner")
    out = {}
    for t, a in [("heating", "actual_heating"), ("cooling", "actual_cooling")]:
        d = m.dropna(subset=[a])
        if len(d):
            err = d[a].astype(float) - d[f"{t}_pred"].astype(float)
            out[t] = {"n": int(len(d)), "MAE": float(err.abs().mean()), "RMSE": float(np.sqrt((err ** 2).mean()))}
    return out or None

"""Building Energy Estimator - Streamlit app (ML process step 8: deployment, step 10: monitoring).

Run:  streamlit run app.py
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import monitor
from inference import GEOMETRY, RAW, TARGETS, nearest_shape, predict_loads

ROOT = Path(__file__).parent
LOG_DIR = Path(os.environ.get("LOG_DIR", ROOT / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Building Energy Estimator", page_icon=":house:", layout="wide")


# ----------------------------------------------------------------------------- loading
@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "deploy_bundle.joblib")


@st.cache_data
def load_assets():
    return json.load(open(ROOT / "models" / "app_assets.json"))


@st.cache_data
def load_reference():
    return pd.read_csv(ROOT / "models" / "reference_sample.csv")


bundle, assets, reference = load_bundle(), load_assets(), load_reference()
shapes = pd.DataFrame(assets["shapes"])
HEAT, COOL = "#EC4259", "#2A7CBD"
NAMES = {"heating_load": "Heating load", "cooling_load": "Cooling load"}
UNIT = "kWh/m² per year"


def shape_label(sid):
    r = shapes[shapes.shape_id == sid].iloc[0]
    return f"Shape {sid}: compactness {r.relative_compactness:.2f}, height {r.overall_height:g} m, roof {r.roof_area:.0f} m²"


def dist_label(v):
    return "No windows (0)" if v == 0 else ("Evenly spread (1)" if v == 1 else f"Concentrated on side {v} ({v})")


def log_prediction(rec):
    f = LOG_DIR / "predictions.csv"
    row = pd.DataFrame([{"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request_id": rec["request_id"],
                         "latency_ms": round(rec["latency"], 2), "heating_pred": round(rec["pred"]["heating_load"], 3),
                         "cooling_pred": round(rec["pred"]["cooling_load"], 3), "in_domain": rec["in_domain"],
                         "inputs_json": json.dumps(rec["inputs"])}])
    row.to_csv(f, mode="a", header=not f.exists(), index=False)


def log_feedback(record):
    f = LOG_DIR / "feedback.csv"
    pd.DataFrame([record]).to_csv(f, mode="a", header=not f.exists(), index=False)


def is_known_shape(inputs):
    """True when the geometry is exactly one of the 12 shapes the accurate model was trained on."""
    return nearest_shape(shapes, {c: inputs[c] for c in GEOMETRY})[1] < 1e-6


def predict_one(inputs):
    """Accurate model for the 12 known shapes, simple linear fallback for anything else."""
    return predict_loads(bundle, pd.DataFrame([inputs])[RAW], fallback=not is_known_shape(inputs)).iloc[0]


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Building Energy Estimator")
st.sidebar.write("Quick estimate of a building's **heating** and **cooling load** from its design, "
                 "for comparing design options early on.")
orient_opts = assets["orientations"]
glaze_opts = assets["glazing_areas"]
dist_opts = [d for d in assets["glazing_distributions"] if d != 0]
for key, default in {"shape": 5, "orientation": orient_opts[0], "glazing": 0.25, "distribution": 1}.items():
    st.session_state.setdefault(key, default)          # starting values of the design form


def apply_preset():
    """Copy the chosen preset into the form. (A widget with a fixed key keeps its old value unless we set it here.)"""
    pr = assets["presets"].get(st.session_state["preset"])
    if pr:
        st.session_state["shape"] = pr["shape_id"]
        st.session_state["orientation"] = pr["orientation"]
        st.session_state["glazing"] = pr["glazing_area"]
        if pr["glazing_area_distribution"] in dist_opts:
            st.session_state["distribution"] = pr["glazing_area_distribution"]


preset = st.sidebar.selectbox("Start from a preset design", ["(none)"] + list(assets["presets"]), key="preset", on_change=apply_preset)
for t in TARGETS:
    st.sidebar.caption(f"{NAMES[t]}: **{bundle[t]['model_name'].split(' (')[0]}**, trained on {bundle[t]['trained_on_rows']} simulated buildings")
st.sidebar.warning("Trained on **simulated** buildings with only 12 different shapes. Use it to compare options, "
                   "not as a replacement for a full energy simulation.")
tab_predict, tab_compare, tab_info, tab_monitor = st.tabs(["Predict", "Compare designs", "Model info", "Monitoring"])

# ----------------------------------------------------------------------------- predict tab
with tab_predict:
    st.header("Estimate heating and cooling load")
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("1. Describe the design")
        ids = [int(i) for i in shapes.shape_id]
        sid = st.selectbox("Building shape", ids, format_func=shape_label, key="shape",
                           help="The dataset contains 12 building shapes. Each fixes the compactness, surface, wall and roof areas and the height.")
        orientation = st.selectbox("Orientation", orient_opts, format_func=lambda v: f"Orientation {v}", key="orientation")
        glazing = st.selectbox("Glazing (window) area", glaze_opts, format_func=lambda v: f"{v:.0%} of floor area", key="glazing")
        distribution = st.selectbox("Window distribution", dist_opts, format_func=dist_label, key="distribution",
                                    disabled=glazing == 0, help="Where the windows are. Ignored when there are no windows.")
        base = shapes[shapes.shape_id == sid].iloc[0]
        geometry = {c: float(base[c]) for c in GEOMETRY}
        with st.expander("Custom geometry (advanced)"):
            st.caption("The models only know 12 shapes. Typing a different geometry is allowed, but the estimate becomes less reliable.")
            custom = st.checkbox("Use custom geometry", key="custom")
            if custom:
                for c in GEOMETRY:
                    lo, hi = assets["ranges"][c]
                    geometry[c] = st.number_input(c.replace("_", " ").capitalize(), value=float(base[c]), min_value=float(lo) * 0.5,
                                                  max_value=float(hi) * 1.5, step=0.01 if c == "relative_compactness" else 0.5,
                                                  key=f"custom_{sid}_{c}")
        go = st.button("Predict", type="primary", key="predict")

    inputs = {**geometry, "orientation": int(orientation), "glazing_area": float(glazing),
              "glazing_area_distribution": int(distribution) if glazing > 0 else 0}

    if go:
        t0 = time.perf_counter()
        pred = predict_one(inputs)
        latency = (time.perf_counter() - t0) * 1000
        near_id, near_d = nearest_shape(shapes, geometry)
        rec = {"request_id": uuid.uuid4().hex[:12], "inputs": inputs, "latency": latency, "near_id": near_id, "near_d": near_d,
               "in_domain": near_d < 1e-6, "model_used": "main" if near_d < 1e-6 else "fallback",
               "pred": {t: float(pred[t]) for t in TARGETS}, "band": {t: float(pred[f"{t}_band"]) for t in TARGETS}}
        log_prediction(rec)
        st.session_state["result"] = rec

    rec = st.session_state.get("result")
    with right:
        st.subheader("2. Estimated loads")
        if not rec:
            st.info("Choose a design on the left and press **Predict**.")
        else:
            c1, c2 = st.columns(2)
            for col, t, color in [(c1, "heating_load", HEAT), (c2, "cooling_load", COOL)]:
                with col:
                    st.markdown(f"<div style='padding:12px;border-radius:8px;background:{color}18;border-left:6px solid {color}'>"
                                f"<div style='font-size:0.9rem;color:#555'>{NAMES[t]}</div>"
                                f"<div style='font-size:2rem;font-weight:700'>{rec['pred'][t]:.1f}</div>"
                                f"<div style='font-size:0.85rem;color:#555'>{UNIT}<br>likely within ± {rec['band'][t]:.1f} (90% of cases)</div></div>",
                                unsafe_allow_html=True)
            st.write("")
            for t in TARGETS:
                share = float((reference[t] < rec["pred"][t]).mean())
                st.caption(f"{NAMES[t]} is higher than {share:.0%} of the buildings in the data.")
            if not rec["in_domain"]:
                if rec["near_d"] > 0.5:
                    st.error(f"**Far from the known shapes.** This geometry is not close to any of the 12 shapes the models were trained on (closest: shape {rec['near_id']}). "
                             "The simple fallback model was used and its error band is wide. Treat the estimate as a rough guess.")
                else:
                    st.warning(f"**Custom geometry.** Not one of the 12 known shapes (closest: shape {rec['near_id']}). "
                               "The simple fallback model was used, so the error band is much wider.")
            g = rec["inputs"]
            if abs(g["surface_area"] - (g["wall_area"] + 2 * g["roof_area"])) > 1.0:
                st.warning("The areas are not consistent: surface area should equal wall area + 2 x roof area.")
            used = "accurate main model (known shape)" if rec["model_used"] == "main" else "simple linear fallback (custom geometry)"
            st.caption(f"Model used: {used}. Computed in {rec['latency']:.0f} ms. Request {rec['request_id']}.")

            if st.button("Add this design to the comparison", key="add"):
                st.session_state.setdefault("designs", []).append(
                    {"Design": f"Design {len(st.session_state.get('designs', [])) + 1}",
                     "Shape": rec["near_id"] if rec["in_domain"] else f"custom (~{rec['near_id']})",
                     "Orientation": g["orientation"], "Glazing": f"{g['glazing_area']:.0%}",
                     "Heating": round(rec["pred"]["heating_load"], 1), "Cooling": round(rec["pred"]["cooling_load"], 1),
                     "Total": round(rec["pred"]["heating_load"] + rec["pred"]["cooling_load"], 1)})
                st.success("Added. See the *Compare designs* tab.")

    if rec:
        st.subheader("3. What if I change something?")
        opt = st.selectbox("Vary one input and keep the rest", ["Glazing area", "Orientation", "Window distribution", "Building shape"], key="whatif")
        base_in = dict(rec["inputs"]); variants = []
        if opt == "Glazing area":
            for v in assets["glazing_areas"]:
                d = dict(base_in, glazing_area=v, glazing_area_distribution=0 if v == 0 else (base_in["glazing_area_distribution"] or 1))
                variants.append((f"{v:.0%}", d))
        elif opt == "Orientation":
            variants = [(f"Orientation {v}", dict(base_in, orientation=v)) for v in assets["orientations"]]
        elif opt == "Window distribution":
            if base_in["glazing_area"] == 0:
                variants = []
                st.info("There are no windows in this design, so the distribution does not matter.")
            else:
                variants = [(dist_label(v), dict(base_in, glazing_area_distribution=v)) for v in assets["glazing_distributions"] if v != 0]
        else:
            for sid_ in ids:
                r = shapes[shapes.shape_id == sid_].iloc[0]
                variants.append((f"Shape {sid_}", dict(base_in, **{c: float(r[c]) for c in GEOMETRY})))
        if variants:
            frame = pd.DataFrame([{"Heating load": float(predict_one(d)["heating_load"]), "Cooling load": float(predict_one(d)["cooling_load"])} for _, d in variants],
                                 index=[n for n, _ in variants])
            st.bar_chart(frame, color=[HEAT, COOL], x_label=opt, y_label=UNIT, stack=False)

        with st.expander("Report the real loads (helps us monitor the model)"):
            with st.form("feedback_form", clear_on_submit=True):
                ah = st.number_input("Actual heating load (leave 0 if unknown)", min_value=0.0, value=0.0, step=0.1)
                ac = st.number_input("Actual cooling load (leave 0 if unknown)", min_value=0.0, value=0.0, step=0.1)
                note = st.text_input("Comment")
                if st.form_submit_button("Send"):
                    log_feedback({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "request_id": rec["request_id"],
                                  "actual_heating": ah if ah > 0 else None, "actual_cooling": ac if ac > 0 else None, "comment": note})
                    st.success("Thank you, saved.")

# ----------------------------------------------------------------------------- compare tab
with tab_compare:
    st.header("Compare design options")
    designs = st.session_state.get("designs", [])
    if not designs:
        st.info("Predict a design and press **Add this design to the comparison**. Repeat for other options.")
    else:
        df = pd.DataFrame(designs)
        st.dataframe(df, hide_index=True, width="stretch")
        st.bar_chart(df.set_index("Design")[["Heating", "Cooling"]], color=[HEAT, COOL], y_label=UNIT, stack=False)
        best = df.loc[df["Total"].idxmin()]
        st.success(f"Lowest combined load: **{best['Design']}** ({best['Total']} {UNIT}).")
        if st.button("Clear the comparison", key="clear"):
            st.session_state["designs"] = []
            st.rerun()

# ----------------------------------------------------------------------------- info tab
with tab_info:
    st.header("About the models")
    st.write("Two models estimate the **heating** and **cooling load** of a residential building from 8 design inputs. They were trained on the public "
             "*Energy Efficiency* dataset (UCI): **768 simulated buildings** made of 12 shapes x 4 orientations x 16 window options.")
    rows = []
    for t in TARGETS:
        b = bundle[t]; m = b["test_metrics"]; u = b["unseen_shape"]
        rows.append({"Target": NAMES[t], "Algorithm": b["model_name"].split(" (")[0],
                     "Test RMSE": f"{m['RMSE']:.2f} (95% CI {m['RMSE_ci']})", "Test R²": f"{m['R2']:.3f}",
                     "Guessing the average": f"RMSE {m['baseline_RMSE']:.1f}", "90% error band": f"± {b['error_band_90']:.1f}",
                     "Unseen shape RMSE": f"{u['RMSE']:.1f}"})
    st.subheader("Accuracy on buildings the model never saw (test set)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("RMSE = typical size of the error, in kWh/m² per year (lower is better). R² = share of the variation in the loads that the model explains (1 is perfect).")
    st.subheader("How reliable is it for a brand-new shape?")
    st.write("The test buildings share their 12 shapes with the training data, so the test score describes **known shapes with a new orientation or windows**. "
             "When a **whole shape is hidden** from the model, the error is much larger (last column above). "
             "For a custom geometry the app therefore switches to a simpler, more robust linear model and shows a much wider error band:")
    st.dataframe(pd.DataFrame([{"Target": NAMES[t], "Fallback model": bundle[t]["fallback"]["model_name"],
                                "Unseen-shape RMSE": f"{bundle[t]['fallback']['unseen_shape']['RMSE']:.1f}",
                                "90% error band": f"± {bundle[t]['fallback']['error_band_90']:.1f}"} for t in TARGETS]),
                 hide_index=True, width="stretch")
    st.subheader("Limitations")
    st.markdown(
        "- The data is **simulated**, not measured, and covers only **12 shapes**.\n"
        "- In the data, **compactness and height are tangled** (all compact shapes are tall), so the effect of compactness on its own cannot be separated from height.\n"
        "- Orientation has almost no effect in this dataset, and windows only make small adjustments on top of the shape.\n"
        "- Estimates are for comparing options. Use a full energy simulation before making real decisions.")

# ----------------------------------------------------------------------------- monitoring tab
with tab_monitor:
    st.header("Monitoring")
    logs = monitor.load_logs(LOG_DIR)
    if logs.empty:
        st.info("No predictions logged yet. Make a prediction in the **Predict** tab and come back.")
    else:
        lat = monitor.latency_summary(logs)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Predictions logged", lat["requests"]); c2.metric("Mean latency", f"{lat['mean_ms']:.0f} ms")
        c3.metric("95th percentile latency", f"{lat['p95_ms']:.0f} ms")
        c4.metric("Mean predicted heating", f"{logs.heating_pred.mean():.1f}"); c5.metric("Outside the 12 shapes", f"{(~logs.in_domain.astype(bool)).mean():.0%}")
        left, right = st.columns(2)
        with left:
            st.subheader("Predicted loads per request")
            st.line_chart(logs[["heating_pred", "cooling_pred"]].reset_index(drop=True), color=[HEAT, COOL])
        with right:
            st.subheader("Latency per request (ms)")
            st.line_chart(logs["latency_ms"].astype(float).reset_index(drop=True))
        st.subheader("Data drift (requests vs. training data)")
        live = monitor.logged_inputs(logs)
        if len(live) < monitor.MIN_ROWS_FOR_DRIFT:
            st.info(f"Drift checks need at least {monitor.MIN_ROWS_FOR_DRIFT} logged predictions (now {len(live)}); with fewer rows the drift score is mostly random noise. "
                    "Predictions made while testing the app are not real usage.")
        else:
            rep = monitor.drift_report(reference, live, RAW)
            st.dataframe(rep, hide_index=True, width="stretch")
            if (rep.status == "LARGE shift").any():
                st.error("Large drift detected in some inputs. Check what changed and consider adding matching simulations to the training data.")
            else:
                st.success("No large drift detected.")
    st.subheader("Live error from user feedback")
    fb_file = LOG_DIR / "feedback.csv"
    fb = pd.read_csv(fb_file) if fb_file.exists() else None
    err = monitor.live_error(logs, fb)
    if err is None:
        st.info("No real loads have been reported yet.")
    else:
        for k, v in err.items():
            st.write(f"**{k.capitalize()}**: {v['n']} reported, MAE {v['MAE']:.2f}, RMSE {v['RMSE']:.2f} ({UNIT}). Compare with the test RMSE on the *Model info* tab.")

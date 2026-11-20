"""Student Success Predictor - Streamlit app (ML process step 8: deployment, step 10: monitoring).

Run:  streamlit run app.py
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import monitor
from inference import CLASSES, predict_students

ROOT = Path(__file__).parent
LOG_DIR = Path(os.environ.get("LOG_DIR", ROOT / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Student Success Predictor", page_icon=":mortar_board:", layout="wide")


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
types, ranges, defaults = assets["types"], assets["ranges"], assets["defaults"]

SCENARIOS = {"early": "Early warning (after 1st semester)", "full": "Full (after 2nd semester)"}
COLORS = {"Dropout": "#d1495b", "Enrolled": "#edae49", "Graduate": "#00798c"}

BINARY_LABELS = {"gender": ("Female", "Male"), "daytime_evening_attendance": ("Evening", "Daytime")}
LIMITS = {"admission_grade": (0.0, 200.0), "previous_qualification_grade": (0.0, 200.0),
          "curricular_units_1st_sem_grade": (0.0, 20.0), "curricular_units_2nd_sem_grade": (0.0, 20.0),
          "age_at_enrollment": (15, 80), "unemployment_rate": (0.0, 30.0),
          "inflation_rate": (-5.0, 15.0), "gdp": (-10.0, 10.0)}
HELP = {
    "tuition_fees_up_to_date": "Are the student's tuition fees paid up to date?",
    "debtor": "Does the student owe money to the institution?",
    "scholarship_holder": "Does the student hold a scholarship?",
    "admission_grade": "Admission grade, scale 0-200.",
    "previous_qualification_grade": "Grade of the previous qualification, scale 0-200.",
    "curricular_units_1st_sem_credited": "Course units recognised from earlier studies.",
    "curricular_units_1st_sem_enrolled": "Course units the student registered for.",
    "curricular_units_1st_sem_evaluations": "Number of exams/evaluations taken.",
    "curricular_units_1st_sem_approved": "Course units passed.",
    "curricular_units_1st_sem_grade": "Average grade of the semester, scale 0-20 (0 = no graded passes).",
    "curricular_units_1st_sem_without_evaluations": "Units with no evaluation taken.",
    "application_order": "0 = first choice ... 9 = last choice.",
}

SECTIONS = [
    ("Student background", ["age_at_enrollment", "gender", "marital_status", "nationality", "international",
                            "displaced", "educational_special_needs"]),
    ("Application and course", ["course", "daytime_evening_attendance", "application_mode", "application_order",
                                "previous_qualification", "previous_qualification_grade", "admission_grade"]),
    ("Finances", ["tuition_fees_up_to_date", "debtor", "scholarship_holder"]),
    ("1st semester results", [f"curricular_units_1st_sem_{k}" for k in
                              ["enrolled", "evaluations", "approved", "grade", "credited", "without_evaluations"]]),
    ("2nd semester results", [f"curricular_units_2nd_sem_{k}" for k in
                              ["enrolled", "evaluations", "approved", "grade", "credited", "without_evaluations"]]),
    ("Economic context at enrolment", ["unemployment_rate", "inflation_rate", "gdp"]),
    ("Family (optional details)", ["mothers_qualification", "fathers_qualification", "mothers_occupation",
                                   "fathers_occupation"]),
]


def nice(col):
    return col.replace("curricular_units_", "").replace("_", " ").replace("1st sem", "1st sem.").replace(
        "2nd sem", "2nd sem.").replace("mothers", "mother's").replace("fathers", "father's").capitalize()


def code_label(col, code):
    return f"{assets['code_tables'][col].get(str(code), 'code')} ({code})"


def widget(col, value, key):
    """Create the right input widget for one raw feature and return its value."""
    kind, label, hlp = types[col], nice(col), HELP.get(col)
    if kind == "binary":
        no, yes = BINARY_LABELS.get(col, ("No", "Yes"))
        return st.selectbox(label, [0, 1], index=int(value), format_func=lambda v: yes if v else no, key=key, help=hlp)
    if kind == "nominal":
        options = assets["observed_codes"][col]
        idx = options.index(int(value)) if int(value) in options else 0
        return st.selectbox(label, options, index=idx, format_func=lambda c: code_label(col, c), key=key, help=hlp)
    lo, hi = ranges[col]
    if col in LIMITS:
        lo_l, hi_l = LIMITS[col]
    else:
        lo_l, hi_l = 0, max(40, int(hi * 2))
    if kind == "int":
        return st.number_input(label, int(lo_l), int(hi_l), int(round(value)), 1, key=key, help=hlp)
    return st.number_input(label, float(lo_l), float(hi_l), float(value), 0.1, format="%.1f", key=key, help=hlp)


def log_prediction(scenario, latency_ms, label, proba, raw):
    f = LOG_DIR / "predictions.csv"
    row = pd.DataFrame([{"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "scenario": scenario, "latency_ms": round(latency_ms, 2), "predicted": label,
                         "p_dropout": round(float(proba[0]), 4), "p_enrolled": round(float(proba[1]), 4),
                         "p_graduate": round(float(proba[2]), 4), "inputs_json": json.dumps(raw)}])
    row.to_csv(f, mode="a", header=not f.exists(), index=False)


def log_feedback(record):
    f = LOG_DIR / "feedback.csv"
    pd.DataFrame([record]).to_csv(f, mode="a", header=not f.exists(), index=False)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Student Success Predictor")
scenario = st.sidebar.radio("Which model?", list(SCENARIOS), format_func=SCENARIOS.get,
                            help="The early-warning model only needs background data and 1st-semester results.")
example = st.sidebar.selectbox("Start from an example profile", ["Blank (typical values)"] + list(assets["examples"]))
entry = bundle[scenario]
st.sidebar.caption(
    f"Model: **{entry['model_name'].split(' (')[0]}**  \n"
    f"Dropout alarm when P(Dropout) >= **{entry['threshold']}**  \n"
    f"Trained on {entry['trained_on_rows']:,} students.")
st.sidebar.warning("Decision support only. A prediction should start a supportive conversation, "
                   "never an automatic decision about a student.")

start = defaults if example.startswith("Blank") else assets["examples"][example]

tab_predict, tab_info, tab_monitor = st.tabs(["Predict", "Model info", "Monitoring"])

# ----------------------------------------------------------------------------- predict tab
with tab_predict:
    st.header("Predict a student's outcome")
    st.write("Fill in what you know, then press **Predict**. Example profiles are *typical* students of a group, "
             "not real individuals.")

    with st.form("student_form"):
        raw = {}
        for title, cols in SECTIONS:
            if "2nd semester" in title and scenario == "early":
                for c in cols:                                   # not used by the early model
                    raw[c] = 0
                continue
            with st.expander(title, expanded=title != "Family (optional details)"):
                grid = st.columns(3)
                for i, c in enumerate(cols):
                    with grid[i % 3]:
                        raw[c] = widget(c, start[c], key=f"{scenario}|{example}|{c}")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        t0 = time.perf_counter()
        names, proba = predict_students(entry, pd.DataFrame([raw])[assets["raw_features"]])
        latency = (time.perf_counter() - t0) * 1000
        log_prediction(scenario, latency, names[0], proba[0], {k: (v.item() if hasattr(v, "item") else v)
                                                              for k, v in raw.items()})
        st.session_state["result"] = {"scenario": scenario, "label": names[0], "proba": proba[0], "latency": latency}

    res = st.session_state.get("result")
    if res and res["scenario"] == scenario:
        label, proba = res["label"], res["proba"]
        st.markdown(f"<div style='padding:14px;border-radius:8px;background:{COLORS[label]}22;"
                    f"border-left:8px solid {COLORS[label]}'><span style='font-size:1.6rem;font-weight:700'>"
                    f"Predicted outcome: {label}</span></div>", unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            st.bar_chart(pd.Series(proba, index=CLASSES, name="probability"), horizontal=True)
        with right:
            if label == "Dropout":
                st.error("**At risk.** Consider early outreach: academic support, financial-aid advice or a "
                         "conversation with the student.")
            elif label == "Enrolled":
                st.warning("**Uncertain / still in progress.** This group is the hardest to predict. Keep monitoring.")
            else:
                st.success("**On track to graduate.** No special action suggested.")
            st.caption(f"P(Dropout) = {proba[0]:.0%}. The alarm threshold is {entry['threshold']:.0%} (lower than "
                       f"50% on purpose, so fewer at-risk students are missed). Computed in {res['latency']:.0f} ms.")

        with st.expander("Give feedback on this prediction"):
            with st.form("feedback_form", clear_on_submit=True):
                rating = st.radio("Was this prediction helpful?", ["Helpful", "Not helpful"], horizontal=True)
                actual = st.selectbox("Actual outcome (if already known)", ["Not known yet"] + CLASSES)
                comment = st.text_input("Comment (do not enter personal data)")
                if st.form_submit_button("Send feedback"):
                    log_feedback({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "scenario": scenario, "predicted": label, "rating": rating,
                                  "actual_outcome": actual, "comment": comment})
                    st.success("Thank you - feedback saved.")

# ----------------------------------------------------------------------------- info tab
with tab_info:
    st.header("About the model")
    st.write("This tool estimates whether a higher-education student will **drop out, stay enrolled or graduate**, "
             "using the public *Predict Students' Dropout and Academic Success* dataset (UCI, 4,424 students, "
             "one Portuguese institution).")
    rows = []
    for key, name in SCENARIOS.items():
        m = bundle[key]["test_metrics"]
        rows.append({"Model": name, "Algorithm": bundle[key]["model_name"].split(" (")[0],
                     "Test macro-F1": f"{m['macro_f1']:.2f} (95% CI {m['f1_ci']})",
                     "Test Dropout recall": f"{m['dropout_recall']:.2f} (95% CI {m['recall_ci']})",
                     "Test Dropout precision": f"{m['dropout_precision']:.2f}",
                     "Test accuracy": f"{m['accuracy']:.2f}"})
    st.subheader("Performance on students the model never saw (test set)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Scores use the alarm threshold of this app (Dropout flagged when its probability is at least "
               f"{bundle['early']['threshold']}). With the plain 'most likely class' rule, macro-F1 is higher "
               "(0.76 full, 0.70 early) and Dropout recall lower (0.76 and 0.77).")
    st.subheader("How to read this")
    st.markdown(
        "- **Recall for Dropout:** of the students who really drop out, the share the tool flags.\n"
        "- **Precision for Dropout:** of the students flagged, the share who really drop out.\n"
        "- The *Enrolled* group is the hardest to predict because those students are in between the two outcomes.")
    st.subheader("Limitations")
    st.markdown(
        "- Trained on **one institution**; it may not transfer to other universities or countries.\n"
        "- It shows **association, not causation**: paying tuition on time does not by itself make a student stay.\n"
        "- Predictions are **probabilities**, and between one in five and one in four students is still classified wrongly.\n"
        "- Gender and nationality are inputs because they are in the dataset; use the tool with care "
        "and check for unfair outcomes between groups before real use.")

# ----------------------------------------------------------------------------- monitoring tab
with tab_monitor:
    st.header("Monitoring")
    logs = monitor.load_logs(LOG_DIR)
    if logs.empty:
        st.info("No predictions logged yet. Make a prediction in the **Predict** tab and come back.")
    else:
        lat = monitor.latency_summary(logs)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictions logged", lat["requests"])
        c2.metric("Mean latency", f"{lat['mean_ms']:.0f} ms")
        c3.metric("95th percentile latency", f"{lat['p95_ms']:.0f} ms")
        c4.metric("Dropout alarms", f"{(logs.predicted == 'Dropout').mean():.0%}")

        left, right = st.columns(2)
        with left:
            st.subheader("Prediction mix")
            st.bar_chart(logs["predicted"].value_counts().reindex(CLASSES).fillna(0))
        with right:
            st.subheader("Latency per request (ms)")
            st.line_chart(logs["latency_ms"].astype(float).reset_index(drop=True))

        st.subheader("Data drift (live inputs vs. training data)")
        live = monitor.logged_inputs(logs)
        numeric = [c for c in assets["raw_features"] if types[c] in ("int", "float") and c in live]
        if len(live) < monitor.MIN_ROWS_FOR_DRIFT:
            st.info(f"Drift checks need at least {monitor.MIN_ROWS_FOR_DRIFT} logged predictions (now {len(live)}); "
                    "with fewer rows the drift score is mostly random noise. Predictions made while testing the "
                    "app are not real student data.")
        else:
            rep = monitor.drift_report(reference, live, numeric)
            st.dataframe(rep, hide_index=True, width="stretch")
            if (rep.status == "LARGE shift").any():
                st.error("Large drift detected in some features - consider reviewing and retraining the model.")
            else:
                st.success("No large drift detected.")

    fb_file = LOG_DIR / "feedback.csv"
    st.subheader("User feedback")
    if fb_file.exists():
        fb = pd.read_csv(fb_file)
        st.write(f"{len(fb)} feedback entries - {(fb.rating == 'Helpful').mean():.0%} rated helpful.")
        known = fb[fb.actual_outcome != "Not known yet"]
        if len(known):
            st.write(f"Where the real outcome is known ({len(known)} cases), the prediction was correct "
                     f"{(known.predicted == known.actual_outcome).mean():.0%} of the time.")
        st.dataframe(fb.tail(10), hide_index=True, width="stretch")
    else:
        st.info("No feedback yet.")

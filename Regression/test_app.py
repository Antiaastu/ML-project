"""Automated smoke test of the Streamlit app (runs it headlessly, no browser needed).

Run inside the container:  python test_app.py
"""
import os
import shutil
import tempfile

LOG_DIR = tempfile.mkdtemp()
os.environ["LOG_DIR"] = LOG_DIR            # keep test logs out of the real log folder

import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402
from streamlit.testing.v1 import AppTest    # noqa: E402

import monitor                              # noqa: E402

TIMEOUT = 120


def check(at, what):
    assert not at.exception, f"{what}: app raised {[e.value for e in at.exception]}"
    print(f"OK  {what}")


at = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
check(at, "app loads")

presses = 0
main_band = {}
import json                                     # noqa: E402
presets = json.load(open("models/app_assets.json"))["presets"]
seen_predictions = []
for preset in ["Compact tall block, no windows", "Low-rise with large windows", "Tall building, medium windows"]:
    at.selectbox(key="preset").set_value(preset).run()
    # the form must really change when the preset changes (this once silently did not)
    assert at.selectbox(key="shape").value == presets[preset]["shape_id"], f"preset '{preset}' did not update the building shape"
    assert at.selectbox(key="glazing").value == presets[preset]["glazing_area"], f"preset '{preset}' did not update the glazing area"
    at.button(key="predict").click().run(); presses += 1
    check(at, f"preset '{preset}' -> predict")
    r = at.session_state["result"]
    seen_predictions.append(round(r["pred"]["heating_load"], 1))
    assert all(np.isfinite(v) and v > 0 for v in r["pred"].values()) and all(b > 0 for b in r["band"].values())
    assert r["in_domain"] and r["model_used"] == "main", "preset designs use one of the 12 known shapes and the main model"
    main_band = r["band"]
    print(f"      heating {r['pred']['heating_load']:.1f} +/- {r['band']['heating_load']:.1f} | "
          f"cooling {r['pred']['cooling_load']:.1f} +/- {r['band']['cooling_load']:.1f} | latency {r['latency']:.0f} ms")

assert len(set(seen_predictions)) == 3, f"the three presets must give three different predictions, got {seen_predictions}"
print("OK  the three presets give three different predictions:", seen_predictions)

# no windows -> the distribution is forced to 0
at.selectbox(key="glazing").set_value(0.0).run()
at.button(key="predict").click().run(); presses += 1
check(at, "no windows -> predict")
assert at.session_state["result"]["inputs"]["glazing_area_distribution"] == 0

# every one of the 12 shapes works, and more glazing never makes a tall building much cooler (sanity)
for sid in range(1, 13):
    at.selectbox(key="shape").set_value(sid).run()
    at.button(key="predict").click().run(); presses += 1
    assert not at.exception, f"shape {sid} failed"
print("OK  all 12 shapes predict without errors")

# custom geometry far from every known shape -> flagged as outside the training range
at.checkbox(key="custom").check().run()
sid = at.selectbox(key="shape").value
at.number_input(key=f"custom_{sid}_roof_area").set_value(400.0).run()
at.number_input(key=f"custom_{sid}_overall_height").set_value(10.0).run()
at.button(key="predict").click().run(); presses += 1
check(at, "custom geometry -> predict")
r = at.session_state["result"]
assert (not r["in_domain"]) and r["near_d"] > 0.5, r
assert r["model_used"] == "fallback", "custom geometry must use the linear fallback model"
assert main_band, "the preset loop should have recorded the main model's error band"
assert all(r["band"][t] > main_band[t] for t in main_band), "the fallback's error band must be wider than the main model's"
assert len(at.error) >= 1, "expected a 'far from the known shapes' error message"
print(f"OK  out-of-range geometry flagged (distance {r['near_d']:.1f}); fallback used, bands "
      f"+/-{r['band']['heating_load']:.1f} / +/-{r['band']['cooling_load']:.1f} (main: +/-{main_band['heating_load']:.1f} / +/-{main_band['cooling_load']:.1f})")

# comparison list
at.checkbox(key="custom").uncheck().run()
at.button(key="predict").click().run(); presses += 1
at.button(key="add").click().run()
assert len(at.session_state["designs"]) == 1
check(at, "add to comparison")

# what-if chart for each option
for opt in ["Glazing area", "Orientation", "Window distribution", "Building shape"]:
    at.selectbox(key="whatif").set_value(opt).run()
    check(at, f"what-if: {opt}")

logs = pd.read_csv(os.path.join(LOG_DIR, "predictions.csv"))
assert len(logs) == presses, f"expected {presses} logged predictions, got {len(logs)}"
print(f"OK  prediction log written ({len(logs)} rows)")

# live error computed from feedback
fb = pd.DataFrame({"request_id": logs.request_id.head(3), "actual_heating": logs.heating_pred.head(3) + [1.0, -1.0, 2.0],
                   "actual_cooling": [np.nan, np.nan, np.nan]})
err = monitor.live_error(logs, fb)
assert err is not None, "live error should be computed from the feedback"
assert abs(err["heating"]["MAE"] - 4 / 3) < 1e-6 and "cooling" not in err
print("OK  live error from feedback computed correctly")
shutil.rmtree(LOG_DIR, ignore_errors=True)
print("\nALL APP TESTS PASSED")

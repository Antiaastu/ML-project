"""Automated smoke test of the Streamlit app (runs it headlessly, no browser needed).

Run inside the container:  python test_app.py
"""
import os
import shutil
import tempfile

LOG_DIR = tempfile.mkdtemp()
os.environ["LOG_DIR"] = LOG_DIR            # keep test logs out of the real log folder

from streamlit.testing.v1 import AppTest   # noqa: E402  (must come after LOG_DIR is set)

TIMEOUT = 120


def check(at, what):
    assert not at.exception, f"{what}: app raised {[e.value for e in at.exception]}"
    print(f"OK  {what}")


at = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
check(at, "app loads (early model, blank profile)")

for scenario in ["early", "full"]:
    at.sidebar.radio[0].set_value(scenario).run()
    check(at, f"switch to {scenario} model")
    for example in ["Typical graduate", "Typical dropout", "Typical still-enrolled"]:
        at.sidebar.selectbox[0].set_value(example).run()
        at.button[0].click().run()                       # form submit button "Predict"
        check(at, f"{scenario} / {example} -> predict")
        res = at.session_state["result"]
        assert abs(sum(res["proba"]) - 1) < 1e-6
        print(f"      predicted {res['label']:<9} P(dropout)={res['proba'][0]:.2f}  latency={res['latency']:.0f} ms")

import pandas as pd                                      # noqa: E402
logs = pd.read_csv(os.path.join(LOG_DIR, "predictions.csv"))
assert len(logs) == 6, f"expected 6 logged predictions, got {len(logs)}"
print(f"OK  prediction log written ({len(logs)} rows)")
shutil.rmtree(LOG_DIR, ignore_errors=True)
print("\nALL APP TESTS PASSED")

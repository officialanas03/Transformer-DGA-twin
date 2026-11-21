"""Regression test for the Live rig tab of app.py WITHOUT Streamlit: the tab's code is executed against a minimal fake
`streamlit` object, so errors in the panel (missing keys, bad formatting, chart columns) show up in CI even where the
real app cannot be started. This does not test Streamlit itself or the auto-refresh timer; those need a browser."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"
st = None

class Box:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def metric(self, label, value, delta=None, **k): self.log.append(("metric", label, value, delta))
    def caption(self, t): self.log.append(("caption", t))
    def line_chart(self, df, **k): self.log.append(("chart", list(df.columns), len(df)))
    def button(self,l,key=None,**k): return key in st.press
    def selectbox(self,*a,**k): return st.selectbox(*a,**k)
    def select_slider(self,*a,**k): return st.select_slider(*a,**k)
    def checkbox(self,*a,**k): return st.checkbox(*a,**k)
    def slider(self,*a,**k): return st.slider(*a,**k)
    def text_input(self,*a,**k): return st.text_input(*a,**k)
    def __getattr__(self, n): return lambda *a, **k: self.log.append((n,))
class ST:
    def __init__(self):
        self.log=[]; self.session_state={}; self.press=set(); self.fragment=self._fragment
        globals()["st"] = self          # the column stubs delegate widget calls to the active fake
    def _fragment(self, func=None, run_every=None):
        def deco(f): return f
        return deco
    def subheader(self,*a,**k): self.log.append(("subheader",a))
    def caption(self,t,**k): self.log.append(("caption",t))
    def radio(self,l,opts,**k): return opts[0]
    def columns(self,n): return [Box(self.log) for _ in range(n if isinstance(n,int) else len(n))]
    def selectbox(self,l,opts,**k): return opts[0]
    def select_slider(self,l,options,value=None,**k): return value
    def checkbox(self,l,value=False,**k): return value
    def slider(self,l,a,b,v=None,**k): return v
    def text_input(self,l,value="",**k): return value
    def button(self,l,key=None,**k): return key in self.press
    def info(self,t): self.log.append(("info",t))
    def warning(self,t): self.log.append(("warning",t))
    def error(self,t): self.log.append(("error",t))
    def write(self,t): self.log.append(("write",t))
    def pyplot(self,f): self.log.append(("pyplot",))
    def metric(self,*a,**k): pass

@pytest.fixture(scope="module")
def tab_code():
    src = APP.read_text(encoding="utf-8")
    return src[src.index("# ---------------- Tab 5"):]

def _run(code, st, n, advance_s=2.0):
    ns = dict(st=st, np=np, plt=plt, tab_live=Box(st.log))
    st.press = {"live_start"}
    exec(code, ns)
    st.press = set()
    for _ in range(n):
        st.session_state["live_last"] -= advance_s            # pretend wall-clock time passed
        st.log.clear()
        exec(code, ns)
    return ns

def test_idle_panel_before_start(tab_code):
    st = ST(); ns = dict(st=st, np=np, plt=plt, tab_live=Box(st.log))
    exec(tab_code, ns)
    assert any(e[0] == "info" for e in st.log)
    assert st.session_state.get("live") is None

def test_panel_runs_and_shows_synthetic_banner_metrics_and_charts(tab_code):
    st = ST(); _run(tab_code, st, 12)
    ses = st.session_state["live"]
    assert ses.synthetic and len(ses.rows) > 1000 and ses.twin.n_updates > 0
    kinds = [e[0] for e in st.log]
    assert "warning" in kinds and "error" not in kinds
    labels = [e[1] for e in st.log if e[0] == "metric"]
    assert labels[0].startswith("Film coefficient") and len(labels) == 4
    charts = [e for e in st.log if e[0] == "chart"]
    assert charts[0][1] == ["meas_T_paper_top", "pred_T_paper_top"]
    assert charts[1][1] == ["estimate", "true (hidden, SYNTHETIC only)"]

def test_dragging_the_cooling_health_slider_moves_the_estimate(tab_code):
    st = ST(); _run(tab_code, st, 25)
    ses = st.session_state["live"]
    before = ses.twin.hf
    st.session_state["live_health"] = 60                      # what the slider does in the browser
    ns = dict(st=st, np=np, plt=plt, tab_live=Box(st.log))
    for _ in range(40):
        st.session_state["live_last"] -= 2.0
        exec(tab_code, ns)
    assert ses.twin.hf < 0.85 * before

def test_stop_freezes_the_run(tab_code):
    st = ST(); ns = _run(tab_code, st, 5)
    st.press = {"live_stop"}; exec(tab_code, ns); st.press = set()
    n = len(st.session_state["live"].rows)
    for _ in range(3):
        st.session_state["live_last"] -= 2.0
        exec(tab_code, ns)
    assert len(st.session_state["live"].rows) == n
    assert st.session_state["live_running"] is False

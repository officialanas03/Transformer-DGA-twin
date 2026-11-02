"""Streamlit app: DGA diagnosis and a simple transformer thermal/ageing twin.
Run with:  streamlit run app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dga.pipeline import load_diagnoser_or_none, diagnose_rules, diagnose_table, MAX_BATCH_ROWS
from dga.rules.duval import classify_pct, ZONE_ORDER
from dga.ageing.paper import dp_chendong
from twin.thermal import ThermalParams, simulate
from twin.ageing import (KineticAgeing, calibrate_A_from_life, arrhenius_factor,
                         relative_ageing_rate, remaining_life_hours, DP_END_OF_LIFE)

st.set_page_config(page_title="DGA + transformer twin", layout="wide")
st.title("DGA diagnosis and transformer twin")
st.caption("Research prototype. Not a substitute for an engineer's judgement or the relevant standards.")

@st.cache_resource
def get_diagnoser():
    """The random forest, or None when the training data is not part of this deployment."""
    return load_diagnoser_or_none()

EXAMPLES = {   # illustrative values, NOT real measurements
    "Custom (type your own)": None,
    "Illustrative: low-temperature overheating": dict(h2=60, ch4=120, c2h6=90, c2h4=30, c2h2=0),
    "Illustrative: high-temperature overheating": dict(h2=150, ch4=400, c2h6=120, c2h4=650, c2h2=8),
    "Illustrative: arcing": dict(h2=800, ch4=150, c2h6=30, c2h4=300, c2h2=900),
    "Illustrative: partial discharge": dict(h2=900, ch4=60, c2h6=10, c2h4=2, c2h2=0),
}
TEMPLATE_CSV = ("H2,CH4,C2H6,C2H4,C2H2,CO,CO2\n"
                "60,120,90,30,0,400,2500\n150,400,120,650,8,600,4000\n800,150,30,300,900,300,1500\n")

@st.cache_data
def triangle_grid(step=1.0):
    pts = []
    for c2h4 in np.arange(0, 100 + 1e-9, step):
        for c2h2 in np.arange(0, 100 - c2h4 + 1e-9, step):
            pts.append((c2h4, c2h2, classify_pct(100 - c2h4 - c2h2, c2h4, c2h2)))
    return pd.DataFrame(pts, columns=["c2h4", "c2h2", "zone"])

def to_xy(p_ch4, p_c2h4, p_c2h2):
    """Triangle corners: CH4 top, C2H4 bottom right, C2H2 bottom left."""
    return (p_c2h4 + 0.5 * p_ch4) / 100.0, (np.sqrt(3) / 2) * p_ch4 / 100.0

def draw_triangle(pct):
    g = triangle_grid()
    colors = dict(zip(ZONE_ORDER, ["#8ecae6", "#ffd166", "#f4a261", "#e76f51", "#b5e48c", "#52b788", "#adb5bd"]))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for z in ZONE_ORDER:
        s = g[g.zone == z]
        xs, ys = to_xy(100 - s.c2h4 - s.c2h2, s.c2h4, s.c2h2)
        ax.scatter(xs, ys, s=6, color=colors[z], label=z, marker="s")
    if pct is not None:
        x, y = to_xy(pct["CH4"], pct["C2H4"], pct["C2H2"])
        ax.scatter([x], [y], s=120, color="k", marker="x", zorder=5, label="sample")
    ax.text(0.5, 0.9, "CH4", ha="center"); ax.text(1.0, -0.04, "C2H4", ha="center"); ax.text(0.0, -0.04, "C2H2", ha="center")
    ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper right", fontsize=7, markerscale=1.5)
    return fig

tab1, tab_batch, tab2 = st.tabs(["DGA diagnosis", "Batch (CSV)", "Transformer twin"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader("Dissolved gas analysis")
    choice = st.selectbox("Start from an example", list(EXAMPLES))
    ex = EXAMPLES[choice] or dict(h2=100, ch4=50, c2h6=20, c2h4=80, c2h2=2)
    c = st.columns(5)
    h2 = c[0].number_input("H2 (ppm)", min_value=0.0, value=float(ex["h2"]))
    ch4 = c[1].number_input("CH4 (ppm)", min_value=0.0, value=float(ex["ch4"]))
    c2h6 = c[2].number_input("C2H6 (ppm)", min_value=0.0, value=float(ex["c2h6"]))
    c2h4 = c[3].number_input("C2H4 (ppm)", min_value=0.0, value=float(ex["c2h4"]))
    c2h2 = c[4].number_input("C2H2 (ppm)", min_value=0.0, value=float(ex["c2h2"]))
    has_co = st.checkbox("I have CO and CO2")
    co = co2 = None
    if has_co:
        cc = st.columns(2)
        co = cc[0].number_input("CO (ppm)", min_value=0.0, value=500.0)
        co2 = cc[1].number_input("CO2 (ppm)", min_value=0.0, value=3000.0)
    has_fal = st.checkbox("I have a 2-FAL (furfural) result")
    fal = None
    if has_fal:
        fal = st.number_input("2-FAL (ppm = mg/kg)", min_value=0.0, value=0.5)

    with st.spinner("Preparing the model..."):
        diag = get_diagnoser()
    if diag is not None:
        res = diag.diagnose(h2, ch4, c2h6, c2h4, c2h2, co=co, co2=co2, fal_ppm=fal)
    else:
        res = diagnose_rules(h2, ch4, c2h6, c2h4, c2h2, co=co, co2=co2, fal_ppm=fal)
        res.update(rf_proba={}, rf_set=[], rf_top=None)

    left, right = st.columns([1, 1])
    with left:
        st.metric("Duval triangle 1", res["duval"] or "n/a")
        st.metric("IEC ratio method", res["iec"])
        if diag is not None:
            label = ", ".join(res["rf_set"]) if res["rf_set"] else "no class"
            st.metric(f"Random forest set ({int((1 - diag.alpha) * 100)}% target coverage)", label)
            if res["rf_proba"]:
                st.table(pd.DataFrame({"probability": res["rf_proba"]}))
        else:
            st.info("The random forest is not part of this deployment (its training data is not redistributed). "
                    "Duval, IEC and the ageing indicators below do not need it.")
    with right:
        st.pyplot(draw_triangle(res["duval_pct"]))
    for w in res["warnings"]:
        st.warning(w)
    if "co2_co" in res:
        r = res["co2_co"]
        st.write(f"CO2/CO = {r['ratio']:.2f} ({r['band']})")
    if "furan" in res:
        f = res["furan"]
        st.write(f"Estimated DP from 2-FAL: Chendong {f['dp_chendong']:.0f}, De Pablo {f['dp_depablo']:.0f}. "
                 f"Condition band: {f['condition']}.")
        st.caption("Chendong is quoted for DP 150-1000 and De Pablo for 150-600; outside that range treat the number as indicative only.")

# ---------------- Batch tab ----------------
with tab_batch:
    st.subheader("Batch diagnosis from a CSV file")
    st.write("Upload a CSV with columns H2, CH4, C2H6, C2H4, C2H2 (ppm). Optional columns: CO, CO2, FAL (2-FAL in ppm). "
             f"At most {MAX_BATCH_ROWS} rows. This is a public demo: do not upload confidential utility data.")
    st.download_button("Download a template CSV", TEMPLATE_CSV, file_name="dga_template.csv", mime="text/csv")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            table = pd.read_csv(up)
            out = diagnose_table(table, get_diagnoser())
            st.dataframe(out)
            st.write("Duval zones in this file:", out["duval"].value_counts(dropna=False).to_dict())
            st.download_button("Download results as CSV", out.to_csv(index=False), file_name="dga_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process this file: {e}")

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Thermal and ageing core (synthetic load profile)")
    st.warning("Thermal parameters are placeholders and the starting ageing constant is an assumption. "
               "Replace them with the transformer's test-report values and measured data.")
    a, b = st.columns(2)
    with a:
        k_mean = st.slider("Mean load (p.u. of rated)", 0.2, 1.5, 0.8, 0.05)
        k_swing = st.slider("Daily load swing (p.u.)", 0.0, 0.5, 0.15, 0.05)
        t_mean = st.slider("Mean ambient (C)", -10.0, 40.0, 18.0, 1.0)
        t_season = st.slider("Seasonal ambient swing (C)", 0.0, 20.0, 12.0, 1.0)
    with b:
        upgraded = st.checkbox("Thermally upgraded paper", value=True)
        life_h = st.number_input("Assumed hours to DP 200 at 110 C hot spot", min_value=10000.0, value=180000.0, step=10000.0)
        measure = st.checkbox("Recalibrate with a furan-derived DP measurement")
        meas_year = meas_fal = None
        if measure:
            meas_year = st.number_input("Age at measurement (years)", min_value=1.0, value=8.0)
            meas_fal = st.number_input("2-FAL at measurement (ppm)", min_value=0.05, value=0.8)
    with st.expander("Thermal parameters (placeholders)"):
        e = st.columns(4)
        p = ThermalParams(
            dtheta_or=e[0].number_input("Top-oil rise at rated load (K)", value=55.0),
            dtheta_hr=e[1].number_input("Hot-spot rise over top-oil (K)", value=23.0),
            R=e[2].number_input("Loss ratio R", value=4.5),
            n=e[3].number_input("Oil exponent n", value=0.8),
            m=e[0].number_input("Winding exponent m", value=0.8),
            tau_o_min=e[1].number_input("Oil time constant (min)", value=180.0),
            tau_w_min=e[2].number_input("Winding time constant (min)", value=4.0))

    hours = 24 * 365
    t = np.arange(hours)
    day, year = 2 * np.pi * (t % 24) / 24, 2 * np.pi * t / hours
    K = np.clip(k_mean + k_swing * np.sin(day - 2.0), 0.0, None)
    theta_a = t_mean + t_season * np.cos(year - np.pi) + 4 * np.sin(day - 2.5)
    df = simulate(K, theta_a, 60.0, p)
    V = relative_ageing_rate(df.theta_h, upgraded=upgraded)
    exposure_year = float(np.sum(arrhenius_factor(df.theta_h)))

    m1, m2, m3 = st.columns(3)
    m1.metric("Max hot-spot (C)", f"{df.theta_h.max():.1f}")
    m2.metric("Mean hot-spot (C)", f"{df.theta_h.mean():.1f}")
    m3.metric("IEC equivalent ageing hours per year", f"{V.sum():.0f}")

    A = calibrate_A_from_life(110.0, life_h)
    yrs = np.arange(0, 61)
    dp_curve = 1.0 / (1.0 / 1000.0 + A * exposure_year * yrs)
    model = KineticAgeing(dp0=1000.0, A=A)
    note = "Projection uses the assumed ageing constant."
    if measure:
        dp_m = dp_chendong(meas_fal)
        model.E = exposure_year * meas_year
        if 0 < dp_m <= 1000:
            model.calibrate_to_measurement(dp_m)
            dp_curve = 1.0 / (1.0 / 1000.0 + model.A * exposure_year * yrs)
            note = f"Recalibrated to a furan-derived DP of {dp_m:.0f} at {meas_year:.0f} years."
        else:
            st.error("The furan-derived DP is outside 0-1000, so the measurement was ignored.")
            model.A = A
    else:
        model.E = 0.0
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(t[:24 * 7] / 24, df.theta_h[:24 * 7]); ax[0].set_title("Hot-spot, first week"); ax[0].set_xlabel("day"); ax[0].set_ylabel("C")
    ax[1].plot(yrs, dp_curve); ax[1].axhline(DP_END_OF_LIFE, color="gray", ls="--"); ax[1].set_title("DP projection"); ax[1].set_xlabel("years"); ax[1].set_ylim(0, 1050)
    st.pyplot(fig)
    if model.A is not None and model.E > 0:
        rul = remaining_life_hours(model, exposure_year / hours) / hours
        st.write(f"{note} Remaining time to DP {DP_END_OF_LIFE:.0f}: {rul:.1f} years from the measurement date.")
    else:
        st.write(note)

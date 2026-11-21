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
from pathlib import Path
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

tab1, tab_batch, tab2, tab3, tab_live = st.tabs(["DGA diagnosis", "Batch (CSV)", "Transformer twin", "Luo reproduction + twin", "Live rig"])

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

# ---------------- Tab 3: Luo et al. reproduction and the digital-twin pipeline ----------------
@st.cache_resource
def get_surrogates():
    """Filter surrogate, a finer-mesh 'truth' and a deliberately wrong-oil 'truth' (all from the ASSUMED model)."""
    from twin.surrogate import WindingSurrogate
    L, H = np.linspace(0.3, 1.4, 9), np.linspace(50, 110, 7)
    return (WindingSurrogate(nz=60).fit_from_model(L, H),
            WindingSurrogate(nz=80, nr_lv=12, nr_hv=16).fit_from_model(L, H),
            WindingSurrogate(nz=60).fit_from_model(L, H, t_top=47.0))

with tab3:
    from twin import report as rp, ansys_io as ai, pipeline_dp as pdp
    st.subheader("Reproduction of Luo et al. (2023): what is reproduced, fitted, or impossible")
    st.info("Reproduced: levelling-off DP 252.13 vs 252.15 and Eq. 13. Fitted: Table 2 row I (one rate constant on year 1). "
            "Impossible as printed: rows II and III (Eq. 11 gives k_DP <= 0 above about 0.62 % moisture), the 7.4x claim and the DP 250 target. "
            "Rows II and III below are VARIANTS with assumed moisture, not the paper's method.")
    st.dataframe(rp.luo_table2_frame())

    st.subheader("ANSYS temperature export to DP (assumed geometry)")
    st.warning("The bundled export comes from a placeholder two-winding model with a fitted film coefficient. In it the HV winding is hotter than "
               "the LV winding, which contradicts the paper. This shows the pipeline, not a result about a real transformer.")
    up3 = st.file_uploader("Your own ANSYS export (columns node,x_m,y_m,T_C), optional", type=["csv"], key="ansys_up")
    try:
        snap = ai.read_snapshot(up3 if up3 is not None else Path(__file__).parent / "examples" / "ansys" / "T_load0p74.csv")
        yrs3 = st.slider("Years of operation at this temperature", 1, 60, 48)
        var3 = st.selectbox("Moisture treatment", ["none", "eq13", "const0p70", "field"],
                            format_func=lambda v: {"none": "temperature only (fitted rate)", "eq13": "VARIANT: moisture from Eq. 13",
                                                   "const0p70": "VARIANT: constant 0.70 % moisture",
                                                   "field": "VARIANT: moisture field (equilibrium shape scaled to Eq. 13)"}[v])
        dm = pdp.dp_map_with_moisture(snap, yrs3) if var3 == "field" else pdp.dp_map_from_snapshot(snap, yrs3, var3)
        hs = pdp.hot_spot_summary(dm)
        c3 = st.columns(len(hs))
        for col, (w, v) in zip(c3, hs.items()):
            col.metric(f"{w} hottest node", f"{v['T']:.1f} C at {100 * v['y_rel']:.0f} % height", f"DP {v['DP']:.0f}", delta_color="off")
        fig3, ax3 = plt.subplots(1, 2, figsize=(9, 4))
        for a, col, ttl in [(ax3[0], "T_C", "temperature (C)"), (ax3[1], "DP", f"DP after {yrs3} years")]:
            sc = a.scatter(dm.x_m, dm.y_m, c=dm[col], s=5); fig3.colorbar(sc, ax=a); a.set_title(ttl); a.set_xlabel("radius (m)"); a.set_ylabel("height (m)")
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Could not process this file: {e}")

    st.subheader("Live calibration on SYNTHETIC data")
    st.warning("All sensor data here are invented. The filter adjusts the film coefficient so the model follows the hot-spot sensor. "
               "If the model is wrong, it hides the error in the parameter: calibration is not validation.")
    cc = st.columns(3)
    drift = cc[0].slider("True film coefficient at the end (W/m2K; starts at 72)", 50.0, 90.0, 58.0)
    qx3 = cc[1].select_slider("Filter process noise (per hour, log scale)", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
    wrong = cc[2].checkbox("Make the model wrong (oil 7 K warmer in the 'truth')")
    if st.button("Run the synthetic calibration"):
        with st.spinner("Building surrogates and running the filter..."):
            filt, truth_ok, truth_bad = get_surrogates()
            hours3 = 24 * 30
            out = rp.calibration_demo(filt, truth_bad if wrong else truth_ok, np.linspace(72.0, 72.0 if wrong else drift, hours3),
                                      hours=hours3, qx=float(qx3))
        m = st.columns(4)
        m[0].metric("Hot-spot RMSE, filter (K)", f"{out['rmse_filter']:.2f}")
        m[1].metric("Raw sensor noise (K)", f"{out['rmse_sensor']:.2f}")
        m[2].metric("Uncalibrated model (K)", f"{out['rmse_uncal']:.2f}")
        m[3].metric("Error at load 1.3 (K)", f"{out['overload_error']:+.1f}")
        fig4, ax4 = plt.subplots(figsize=(8, 3.2))
        ax4.plot(out["hf_true"], "k", label="true (hidden)"); ax4.plot(out["hf_est"], label="filter estimate")
        ax4.set_xlabel("hour"); ax4.set_ylabel("film coefficient (W/m2K)"); ax4.set_title("SYNTHETIC data"); ax4.legend()
        st.pyplot(fig4)

# ---------------- Tab 5: live twin (virtual rig or a CSV from the real logger) ----------------
with tab_live:
    import time as _time
    from twin import live as lv, rig as rg

    st.subheader("Live twin: the estimate updates itself as new readings arrive")
    st.caption("No parameter is typed in here. New rows arrive, the filter re-estimates the film coefficient, and the charts redraw. "
               "The only knobs are for the SIMULATED rig (what the 'world' does), not for the twin.")
    src = st.radio("Data source", ["Virtual rig (SYNTHETIC)", "Follow a CSV file"], horizontal=True, key="live_src")
    ctl = st.columns(4)
    if src.startswith("Virtual"):
        sched = ctl[0].selectbox("Load schedule", list(rg.SCHEDULES), key="live_sched",
                                 help="load_steps: 0.6, 1.0, 0.6, 1.3, 0.6 pu. sweep: 0.4 to 1.2 pu, 6 h each. daily: smooth 48 h profile.")
        speed = ctl[1].select_slider("Speed (rig seconds per real second)", options=[60, 150, 300, 600, 1200, 2400], value=600, key="live_speed")
        calib = ctl[2].checkbox("Probes calibrated (offsets known to the twin)", value=False, key="live_calib",
                                help="Simulated probes carry fixed offsets. Ticking this gives the twin those offsets, as step B0 would.")
        health = ctl[3].slider("Cooling health of the simulated rig (%)", 40, 100, 100, key="live_health",
                               help="Scales the true film coefficient. Drag it while the run is going and watch the estimate follow.")
    else:
        path = ctl[0].text_input("CSV path (logger output)", value="rig_run.csv", key="live_path")
        calib = False
    years_live = st.slider("Years for the DP projection at the present hot-spot temperature", 1, 48, 10, key="live_years")

    b = st.columns(3)
    if b[0].button("Start", key="live_start"):
        if st.session_state.get("live") is None:
            if src.startswith("Virtual"):
                source = lv.VirtualRigSource(schedule=sched, speed=float(speed), seed=11)
                twin = lv.LiveTwin(years=years_live, offsets=source.rig.offsets if calib else None)
            else:
                source, twin = lv.CsvTailSource(path), lv.LiveTwin(years=years_live)
            st.session_state["live"] = lv.LiveSession(source, twin)
        st.session_state["live_running"] = True
        st.session_state["live_last"] = _time.monotonic()
    if b[1].button("Stop", key="live_stop"):
        st.session_state["live_running"] = False
    if b[2].button("Reset", key="live_reset"):
        st.session_state["live"] = None
        st.session_state["live_running"] = False

    def live_panel():
        ses = st.session_state.get("live")
        if ses is None:
            st.info("Press Start. The virtual rig plays a simulated heated coil in oil faster than real time. "
                    "For a real rig, choose 'Follow a CSV file' and point it at the file the logger writes.")
            return
        if st.session_state.get("live_running"):
            now = _time.monotonic()
            dt = min(max(now - st.session_state.get("live_last", now), 0.0), 10.0)
            st.session_state["live_last"] = now
            if hasattr(ses.source, "rig"):
                ses.source.rig.h_scale = st.session_state.get("live_health", 100) / 100.0
            ses.step(dt)
        if ses.synthetic:
            st.warning("SYNTHETIC: every reading here is simulated, including sensor offsets, noise and dropouts. "
                       "This shows that the pipeline works. It is not a measurement and it validates no physics.")
        last = ses.latest()
        if last is None:
            st.write("Waiting for the first rows...")
            return
        run_state = "running" if st.session_state.get("live_running") else "stopped"
        st.caption(f"{run_state} | rows {len(ses.rows)} | rig time {last['time_s'] / 3600:.2f} h | filter updates {ses.twin.n_updates} | status: {last['status']}")
        m = st.columns(4)
        m[0].metric("Film coefficient (W/m2K)", f"{last['hf']:.1f}", f"+/- {last['hf_sd']:.1f} (statistical only)", delta_color="off")
        mt, pt = last.get("meas_T_paper_top", float("nan")), last.get("pred_T_paper_top", float("nan"))
        m[1].metric("Top thermocouple H1 (C)", f"{mt:.1f}", f"model {pt:.1f}", delta_color="off")
        ho = last.get("held_out_rms", float("nan"))
        m[2].metric("Held-out error H2, H3 (K)", "n/a (settling)" if not np.isfinite(ho) else f"{ho:.2f}", help="H2 and H3 are not used by the filter.")
        dp = last.get("dp_hot_spot", float("nan"))
        m[3].metric(f"DP after {ses.twin.years} y at this temperature", "n/a" if not np.isfinite(dp) else f"{dp:.0f}", help="Model output from the Luo-style chain, moisture ignored.")
        h = ses.twin.history_frame().tail(720)
        h["hours"] = h["time_s"] / 3600.0
        c1, c2 = st.columns(2)
        c1.caption("Top thermocouple: measured and model (C)")
        c1.line_chart(h.set_index("hours")[["meas_T_paper_top", "pred_T_paper_top"]], height=260)
        hf_cols = h.set_index("hours")[["hf"]].rename(columns={"hf": "estimate"})
        if ses.synthetic and hasattr(ses.source, "rig"):
            tr = ses.source.rig.truth_frame().tail(len(h))
            if len(tr) == len(h):
                hf_cols["true (hidden, SYNTHETIC only)"] = tr["h_true"].to_numpy()
        c2.caption("Film coefficient (W/m2K)")
        c2.line_chart(hf_cols, height=260)
        row = ses.rows.iloc[-1]
        zs = list(rg.OIL_Z_MM)
        fig_l, ax_l = plt.subplots(figsize=(4.2, 3.2))
        ax_l.plot([row[f"T_oil_{i}"] - ses.twin.offsets.get(f"T_oil_{i}", 0.0) for i in range(1, 7)], zs, "o-", label="oil probes T1..T6")
        ax_l.plot([row[c] - ses.twin.offsets.get(c, 0.0) for c in rg.HOT_Z_MM], list(rg.HOT_Z_MM.values()), "D", color="tab:red", label="paper thermocouples H3..H1")
        ax_l.set_xlabel("temperature (C)"); ax_l.set_ylabel("height z (mm)"); ax_l.legend(fontsize=7)
        st.pyplot(fig_l)
        plt.close(fig_l)
        if str(row.get("note", "")) == "THERMOSTAT_OPEN":
            st.error("Thermostat is open: the heater is cut off.")

    frag = getattr(st, "fragment", None)
    if frag is not None:
        frag(run_every=2 if st.session_state.get("live_running") else None)(live_panel)()
    else:
        st.button("Refresh", key="live_refresh")
        live_panel()

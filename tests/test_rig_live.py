"""Tests for the SYNTHETIC virtual rig (twin.rig) and the live estimator (twin.live).

These tests check that the code does what it says. They do not validate any physics: the virtual rig is a model, and a
model tested against its own simulated data agrees with itself by construction.
"""
import math
import numpy as np
import pandas as pd
import pytest

from twin.rig import (VirtualRig, RigDesign, COLUMNS, OIL_Z_MM, HOT_Z_MM, write_csv, read_csv, schedule_segments, SCHEDULES)
from twin.live import LiveTwin, LiveSession, VirtualRigSource, CsvTailSource

H = 3600.0
IDEAL = dict(offset_scale=0, noise_scale=0, dropout=0, r_err=0)


def run_rig(schedule, hours, seed=1, **kw):
    rig = VirtualRig(schedule=schedule, loop=False, seed=seed, **kw)
    return rig, rig.advance(hours * H)


# ---------------- the virtual rig ----------------
def test_rows_have_logger_columns_and_are_flagged_synthetic():
    rig, df = run_rig("load_steps", 1.0)
    assert list(df.columns) == COLUMNS
    assert len(df) == 360                                  # one row per 10 s
    assert rig.synthetic is True and "SYNTHETIC" in rig.note
    assert OIL_Z_MM == (55, 105, 155, 205, 255, 305)       # same heights as the rig page
    assert HOT_Z_MM == {"T_paper_bot": 75, "T_paper_mid": 155, "T_paper_top": 235}


@pytest.mark.parametrize("name", SCHEDULES)
def test_schedules_exist_and_have_positive_duration(name):
    segs = schedule_segments(name)
    assert segs and all(d > 0 and 0.3 <= pu <= 1.4 for d, pu in segs)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        schedule_segments("nope")


def test_steady_state_energy_balance():
    """After 40 h at 1.0 pu: oil rise = P/UA and coil-to-oil drop = P/(A U), where U = 1/(1/h + R_paper)."""
    rig, _ = run_rig([(40 * H, 1.0)], 40, **IDEAL)
    tr, d = rig.truth_frame(), rig.d
    last = tr.iloc[-1]
    assert last.T_oil_mean - 25.0 == pytest.approx(last.P_true_W / d.ua_w_per_k, rel=0.005)
    U = 1.0 / (1.0 / last.h_true + d.r_paper)
    assert last.T_coil - last.T_oil_mean == pytest.approx(last.P_true_W / (d.area_m2 * U), rel=0.005)
    assert last.P_true_W == pytest.approx(d.p_rated_w, rel=1e-6)           # 1.0 pu = 40 W when r_err = 0


def test_oil_time_constant_is_c_over_ua():
    rig, _ = run_rig([(40 * H, 1.0)], 40, **IDEAL)
    tr, d = rig.truth_frame(), rig.d
    tau = d.c_oil_j_per_k / d.ua_w_per_k
    frac = (tr.T_oil_mean.iloc[int(tau / 10)] - 25.0) / (tr.T_oil_mean.iloc[-1] - 25.0)
    assert 0.60 < frac < 0.66                              # 0.632 for a pure first-order response; h(dT) bends it a little


def test_hot_spot_is_above_oil_and_grows_with_height():
    rig, df = run_rig([(30 * H, 1.0)], 30, **IDEAL)
    r = df.iloc[-1]
    oil = [r[f"T_oil_{i}"] for i in range(1, 7)]
    assert all(np.diff(oil) > 0)                           # warmer towards the top
    assert r.T_paper_bot < r.T_paper_mid < r.T_paper_top
    assert r.T_paper_top > r.T_oil_4 + 10                  # film + paper drop is tens of kelvin at 1.0 pu


def test_measured_power_follows_the_real_coil_not_the_setpoint():
    """The supply is set from the DESIGN resistance; a coil that is 2 % off draws 40/1.02 W. P_W reports that."""
    _, df = run_rig([(1 * H, 1.0)], 1, offset_scale=0, noise_scale=0, dropout=0, r_err=0.02)
    assert df.P_W.mean() == pytest.approx(40.0 / 1.02, rel=1e-3)
    assert df.setpoint_pu.iloc[0] == 1.0


def test_quantisation_steps():
    _, df = run_rig([(3 * H, 0.8)], 3, offset_scale=0, noise_scale=0, dropout=0)
    ds = df[[f"T_oil_{i}" for i in range(1, 7)]].to_numpy() / 0.0625
    assert np.allclose(ds, np.round(ds))                   # DS18B20 steps of 0.0625 C
    tc = df.T_paper_top.to_numpy() / 0.0078125
    assert np.allclose(tc, np.round(tc))


def test_sensor_noise_level():
    _, df = run_rig([(60 * H, 0.5)], 60, offset_scale=0, dropout=0)
    x = df.T_oil_3.to_numpy()[-400:]
    assert np.std(np.diff(x)) / math.sqrt(2) == pytest.approx(0.10, rel=0.25)


def test_dropouts_are_nan_at_the_requested_rate():
    _, df = run_rig([(20 * H, 0.8)], 20, dropout=0.05, seed=7)
    frac = df.T_oil_2.isna().mean()
    assert 0.035 < frac < 0.065


def test_offsets_are_fixed_per_probe_and_zero_when_calibrated():
    a = VirtualRig(seed=5)
    assert any(abs(v) > 0 for v in a.offsets.values())
    b = VirtualRig(seed=5, offset_scale=0)
    assert all(v == 0 for v in b.offsets.values())


def test_thermostat_opens_and_closes():
    big = RigDesign(p_rated_w=200.0)
    rig, df = run_rig([(20 * H, 1.0)], 20, design=big, **IDEAL)
    tr = rig.truth_frame()
    opened = (df.note == "THERMOSTAT_OPEN").to_numpy()
    assert opened.any()
    assert tr.hot_spot_true.max() < 102.0                  # overshoot after opening stays small
    assert tr.hot_spot_true[opened].min() < 95.5           # it closes again once the coil is below 95 C
    assert (df.P_W[opened] < 1.0).all()


def test_same_seed_same_data_different_seed_different_data():
    a = VirtualRig(seed=3).advance(600)
    b = VirtualRig(seed=3).advance(600)
    c = VirtualRig(seed=4).advance(600)
    pd.testing.assert_frame_equal(a, b)
    assert not a.T_oil_1.equals(c.T_oil_1)


def test_advance_in_pieces_equals_one_call():
    a = VirtualRig(seed=3).advance(3600)
    r = VirtualRig(seed=3)
    b = pd.concat([r.advance(1000), r.advance(1000), r.advance(1600)], ignore_index=True)
    pd.testing.assert_frame_equal(a, b)


# ---------------- logger CSV ----------------
def test_csv_roundtrip_keeps_columns_and_nan(tmp_path):
    _, df = run_rig([(2 * H, 0.8)], 2, dropout=0.1, seed=2)
    p = tmp_path / "run.csv"
    write_csv(df, p)
    back = read_csv(p)
    assert list(back.columns) == COLUMNS
    assert len(back) == len(df)
    assert back.T_oil_1.isna().sum() == df.T_oil_1.isna().sum() > 0
    np.testing.assert_allclose(back.P_W.to_numpy(), df.P_W.to_numpy())


def test_read_csv_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"time_s": [0.0], "P_W": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        read_csv(p)


# ---------------- the live estimator ----------------
def rel_err(est, true):
    return abs(est - true) / true


def sweep_errors(twin_kw=None, **rig_kw):
    rig = VirtualRig(schedule="sweep", loop=False, seed=2, **rig_kw)
    df = rig.advance(rig.total)
    tw = LiveTwin(**(twin_kw or {}))
    tw.ingest_frame(df)
    h, tr = tw.history_frame(), rig.truth_frame()
    idx = [int((k + 1) * 6 * H / 10) - 1 for k in range(5)]
    return [rel_err(h.hf[i], tr.h_true[i]) for i in idx], tw, h, df, rig


def test_estimator_recovers_film_coefficient_with_ideal_probes():
    errs, tw, h, df, rig = sweep_errors(offset_scale=0)
    assert max(errs) < 0.03                                # observed about 1 %
    assert tw.n_updates > 5000


def test_probe_offsets_bias_the_estimate_and_calibration_removes_it():
    e_uncal, *_ = sweep_errors(offset_scale=1)
    rig = VirtualRig(seed=2, offset_scale=1)               # same seed gives the same offsets
    e_cal, *_ = sweep_errors(twin_kw=dict(offsets=rig.offsets), offset_scale=1)
    assert np.mean(e_cal) < np.mean(e_uncal)
    assert max(e_cal) < 0.03


def test_updates_are_skipped_while_the_power_changes():
    rig = VirtualRig(schedule=[(3 * H, 0.6), (3 * H, 1.0)], loop=False, seed=2, offset_scale=0)
    df = rig.advance(6 * H)
    tw = LiveTwin()
    tw.ingest_frame(df)
    h = tw.history_frame()
    step_i = int(3 * H / 10)
    just_after = h.status.iloc[step_i + 2: step_i + 30]
    assert just_after.str.startswith("settling").all()     # first ~5 minutes after the step
    assert h.status.iloc[step_i + 100] == "tracking"       # more than 10 minutes later
    assert h.status.iloc[step_i - 5] == "tracking"


def test_estimator_follows_a_change_in_cooling_without_any_manual_edit():
    """Fouling: the hidden film coefficient drops by 30 % at 15 h. The estimate follows on its own."""
    rig = VirtualRig(schedule=[(100 * H, 1.0)], loop=False, seed=4, offset_scale=0)
    tw = LiveTwin()
    tw.ingest_frame(rig.advance(15 * H))
    before = tw.hf
    rig.h_scale = 0.7
    tw.ingest_frame(rig.advance(15 * H))
    tr = rig.truth_frame()
    true_ratio = tr.h_true.iloc[-1] / tr.h_true.iloc[int(15 * H / 10) - 1]
    assert true_ratio == pytest.approx(0.75, abs=0.03)     # natural-convection exponent softens the drop a little
    assert tw.hf / before == pytest.approx(true_ratio, abs=0.04)


def test_held_out_thermocouple_catches_a_faulty_probe():
    rig = VirtualRig(schedule=[(20 * H, 0.8)], loop=False, seed=6, offset_scale=0)
    df = rig.advance(20 * H)
    good = LiveTwin(); good.ingest_frame(df)
    bad_df = df.copy(); bad_df["T_paper_mid"] = bad_df["T_paper_mid"] + 2.0       # sensor fault, +2 K
    bad = LiveTwin(); bad.ingest_frame(bad_df)
    g, b = good.history_frame(), bad.history_frame()
    assert g.held_out_rms.iloc[-200:].mean() < 0.4
    assert b.held_out_rms.iloc[-200:].mean() > 1.0         # raised by the faulty held-out sensor
    assert b.hf.iloc[-1] == pytest.approx(g.hf.iloc[-1], rel=1e-6)     # and the fit sensor is untouched


def test_wrong_paper_resistance_is_hidden_in_the_parameter():
    """Calibration is not validation: with R_paper twice too large the fit sensor still agrees, but h is wrong."""
    rig = VirtualRig(schedule=[(20 * H, 1.0)], loop=False, seed=6, offset_scale=0)
    df = rig.advance(20 * H)
    wrong = LiveTwin(design=RigDesign(r_paper=0.010))
    wrong.ingest_frame(df)
    h = wrong.history_frame()
    tr = rig.truth_frame()
    assert abs(h.resid_T_paper_top.iloc[-100:]).mean() < 0.3          # looks fine on the fitted sensor
    assert rel_err(h.hf.iloc[-1], tr.h_true.iloc[-1]) > 0.10          # but the physical parameter is off


def test_dropouts_do_not_break_the_estimator():
    rig = VirtualRig(schedule=[(20 * H, 0.9)], loop=False, seed=8, offset_scale=0, dropout=0.10)
    df = rig.advance(20 * H)
    tw = LiveTwin()
    out = tw.ingest_frame(df)
    assert len(out) == len(df)
    assert np.isfinite(tw.hf)
    assert rel_err(tw.hf, rig.truth_frame().h_true.iloc[-1]) < 0.05
    assert set(tw.history_frame().status) >= {"tracking"}


def test_missing_power_or_probes_give_a_status_not_an_exception():
    tw = LiveTwin()
    row = {c: float("nan") for c in COLUMNS if c != "note"}
    row.update(time_s=0.0, note="")
    assert tw.ingest(row)["status"] == "no power reading"
    row.update(time_s=10.0, P_W=40.0)
    assert tw.ingest(row)["status"] == "too few oil probes"


def test_dp_responds_to_temperature():
    rig = VirtualRig(schedule=[(8 * H, 0.4), (30 * H, 1.0)], loop=False, seed=2, offset_scale=0)
    tw = LiveTwin(years=10)
    d1 = tw.ingest_frame(rig.advance(8 * H))[-1]["dp_hot_spot"]
    d2 = tw.ingest_frame(rig.advance(30 * H))[-1]["dp_hot_spot"]
    assert d2 < d1                                         # hotter coil, lower DP after the same number of years


# ---------------- sources and the session ----------------
def test_csv_tail_source_returns_only_new_rows(tmp_path):
    _, df = run_rig([(1 * H, 0.8)], 1, seed=2)
    p = tmp_path / "live.csv"
    src = CsvTailSource(p)
    assert len(src.poll()) == 0                            # file does not exist yet
    write_csv(df.iloc[:5], p)
    assert len(src.poll()) == 5
    with open(p, "a", encoding="utf-8") as f:              # logger appends three complete lines
        f.write(df.iloc[5:8].to_csv(header=False, index=False))
    got = src.poll()
    assert len(got) == 3 and got.time_s.iloc[0] == df.time_s.iloc[5]
    line = df.iloc[8:9].to_csv(header=False, index=False).rstrip("\n")
    with open(p, "a", encoding="utf-8") as f:              # half-written line: no newline yet
        f.write(line[:30])
    assert len(src.poll()) == 0
    with open(p, "a", encoding="utf-8") as f:
        f.write(line[30:] + "\n")
    got = src.poll()
    assert len(got) == 1 and got.time_s.iloc[0] == df.time_s.iloc[8]
    assert len(src.poll()) == 0                            # nothing is delivered twice


def test_session_with_virtual_source_updates_itself():
    ses = LiveSession(VirtualRigSource(schedule="sweep", speed=600.0, seed=2, offset_scale=0))
    assert ses.synthetic is True
    n = [ses.step(2.0) for _ in range(30)]
    assert all(x == 120 for x in n)                        # 2 s x 600 = 1200 rig seconds = 120 rows
    assert len(ses.rows) == 3600
    last = ses.latest()
    assert last["status"] in {"tracking", "settling: power changing, update skipped"}
    assert ses.twin.n_updates > 0 and np.isfinite(last["hf"])


def test_session_replaying_a_csv_matches_running_it_directly(tmp_path):
    rig = VirtualRig(schedule=[(6 * H, 0.8)], loop=False, seed=2, offset_scale=0)
    df = rig.advance(6 * H)
    p = tmp_path / "replay.csv"
    write_csv(df, p)
    ses = LiveSession(CsvTailSource(p))
    assert ses.synthetic is False
    ses.step(0)
    direct = LiveTwin(); direct.ingest_frame(read_csv(p))
    assert ses.twin.hf == pytest.approx(direct.hf, rel=1e-12)
    assert ses.twin.n_updates == direct.n_updates


def test_estimator_state_is_not_shared_between_sessions():
    a = LiveSession(VirtualRigSource(schedule="sweep", seed=2, offset_scale=0))
    b = LiveSession(VirtualRigSource(schedule="sweep", seed=2, offset_scale=0))
    for _ in range(20):
        a.step(2.0)
    assert b.twin.n_updates == 0 and len(b.rows) == 0

"""Live twin: data sources and an estimator that updates itself as new rig readings arrive.

Data flow (no manual parameter edits):
    source (virtual rig, or a CSV that the real logger keeps appending to)
      -> LiveTwin.ingest(row): corrects probe offsets, interpolates the measured oil temperature at the H1 height,
         updates the film coefficient with the extended Kalman filter, predicts H1..H3, computes DP
      -> LiveSession keeps the history that the dashboard draws.

What the estimator does (and does not do):
 * Model: T_H(z) = T_oil_measured(z) + q'' (1/h + R_paper), q'' = P / A_coil. The unknown is h. The oil temperature is
   MEASURED (six probes), so no oil model is needed. R_paper and the coil area are design values (ASSUMED).
 * H1 (top thermocouple) is the only sensor the filter is fitted to. H2 and H3 are HELD OUT: their residuals are a
   check, not a fit. In uniform-flux physics the three share one temperature drop, so they catch probe offsets and
   noise, not a wrong R_paper (that error is absorbed in h: calibration is not validation).
 * Updates are skipped while the power is changing (the coil needs minutes to settle). Status says so.
 * DP is the Luo-style chain from twin.pipeline_dp (rate constant fitted to Table 2 row I, moisture ignored). It is a
   model output driven by the measured temperature, not a measurement.
"""
from collections import deque
import io
import math
import numpy as np
import pandas as pd

from twin.kalman import EKF
from twin.rig import RigDesign, DEFAULT, OIL_Z_MM, HOT_Z_MM, COLUMNS, VirtualRig, read_csv

HOLD_OUT = ("T_paper_mid", "T_paper_bot")      # not used by the filter
FIT_SENSOR = "T_paper_top"

class LiveTwin:
    def __init__(self, design=DEFAULT, offsets=None, hf0=70.0, sigma=0.3, qx=2e-3, steady_window_s=600.0,
                 steady_tol=0.03, years=48, p0_log_sd=0.5, max_history=30000):
        self.d = design
        self.offsets = dict(offsets or {})                 # per-channel offsets from the calibration step (B0)
        self.sigma, self.years = float(sigma), int(years)
        self.window, self.tol = float(steady_window_s), float(steady_tol)
        self._z_hot = HOT_Z_MM[FIT_SENSOR]
        self._pbuf = deque()                               # (time, power)
        r_p, area = design.r_paper, design.area_m2
        self._r_p, self._area = r_p, area
        self.ekf = EKF([math.log(hf0)], [[p0_log_sd ** 2]], [[qx ** 2]], self.sigma ** 2,
                       lambda x, u: u[0] + u[1] * (1.0 / math.exp(float(x[0])) + r_p))
        self.n_updates = 0
        self.history = deque(maxlen=max_history)           # rolling window, so a long run does not grow without bound
        from twin.pipeline_dp import TABLE2_I_YEAR1
        from twin.luo_chain import fit_k_scale
        self._k_scale = fit_k_scale(TABLE2_I_YEAR1)

    # ----- helpers -----
    def _corr(self, row, name):
        v = row.get(name, float("nan"))
        return float(v) - self.offsets.get(name, 0.0) if v is not None and np.isfinite(v) else float("nan")

    def oil_profile(self, row):
        """Offset-corrected oil temperatures [C] at the probe heights; NaN where a probe dropped out."""
        return np.array([self._corr(row, f"T_oil_{i}") for i in range(1, 7)])

    def _oil_at(self, row, z_mm):
        T = self.oil_profile(row)
        ok = np.isfinite(T)
        if ok.sum() < 2:
            return float("nan")
        return float(np.interp(z_mm, np.array(OIL_Z_MM, float)[ok], T[ok]))

    @property
    def hf(self):
        return math.exp(float(self.ekf.x[0]))

    @property
    def hf_sd(self):
        return self.hf * math.sqrt(float(self.ekf.P[0, 0]))

    def predict(self, row, sensor, hf=None):
        """Model value of a hot-spot thermocouple from measured oil temperature, power and h."""
        hf = self.hf if hf is None else hf
        q = float(row["P_W"]) / self._area
        return self._oil_at(row, HOT_Z_MM[sensor]) + q * (1.0 / hf + self._r_p)

    def _steady(self, t):
        b = self._pbuf
        if len(b) < 2 or b[-1][0] - b[0][0] < self.window - 1e-6:
            return False
        p = np.array([x[1] for x in b])
        return float(p.mean()) > 0 and (p.max() - p.min()) / p.mean() < self.tol

    # ----- main entry -----
    def ingest(self, row, with_dp=True):
        row = dict(row)
        t = float(row["time_s"])
        P = row.get("P_W", float("nan"))
        st = dict(time_s=t, status="", hf=self.hf, hf_sd=self.hf_sd, steady=False)
        if not np.isfinite(P):
            st["status"] = "no power reading"
            self.history.append(st); return st
        self._pbuf.append((t, float(P)))
        while self._pbuf and t - self._pbuf[0][0] > self.window:
            self._pbuf.popleft()
        oil_loc = self._oil_at(row, self._z_hot)
        z_meas = self._corr(row, FIT_SENSOR)
        steady = self._steady(t)
        st["steady"] = steady
        if not np.isfinite(oil_loc):
            st["status"] = "too few oil probes"
        elif not np.isfinite(z_meas):
            st["status"] = "top thermocouple missing"
        elif P < 1.0:
            st["status"] = "heater off"
        elif not steady:
            st["status"] = "settling: power changing, update skipped"
        else:
            self.ekf.step(z_meas, (oil_loc, float(P) / self._area))
            self.n_updates += 1
            st["status"] = "tracking"
        st["hf"], st["hf_sd"] = self.hf, self.hf_sd
        st["P_W"] = float(P)
        st["oil_top_C"] = self._oil_at(row, 305)
        for c in HOT_Z_MM:
            m = self._corr(row, c)
            pred = self.predict(row, c) if np.isfinite(oil_loc) or np.isfinite(self._oil_at(row, HOT_Z_MM[c])) else float("nan")
            st[f"meas_{c}"], st[f"pred_{c}"] = m, pred
            st[f"resid_{c}"] = (m - pred) if (steady and np.isfinite(m) and np.isfinite(pred)) else float("nan")
        st["held_out_rms"] = self._rms([st[f"resid_{c}"] for c in HOLD_OUT])
        if with_dp and np.isfinite(st["meas_T_paper_top"]):
            from twin.pipeline_dp import dp_profile
            st["dp_hot_spot"] = float(dp_profile(st["meas_T_paper_top"], self.years, "none", self._k_scale))
        self.history.append(st)
        return st

    @staticmethod
    def _rms(vals):
        v = np.array([x for x in vals if np.isfinite(x)])
        return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")

    def ingest_frame(self, df):
        """Process a DataFrame of rows; DP is computed only for the last row of the batch."""
        out = []
        n = len(df)
        for k, row in enumerate(df.to_dict("records")):
            out.append(self.ingest(row, with_dp=(k == n - 1)))
        return out

    def history_frame(self):
        return pd.DataFrame(list(self.history))


# ----- sources -----
class VirtualRigSource:
    """Plays a VirtualRig faster than real time. speed = rig seconds per wall-clock second. SYNTHETIC."""
    synthetic = True
    label = "Virtual rig (SYNTHETIC)"

    def __init__(self, rig=None, speed=600.0, **rig_kw):
        self.rig = rig or VirtualRig(**rig_kw)
        self.speed = float(speed)

    def poll(self, wall_dt):
        return self.rig.advance(float(wall_dt) * self.speed)


class CsvTailSource:
    """Follows a CSV that the logger keeps appending to (or replays a finished file). Each poll returns only new rows.
    A half-written last line is ignored until its newline arrives."""
    synthetic = False
    label = "CSV file"

    def __init__(self, path):
        self.path, self._n = path, 0

    def poll(self, wall_dt=None):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            return pd.DataFrame(columns=COLUMNS)
        if text and not text.endswith("\n"):
            text = text[: text.rfind("\n") + 1]
        if not text.strip():
            return pd.DataFrame(columns=COLUMNS)
        df = read_csv(io.StringIO(text))
        new = df.iloc[self._n:]
        self._n = len(df)
        return new


class LiveSession:
    """One live run: a source, a twin and the rolling data the dashboard draws."""

    def __init__(self, source, twin=None, max_rows=30000):
        self.source, self.twin, self.max_rows = source, twin or LiveTwin(), max_rows
        self.rows = pd.DataFrame(columns=COLUMNS)
        self.synthetic = bool(getattr(source, "synthetic", False))
        self.polls = 0

    def step(self, wall_dt):
        """Pull new rows, update the twin, return the number of new rows."""
        new = self.source.poll(wall_dt)
        self.polls += 1
        if len(new):
            self.twin.ingest_frame(new)
            self.rows = pd.concat([self.rows, new], ignore_index=True).iloc[-self.max_rows:]
        return len(new)

    def latest(self):
        return self.twin.history[-1] if self.twin.history else None

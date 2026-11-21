"""VIRTUAL test rig: a lumped simulation of the heated-coil-in-oil bench rig, with simulated sensors.

EVERYTHING HERE IS SYNTHETIC. No number below is a measurement, and none comes from Luo et al. The values are design
choices from the rig page (coil, 10.5 ohm, 40 W at 1.0 pu) plus typical textbook-order guesses, each marked ASSUMED.
The purpose is to develop and test the live pipeline (logger format, loader, estimator, dashboard) before the real rig
exists. A model tested against its own simulated data always agrees with itself, so nothing produced here validates
any physics.

Model (two lumped nodes, explicit Euler):
  coil node  C_c dT_c/dt = P - Q_out
  oil node   C_o dT_o/dt = Q_out - UA (T_o - T_amb)
  Q_out = A_c (T_c - T_o) / (1/h + R_paper)
  h = h_ref * (film_dT / dT_ref)^0.25    (ASSUMED natural-convection-like dependence on the film temperature drop)
Oil temperature along height is linear around the mean: T_oil(z) = T_o + beta (T_o - T_amb)(z/H_oil - 0.5)   (ASSUMED beta)
Hot-spot temperature at height z: T_H(z) = T_oil(z) + (T_c - T_o).

Sensor model: fixed offset per probe (an UNCALIBRATED probe), Gaussian noise, quantisation step and random dropouts
(NaN). A simulated hardware thermostat opens at 100 C and closes again below 95 C.
"""
from dataclasses import dataclass
from collections import deque
import math
import numpy as np
import pandas as pd

OIL_Z_MM = (55, 105, 155, 205, 255, 305)                        # oil probe heights (rig page, side view)
HOT_Z_MM = {"T_paper_bot": 75, "T_paper_mid": 155, "T_paper_top": 235}   # H3, H2, H1
COLUMNS = (["time_s", "setpoint_pu", "P_W", "V", "I"] + [f"T_oil_{i}" for i in range(1, 7)]
           + ["T_paper_top", "T_paper_mid", "T_paper_bot", "T_amb", "T_wall", "note"])
SYNTHETIC_NOTE = "SYNTHETIC: simulated rig and simulated sensors; not a measurement"

@dataclass(frozen=True)
class RigDesign:
    """Design values from the rig page (proposed, not built) and ASSUMED physical constants."""
    r_coil_m: float = 0.032            # paper surface radius (OD 64)
    height_active_m: float = 0.200     # active winding length
    oil_height_m: float = 0.320
    r_nominal_ohm: float = 10.5        # design resistance of the nichrome coil
    p_rated_w: float = 40.0            # power at 1.0 pu
    oil_volume_l: float = 9.25         # fill 320 mm in a 200 mm tank minus the coil
    rho_oil: float = 870.0             # kg/m3, ASSUMED typical mineral oil
    cp_oil: float = 1900.0             # J/kgK, ASSUMED typical
    ua_w_per_k: float = 2.0            # tank-to-room loss, ASSUMED (uninsulated glass or steel tank)
    c_coil_j_per_k: float = 350.0      # tube + wire + paper, rough estimate
    r_paper: float = 0.005             # m2K/W, 1 mm paper at k = 0.2, ASSUMED
    h_ref: float = 70.0                # W/m2K at film_dT = dT_ref, ASSUMED
    dT_ref: float = 20.0
    h_exp: float = 0.25                # ASSUMED
    beta: float = 0.4                  # top-minus-bottom oil spread as a fraction of (T_oil - T_amb), ASSUMED

    @property
    def area_m2(self):
        return 2 * math.pi * self.r_coil_m * self.height_active_m

    @property
    def c_oil_j_per_k(self):
        return self.rho_oil * self.cp_oil * self.oil_volume_l * 1e-3

DEFAULT = RigDesign()

def schedule_segments(name):
    """List of (duration_s, load_pu). 'sweep' holds each level 6 h (the oil needs hours to settle)."""
    h = 3600.0
    if name == "sweep":
        return [(6 * h, pu) for pu in (0.4, 0.6, 0.8, 1.0, 1.2)]
    if name == "load_steps":
        return [(2 * h, 0.6), (2 * h, 1.0), (2 * h, 0.6), (1 * h, 1.3), (2 * h, 0.6)]
    if name == "daily":
        return [(600.0, 0.75 + 0.15 * math.sin(2 * math.pi * k / 144.0 - 2.0)) for k in range(288)]   # 48 h
    raise ValueError(f"unknown schedule {name!r}")

SCHEDULES = ("load_steps", "sweep", "daily")

def _quant(x, step):
    return np.round(x / step) * step

class VirtualRig:
    """Simulated rig. advance(seconds) returns the new logger rows as a DataFrame with the COLUMNS above."""
    synthetic = True
    note = SYNTHETIC_NOTE

    def __init__(self, design=DEFAULT, schedule="load_steps", dt_sample=10.0, seed=0, t_amb=25.0, amb_swing=0.0,
                 offset_scale=1.0, noise_scale=1.0, dropout=0.002, r_err=0.02, loop=True):
        self.d, self.dt_sample, self.loop = design, float(dt_sample), loop
        self.segments = schedule_segments(schedule) if isinstance(schedule, str) else list(schedule)
        self.total = sum(s for s, _ in self.segments)
        self.rng = np.random.default_rng(seed)
        self.t_amb0, self.amb_swing = t_amb, amb_swing
        self.noise_scale, self.dropout = noise_scale, dropout
        self.r_true = design.r_nominal_ohm * (1.0 + r_err)       # the real coil is not exactly the design value
        self.n_sub = max(1, math.ceil(self.dt_sample / 2.0))
        self.dt_sim = self.dt_sample / self.n_sub
        # ASSUMED sensor errors. offset_scale = 0 means perfectly calibrated probes.
        sig = dict(oil=0.25, tc=0.5, room=0.25)
        self.offsets = {}
        for i in range(1, 7):
            self.offsets[f"T_oil_{i}"] = offset_scale * sig["oil"] * self.rng.standard_normal()
        for c in HOT_Z_MM:
            self.offsets[c] = offset_scale * sig["tc"] * self.rng.standard_normal()
        for c in ("T_amb", "T_wall"):
            self.offsets[c] = offset_scale * sig["room"] * self.rng.standard_normal()
        self._noise = {**{f"T_oil_{i}": 0.10 for i in range(1, 7)}, **{c: 0.15 for c in HOT_Z_MM}, "T_amb": 0.10, "T_wall": 0.10}
        self._step = {**{f"T_oil_{i}": 0.0625 for i in range(1, 7)}, **{c: 0.0078125 for c in HOT_Z_MM}, "T_amb": 0.0625, "T_wall": 0.0625}
        # state
        self.t = 0.0
        self.T_o = self.T_c = t_amb
        self._q_out = 0.0
        self._h = design.h_ref * (0.5 / design.dT_ref) ** design.h_exp
        self._open = False
        self.h_scale = 1.0                 # hidden fouling knob: multiplies the film coefficient (change it while running)
        self._pending = 0.0
        self._truth = deque(maxlen=30000)    # rolling window, same length as the twin's history

    # ----- schedule and plant -----
    def load_at(self, t):
        if self.loop and self.total > 0:
            t = t % self.total
        acc = 0.0
        for dur, pu in self.segments:
            acc += dur
            if t < acc:
                return pu
        return self.segments[-1][1]

    def t_amb(self, t):
        return self.t_amb0 + self.amb_swing * math.sin(2 * math.pi * t / 86400.0)

    def _oil_at(self, z_mm, T_o, T_amb):
        return T_o + self.d.beta * (T_o - T_amb) * (z_mm * 1e-3 / self.d.oil_height_m - 0.5)

    def _substep(self, pu):
        d = self.d
        Ta = self.t_amb(self.t)
        hot = self._oil_at(235, self.T_o, Ta) + (self.T_c - self.T_o)
        if self._open and hot < 95.0:
            self._open = False
        elif not self._open and hot > 100.0:
            self._open = True
        V = 0.0 if self._open else math.sqrt(d.p_rated_w * pu ** 2 * d.r_nominal_ohm)    # the supply is set from the DESIGN R
        P = V * V / self.r_true
        film_dT = (self._q_out / d.area_m2) / self._h
        self._h = self.h_scale * d.h_ref * (max(film_dT, 0.5) / d.dT_ref) ** d.h_exp
        U = 1.0 / (1.0 / self._h + d.r_paper)
        q_out = d.area_m2 * U * (self.T_c - self.T_o)
        dt = self.dt_sim
        self.T_c += (P - q_out) / d.c_coil_j_per_k * dt
        self.T_o += (q_out - d.ua_w_per_k * (self.T_o - Ta)) / d.c_oil_j_per_k * dt
        self._q_out = q_out
        self.t += dt
        return V, P

    def _sense(self, name, value):
        v = value + self.offsets[name] + self._noise[name] * self.noise_scale * self.rng.standard_normal()
        v = float(_quant(v, self._step[name]))
        return float("nan") if self.rng.random() < self.dropout else v

    def _sample_row(self, pu, V, P):
        d = self.d
        Ta = self.t_amb(self.t)
        row = {"time_s": round(self.t, 3), "setpoint_pu": round(pu, 4)}
        Vm = V + 0.01 * self.noise_scale * self.rng.standard_normal()
        Im = V / self.r_true + 0.005 * self.noise_scale * self.rng.standard_normal()
        row.update(V=round(Vm, 3), I=round(Im, 4), P_W=round(Vm * Im, 3))
        for i, z in enumerate(OIL_Z_MM, 1):
            row[f"T_oil_{i}"] = self._sense(f"T_oil_{i}", self._oil_at(z, self.T_o, Ta))
        for c, z in HOT_Z_MM.items():
            row[c] = self._sense(c, self._oil_at(z, self.T_o, Ta) + (self.T_c - self.T_o))
        row["T_amb"] = self._sense("T_amb", Ta)
        row["T_wall"] = self._sense("T_wall", Ta + 0.8 * (self.T_o - Ta))       # ASSUMED wall fraction
        row["note"] = "THERMOSTAT_OPEN" if self._open else ""
        self._truth.append(dict(time_s=row["time_s"], h_true=self._h, T_oil_mean=self.T_o, T_coil=self.T_c,
                                P_true_W=P, hot_spot_true=self._oil_at(235, self.T_o, Ta) + (self.T_c - self.T_o)))
        return row

    def advance(self, seconds):
        """Simulate `seconds` of rig time; return one logger row per dt_sample."""
        self._pending += float(seconds)
        rows = []
        while self._pending >= self.dt_sample - 1e-9:
            pu = self.load_at(self.t)
            V = P = 0.0
            for _ in range(self.n_sub):
                V, P = self._substep(pu)
            rows.append(self._sample_row(pu, V, P))
            self._pending -= self.dt_sample
        return pd.DataFrame(rows, columns=COLUMNS)

    def truth_frame(self):
        """Hidden true quantities (a real rig never gives these). Used only to score the estimator."""
        return pd.DataFrame(list(self._truth))

def run_schedule(rig, hours=None):
    """Convenience: run a whole schedule (or `hours`) and return the logger DataFrame."""
    return rig.advance((hours * 3600.0) if hours is not None else rig.total)

# ----- logger CSV format -----
def write_csv(df, path):
    df.to_csv(path, index=False, columns=COLUMNS)

def read_csv(path_or_buffer):
    """Read a logger CSV. Requires the COLUMNS above; extra columns are kept; empty cells become NaN."""
    df = pd.read_csv(path_or_buffer)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df

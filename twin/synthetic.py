"""SYNTHETIC data for developing and testing the twin. Nothing here is a measurement.

Every generator returns a SyntheticRun whose `synthetic` flag is True and whose `truth` holds the hidden
quantities that a real transformer would never reveal (used only to score the filter). Sensor noise and the
drift in the film coefficient are INVENTED for the exercise.
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SyntheticRun:
    hours: np.ndarray
    load: np.ndarray
    measured_hot_spot: np.ndarray
    truth: dict = field(default_factory=dict)
    synthetic: bool = True
    note: str = "SYNTHETIC: invented load, drift and noise; not a measurement"

def load_profile(n_hours, seed=0, mean=0.75, daily=0.15, weekly=0.05, noise=0.03, lo=0.35, hi=1.3):
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    k = mean + daily * np.sin(2 * np.pi * (t % 24) / 24 - 2.0) + weekly * np.sin(2 * np.pi * t / (24 * 7)) \
        + noise * rng.standard_normal(n_hours)
    return np.clip(k, lo, hi)

def make_run(n_hours, hot_spot_fn, hf_true, seed=0, sigma=1.0, **load_kw):
    """hot_spot_fn(load, hf) -> LV hot-spot temperature; hf_true is an array (length n_hours) or a scalar."""
    rng = np.random.default_rng(seed + 1)
    load = load_profile(n_hours, seed, **load_kw)
    hf = np.broadcast_to(np.asarray(hf_true, float), load.shape)
    clean = np.array([hot_spot_fn(k, h) for k, h in zip(load, hf)])
    return SyntheticRun(np.arange(n_hours), load, clean + sigma * rng.standard_normal(n_hours),
                        truth=dict(hf=np.array(hf), hot_spot=clean, sigma=sigma))

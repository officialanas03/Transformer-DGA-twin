"""Calibration metrics and split-conformal prediction sets."""
import numpy as np

def expected_calibration_error(probs, y_idx, bins=10):
    """Top-label ECE: weighted gap between confidence and accuracy per confidence bin."""
    probs, y_idx = np.asarray(probs), np.asarray(y_idx)
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)

def conformal_threshold(probs_cal, y_cal_idx, alpha=0.1):
    """Split conformal with score = 1 - p(true class). Returns q such that the set
    {c : 1 - p_c <= q} covers the truth with probability >= 1 - alpha, provided the
    calibration and test points are exchangeable."""
    probs_cal, y_cal_idx = np.asarray(probs_cal), np.asarray(y_cal_idx)
    scores = 1 - probs_cal[np.arange(len(y_cal_idx)), y_cal_idx]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))

def prediction_sets(probs, q):
    """Boolean matrix (rows = samples, columns = classes)."""
    return (1 - np.asarray(probs)) <= q

def coverage_and_size(sets, y_idx):
    sets, y_idx = np.asarray(sets), np.asarray(y_idx)
    covered = sets[np.arange(len(y_idx)), y_idx]
    return {"coverage": float(covered.mean()), "mean_size": float(sets.sum(1).mean()),
            "empty_rate": float((sets.sum(1) == 0).mean())}

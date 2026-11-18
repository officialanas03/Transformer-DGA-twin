"""Extended Kalman filter for live calibration of a model parameter from a noisy sensor.

State x (default: log of the film coefficient) follows a random walk; the measurement is the model's hot-spot
temperature h(x, u) for input u (the load). The measurement Jacobian is taken numerically.

Limits: the filter makes the MODEL agree with the sensor. If the model structure is wrong (for example a different
oil temperature) it absorbs the error into the parameter, so the estimate can track the sensor well and still be
wrong as a physical quantity. This is a calibration, not a validation.
"""
import numpy as np

class EKF:
    def __init__(self, x0, P0, Q, R, h, eps=1e-4):
        self.x = np.atleast_1d(np.asarray(x0, float)).copy()
        self.P = np.atleast_2d(np.asarray(P0, float)).copy()
        self.Q = np.atleast_2d(np.asarray(Q, float))
        self.R = float(R)
        if self.R <= 0:
            raise ValueError("R must be positive")
        self.h, self.eps = h, eps

    def _jac(self, u):
        base = self.h(self.x, u)
        base = float(np.asarray(base).ravel()[0])
        J = np.empty((1, len(self.x)))
        for i in range(len(self.x)):
            dx = np.zeros_like(self.x); dx[i] = self.eps
            J[0, i] = (self.h(self.x + dx, u) - base) / self.eps
        return J, base

    def step(self, z, u):
        """Predict then update with measurement z (NaN or None = no measurement). Returns (innovation, predicted z)."""
        self.P = self.P + self.Q
        if z is None or not np.isfinite(z):
            return np.nan, np.nan
        J, zhat = self._jac(u)
        S = float((J @ self.P @ J.T).item()) + self.R
        K = (self.P @ J.T) / S
        innov = z - zhat
        self.x = self.x + (K * innov).ravel()
        self.P = (np.eye(len(self.x)) - K @ J) @ self.P
        return float(innov), float(zhat)

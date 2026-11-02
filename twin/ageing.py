"""Insulation ageing: IEC relative ageing rate and Ekenstam/Arrhenius kinetic DP.

Verified against retrieved sources:
 - Relative ageing rate, standard paper: V = 2^((theta_h - 98)/6)            (IEC 60076-7)
 - Thermally upgraded paper: V = exp(15000/(110+273) - 15000/(theta_h+273))  (IEC 60076-7 / IEEE C57.91)
 - Ekenstam: 1/DP_t - 1/DP_0 = k t, with Arrhenius k = A exp(-Ea/(R T)), R = 8.314 J/(mol K).
 - Ea is about 113 kJ/mol (Atlantis Press paper on cellulose ageing); Emsley and Stevens
   (Cellulose 1994) is the usual primary source. The pre-exponential factor A depends strongly
   on water, oxygen and acids, so NO default is given: calibrate it (see calibrate_A).
 - A DP of about 200 is commonly taken as end of life (IEEE Trans. Power Delivery 28(2), 2013).
The Ekenstam form assumes a constant rate constant at a given temperature; the Emsley model
(rate constant decaying with ageing) is not implemented.
"""
from dataclasses import dataclass
import numpy as np

R_GAS = 8.314
EA_DEFAULT = 113e3          # J/mol, approximate
DP_END_OF_LIFE = 200.0

def relative_ageing_rate(theta_h, upgraded=True):
    theta_h = np.asarray(theta_h, float)
    if upgraded:
        return np.exp(15000.0 / 383.0 - 15000.0 / (theta_h + 273.0))
    return 2.0 ** ((theta_h - 98.0) / 6.0)

def loss_of_life_hours(theta_h, dt_hours, upgraded=True):
    """Equivalent hours of normal-temperature ageing consumed: sum of V * dt."""
    return float(np.sum(relative_ageing_rate(theta_h, upgraded) * dt_hours))

def arrhenius_factor(theta_h, Ea=EA_DEFAULT):
    """exp(-Ea / (R T)) for hot-spot temperature in C."""
    return np.exp(-Ea / (R_GAS * (np.asarray(theta_h, float) + 273.15)))

@dataclass
class KineticAgeing:
    """Tracks E = integral of exp(-Ea/RT) dt [hours]; then 1/DP = 1/DP0 + A * E."""
    dp0: float = 1000.0
    A: float = None          # pre-exponential factor, in (1/DP) per hour; must be calibrated
    Ea: float = EA_DEFAULT
    E: float = 0.0           # accumulated Arrhenius exposure [hours]

    def advance(self, theta_h, dt_hours):
        self.E += float(np.sum(arrhenius_factor(theta_h, self.Ea) * dt_hours))

    def dp(self):
        if self.A is None:
            raise ValueError("A is not calibrated; call calibrate_A first")
        return 1.0 / (1.0 / self.dp0 + self.A * self.E)

    def calibrate_to_measurement(self, dp_measured):
        """Set A so the model reproduces a measured DP at the current exposure."""
        if self.E <= 0:
            raise ValueError("no exposure accumulated yet")
        if not 0 < dp_measured <= self.dp0:
            raise ValueError("measured DP must be positive and not above DP0")
        self.A = (1.0 / dp_measured - 1.0 / self.dp0) / self.E
        return self.A

def calibrate_A_from_life(theta_h_const, hours_to_eol, dp0=1000.0, dp_eol=DP_END_OF_LIFE, Ea=EA_DEFAULT):
    """A such that DP falls from dp0 to dp_eol after `hours_to_eol` at constant hot-spot temperature.
    This encodes an ASSUMPTION about life at a reference temperature, not a measurement."""
    return (1.0 / dp_eol - 1.0 / dp0) / (float(arrhenius_factor(theta_h_const, Ea)) * hours_to_eol)

def remaining_life_hours(model: KineticAgeing, exposure_rate_per_hour, dp_eol=DP_END_OF_LIFE):
    """Hours until DP reaches dp_eol if the Arrhenius exposure keeps accruing at the given rate."""
    if model.A is None:
        raise ValueError("A is not calibrated")
    need = (1.0 / dp_eol - 1.0 / model.dp0) / model.A      # total exposure at end of life
    left = max(0.0, need - model.E)
    return left / exposure_rate_per_hour if exposure_rate_per_hour > 0 else np.inf

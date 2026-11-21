# Test rig: design values, logger format, and the virtual rig

Status: the rig is a PROPOSED design. Nothing has been built or measured. The virtual rig (`twin/rig.py`) is a simulation
of that design and every number it produces is SYNTHETIC.

## Design values (proposed, adjust to the parts you have)

| Item | Value |
|---|---|
| Tank | inside diameter 200 mm, height 400 mm, oil filled to z = 320 mm (about 9 to 10 L) |
| Coil | tube OD 60 x 250 mm (z = 30 to 280), 39 turns of 1.0 mm nichrome over 200 mm (z = 55 to 255), 10 layers of Kraft paper |
| Heater | about 10.5 ohm, 40 W at 1.0 pu (20.5 V, 1.95 A), 67.6 W at 1.3 pu (26.6 V, 2.54 A); power = 40 W x load^2 |
| Oil probes T1..T6 | DS18B20 at z = 55, 105, 155, 205, 255, 305 mm, 70 mm from the axis |
| Hot-spot thermocouples | H3, H2, H1 (K-type, under the first paper layer) at z = 75, 155, 235 mm (10, 50, 90 % of the winding) |
| Safety | 100 C bimetal thermostat in series with the heater, 4 A slow-blow fuse |

The nichrome resistivity (1.10 uOhm m), tube bore and paper thickness are assumptions to check. Measure the cold resistance
and recompute the voltages from your value. Full drawings, parts and equipment are on the rig page that accompanies this repository.

## Logger CSV format (what `twin.rig.read_csv` and `twin.live.CsvTailSource` expect)

One header line, then one line per sample. Empty cell = missing reading (probe dropout). Flush after every line.

| Column | Unit | Meaning |
|---|---|---|
| `time_s` | s | time since the run started |
| `setpoint_pu` | pu | commanded load (for the record) |
| `P_W`, `V`, `I` | W, V, A | MEASURED power, volts, amps (the twin uses `P_W`, not the setpoint) |
| `T_oil_1` .. `T_oil_6` | C | oil probes, bottom to top |
| `T_paper_top`, `T_paper_mid`, `T_paper_bot` | C | H1, H2, H3 |
| `T_amb`, `T_wall` | C | room and outside tank wall |
| `note` | text | free text; the virtual rig writes `THERMOSTAT_OPEN` |

The ESP32 firmware and the wiring are NOT written or tested. The format above is what it has to produce.

## What the live twin does

`twin/live.py`: probe offsets from calibration are subtracted; the measured oil temperature at the H1 height is interpolated
from the probes; an extended Kalman filter updates the film coefficient h in
`T_H1 = T_oil(z_H1) + (P / A) (1/h + R_paper)`. The filter is fitted to H1 only. H2 and H3 are held out and their residuals
are reported. Updates are skipped while the power changes by more than 3 % within 10 minutes (status: "settling").
DP comes from the Luo-style chain (rate constant fitted to Table 2 row I, moisture ignored) at the measured H1 temperature.

Limits: `R_paper` and the coil area are design values. If they are wrong, the error moves into h and the fitted sensor still
agrees (tests show h off by about 49 % with R_paper doubled while the fitted-sensor residual stays near 0.1 K). Probe offsets
bias h by a few percent unless calibrated. The statistical sd printed with h does not include either.

## Virtual rig assumptions (all ASSUMED, none measured)

Two lumped nodes (coil, oil), oil volume 9.25 L, rho 870 kg/m3, cp 1900 J/kgK, tank-to-room loss UA = 2 W/K, coil heat capacity
350 J/K, paper 1 mm at k = 0.2, film coefficient h = 70 (dT_film/20)^0.25 W/m2K, linear oil stratification with
top-minus-bottom = 0.4 (T_oil - T_amb), a coil that is 2 % above its design resistance, probe offsets (0.25 C oil, 0.5 C thermocouples),
noise (0.10 and 0.15 C), quantisation (0.0625 and 0.0078125 C) and 0.2 % random dropouts.

Consequences worth knowing before building (they follow from the assumed values, not from measurements):
* The oil time constant is C/UA, about 2.1 h, so each load level of a steady-state sweep needs about 6 h.
* At 1.0 pu the simulated hot spot settles near 67 C. Holding 1.3 pu for hours would reach about 93 C; keep the overload test short.

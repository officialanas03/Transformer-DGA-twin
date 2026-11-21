"""Run the SYNTHETIC virtual rig through the live estimator and print what happened.

    python run_virtual_rig.py                      # load_steps schedule, table only
    python run_virtual_rig.py --schedule sweep --out rig_virtual.csv

Everything is simulated (heated coil in oil, probe offsets, noise, dropouts). The CSV has the same columns as the
real logger will write, so `twin.live.CsvTailSource` can follow either. Nothing here validates any physics."""
import argparse
import numpy as np
from twin.rig import VirtualRig, SCHEDULES, write_csv
from twin.live import LiveTwin

ap = argparse.ArgumentParser()
ap.add_argument("--schedule", choices=SCHEDULES, default="load_steps")
ap.add_argument("--hours", type=float, default=None, help="default: the whole schedule")
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--calibrated", action="store_true", help="give the twin the true probe offsets (as step B0 would)")
ap.add_argument("--out", default=None, help="write the logger-format CSV here")
a = ap.parse_args()

rig = VirtualRig(schedule=a.schedule, loop=False, seed=a.seed)
df = rig.advance((a.hours * 3600.0) if a.hours else rig.total)
tw = LiveTwin(offsets=rig.offsets if a.calibrated else None, years=10)
tw.ingest_frame(df)
h, tr = tw.history_frame(), rig.truth_frame()
print("SYNTHETIC run:", rig.note)
print(f"schedule {a.schedule}: {len(df)} rows, {tw.n_updates} filter updates, status counts {h.status.value_counts().to_dict()}")
steady = h[h.status == "tracking"]
print(f"film coefficient: estimate {tw.hf:.1f} +/- {tw.hf_sd:.1f} (statistical), hidden true value {tr.h_true.iloc[-1]:.1f}")
print(f"held-out thermocouples H2, H3: RMS {steady.held_out_rms.dropna().mean():.2f} K while tracking")
print(f"max true hot spot {tr.hot_spot_true.max():.1f} C; thermostat rows: {(df.note == 'THERMOSTAT_OPEN').sum()}")
if a.out:
    write_csv(df, a.out)
    print("wrote", a.out)

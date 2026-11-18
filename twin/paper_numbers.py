"""Loader for reference/luo2023_numbers.csv: numbers printed in Luo et al., Measurement 222 (2023) 113714.

Each row has a PDF page, a location (equation / figure / table), and a `kind`:
  printed                 printed as text, equation, legend or label
  figure_reading_approx   read by eye from a plot: APPROXIMATE
  assumption              not in the paper; chosen here and flagged
  not_given / not_extracted   absent from the paper, or left out on purpose
The file is used by tests to check that constants in the code were not mistyped.
"""
import csv
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "reference" / "luo2023_numbers.csv"
KINDS = {"printed", "figure_reading_approx", "assumption", "not_given", "not_extracted"}

def load(path=_PATH):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["kind"] not in KINDS:
            raise ValueError(f"row {r['id']}: unknown kind {r['kind']!r}")
    return rows

def value(row_id, rows=None):
    """Numeric value of a row. Raises if the row has no number (not_given etc.)."""
    rows = rows or load()
    for r in rows:
        if r["id"] == row_id:
            if r["value"] == "":
                raise ValueError(f"{row_id} ({r['quantity']}) has no value: {r['kind']}")
            return float(r["value"])
    raise KeyError(row_id)

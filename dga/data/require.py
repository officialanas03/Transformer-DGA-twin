from pathlib import Path
import sys

def require_data(*names):
    """Exit with a clear message if a data file the script needs is missing (it is not shipped in the repo)."""
    root = Path(__file__).resolve().parents[2] / "data"
    missing = [n for n in names if not (root / n).exists()]
    if missing:
        sys.exit("Missing data file(s) in data/: " + ", ".join(missing) + "\nThey are not distributed with this repository; see DATA.md.")

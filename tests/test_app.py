from pathlib import Path
import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"

def test_app_runs_without_exceptions():
    """Smoke test of the Streamlit script (skipped where Streamlit is not installed)."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) >= 5

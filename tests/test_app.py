import pytest

def test_app_runs_without_exceptions():
    """Smoke test of the Streamlit script (skipped where Streamlit is not installed). NOT run in the authoring environment."""
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) >= 4

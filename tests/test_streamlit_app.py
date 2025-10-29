"""Tests for the Streamlit dashboard entry point."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP_PATH = APP_ROOT / "app" / "streamlit_app.py"


def test_streamlit_app_renders_without_exceptions() -> None:
    """Ensure the Streamlit app can be executed without runtime errors."""

    app_test = AppTest.from_file(str(STREAMLIT_APP_PATH))
    app_test.run(timeout=20)

    assert not app_test.exception, f"Streamlit app raised {app_test.exception!r}"
    assert len(app_test.main) > 0, "Streamlit app did not render any elements."

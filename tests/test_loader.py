import pandas as pd
import pytest
from pfd_toolkit.loader import load_reports


def test_load_reports_date_range():
    df = load_reports(start_date="2024-12-01", end_date="2024-12-31", n_reports=5)
    assert not df.empty
    assert (df['date'] >= pd.Timestamp("2024-12-01")).all()
    assert (df['date'] <= pd.Timestamp("2024-12-31")).all()
    assert len(df) <= 5


def test_load_reports_invalid_range():
    with pytest.raises(ValueError):
        load_reports(start_date="2024-12-31", end_date="2024-01-01")


def test_load_reports_truncate():
    df = load_reports(n_reports=3)
    assert len(df) <= 3

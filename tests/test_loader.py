import pandas as pd
from pfd_toolkit.loader import load_reports


def test_load_reports_date_range():
    df = load_reports(start_date="2024-12-01", end_date="2024-12-31", n_reports=5)
    assert not df.empty
    assert (df['Date'] >= pd.Timestamp("2024-12-01")).all()
    assert (df['Date'] <= pd.Timestamp("2024-12-31")).all()
    assert len(df) <= 5

from pathlib import Path

import pandas as pd
import pytest
from pfd_toolkit import loader


@pytest.fixture(autouse=True)
def use_local_dataset(tmp_path, monkeypatch):
    data_path = Path(__file__).resolve().parents[1] / "src" / "pfd_toolkit" / "data" / "all_reports.csv"
    cache_file = tmp_path / "all_reports.csv"

    def fake_ensure_dataset(force_download: bool = False) -> Path:
        if force_download and cache_file.exists():
            cache_file.unlink()
        if not cache_file.exists():
            cache_file.write_bytes(data_path.read_bytes())
        return cache_file

    monkeypatch.setattr(loader, "_ensure_dataset", fake_ensure_dataset)
    yield cache_file


def test_load_reports_date_range():
    df = loader.load_reports(start_date="2024-12-01", end_date="2024-12-31", n_reports=5)
    assert not df.empty
    assert (df['date'] >= pd.Timestamp("2024-12-01")).all()
    assert (df['date'] <= pd.Timestamp("2024-12-31")).all()
    assert len(df) <= 5


def test_load_reports_invalid_range():
    with pytest.raises(ValueError):
        loader.load_reports(start_date="2024-12-31", end_date="2024-01-01")


def test_load_reports_truncate():
    df = loader.load_reports(n_reports=3)
    assert len(df) <= 3


def test_load_reports_clear_cache(use_local_dataset):
    fake_cache = use_local_dataset
    fake_cache.write_text("dummy")

    df = loader.load_reports(n_reports=1, clear_cache=True)
    assert not df.empty
    assert fake_cache.exists()
    assert fake_cache.stat().st_size > len("dummy")

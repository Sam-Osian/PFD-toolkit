from pathlib import Path

import pandas as pd
import pytest
from pfd_toolkit import loader


@pytest.fixture(autouse=True)
def use_local_dataset(tmp_path, monkeypatch):
    """Provide a tiny CSV dataset to avoid network calls during tests."""

    loader._reset_dataframe_cache()
    cache_file = tmp_path / "all_reports.csv"

    # Small dataset covering a range of dates
    df = pd.DataFrame(
        {
            "url": ["u1", "u2", "u3", "u4"],
            "id": ["2024-0001", "2024-0002", "2024-0003", "2025-0001"],
            "date": [
                "2024-12-01",
                "2024-12-15",
                "2024-12-30",
                "2025-01-01",
            ],
            "theme_patient_safety": [True, False, True, False],
            "theme_mental_health": [False, True, True, False],
        }
    )

    def fake_ensure_dataset(force_download: bool = False) -> Path:
        if force_download and cache_file.exists():
            cache_file.unlink()
        if not cache_file.exists():
            df.to_csv(cache_file, index=False)
        return cache_file

    monkeypatch.setattr(loader, "_ensure_dataset", fake_ensure_dataset)
    yield cache_file
    loader._reset_dataframe_cache()


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

    df = loader.load_reports(n_reports=1)
    assert not df.empty
    assert fake_cache.exists()
    assert fake_cache.stat().st_size > len("dummy")


def test_load_reports_reuses_process_cache(monkeypatch):
    loader._reset_dataframe_cache()
    original_read_csv = loader.pd.read_csv
    call_count = {"value": 0}

    def counting_read_csv(*args, **kwargs):
        call_count["value"] += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(loader.pd, "read_csv", counting_read_csv)

    first = loader.load_reports(n_reports=2, refresh=False)
    second = loader.load_reports(n_reports=3, refresh=False)

    assert len(first) == 2
    assert len(second) == 3
    assert call_count["value"] == 1


def test_load_reports_filters_single_theme():
    df = loader.load_reports(theme="patient_safety", refresh=False)
    assert set(df["url"]) == {"u1", "u3"}


def test_load_reports_filters_multiple_themes_with_or_semantics():
    df = loader.load_reports(
        theme=["patient_safety", "mental_health"],
        refresh=False,
    )
    assert set(df["url"]) == {"u1", "u2", "u3"}


def test_load_reports_invalid_theme_raises():
    with pytest.raises(ValueError, match="Unknown theme"):
        loader.load_reports(theme="not_a_real_theme", refresh=False)

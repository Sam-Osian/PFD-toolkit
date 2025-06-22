from __future__ import annotations

import importlib.resources as resources
from typing import Final

from pathlib import Path
import requests

import pandas as pd
from dateutil import parser as _date_parser

# Path to the embedded CSV
_DATA_PACKAGE: Final[str] = "pfd_toolkit.data"
_DATA_FILE: Final[str] = "all_reports.csv"

# Location of the latest dataset release
_DATA_URL: Final[str] = (
    "https://github.com/Sam-Osian/PFD-toolkit/releases/download/"
    "dataset-latest/all_reports.csv"
)

# Cache path for the downloaded dataset
_CACHE_DIR: Final[Path] = Path.home() / ".cache" / "pfd_toolkit"
_CACHE_FILE: Final[Path] = _CACHE_DIR / _DATA_FILE


def _ensure_dataset() -> Path:
    """Download the dataset if not already cached and return its path."""
    if not _CACHE_FILE.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(_DATA_URL, timeout=30)
            resp.raise_for_status()
            _CACHE_FILE.write_bytes(resp.content)
        except Exception as exc:  # pragma: no cover - network failure
            raise FileNotFoundError(
                "Failed to download dataset from GitHub release"
            ) from exc
    return _CACHE_FILE


def load_reports(
    start_date: str = "2000-01-01",
    end_date: str = "2050-01-01",
    n_reports: int | None = None,
) -> pd.DataFrame:
    """Load the bundled Prevention of Future Death reports as a DataFrame.

    Parameters
    ----------
    start_date : str, optional
        Inclusive lower bound for the report date in ``YYYY-MM-DD`` format.
        Defaults to ``"2000-01-01"``.
    end_date : str, optional
        Inclusive upper bound for the report date in ``YYYY-MM-DD`` format.
        Defaults to ``"2050-01-01"``.
    n_reports : int or None, optional
        Keep only the most recent ``n_reports`` rows after filtering by date.
        ``None`` (the default) returns all rows.

    Returns
    -------
    pandas.DataFrame
        Reports filtered by date, sorted newest first and optionally
        limited to ``n_reports`` rows.

    Raises
    ------
    ValueError
        If *start_date* is after *end_date*.
    FileNotFoundError
        If the bundled CSV cannot be located.

    Examples
    --------
        from pfd_toolkit import load_reports
        df = load_reports(start_date="2020-01-01", end_date="2022-12-31", n_reports=100)
        df.head()
    """
    
    # Date param reading
    date_from = _date_parser.parse(start_date)
    date_to = _date_parser.parse(end_date)
    if date_from > date_to:
        raise ValueError("start_date must be earlier than or equal to end_date")

    # Read CSV from release cache, falling back to bundled data
    try:
        csv_path = _ensure_dataset()
        reports = pd.read_csv(csv_path)
    except Exception:
        csv_path = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE)
        with csv_path.open("r", encoding="utf-8") as fh:
            reports = pd.read_csv(fh)

    # Drop any Unnamed columns
    reports = reports.loc[:, ~reports.columns.str.startswith("Unnamed")]

    # Cleaning
    # ...Parse the Date column, drop rows with invalid dates, and filter window
    reports["date"] = pd.to_datetime(
        reports["date"], format="%Y-%m-%d", errors="coerce"
    )
    reports = (
        reports.dropna(subset=["date"])
        .loc[lambda df: (df["date"] >= date_from) & (df["date"] <= date_to)]
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )

    # Limit to n_reports
    if n_reports is not None:
        # .head(n) will return all rows if n > len(reports)
        reports = reports.head(n_reports).reset_index(drop=True)

    return reports

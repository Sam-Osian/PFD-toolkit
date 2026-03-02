from __future__ import annotations

from typing import Final

from pathlib import Path
from threading import Lock
import requests

import pandas as pd
from dateutil import parser as _date_parser

# Path within the package used for tests
_DATA_FILE: Final[str] = "all_reports.csv"

# Location of the latest dataset release
_DATA_URL: Final[str] = (
    "https://github.com/Sam-Osian/PFD-toolkit/releases/download/"
    "dataset-latest/all_reports.csv"
)

# Cache path for the downloaded dataset
_CACHE_DIR: Final[Path] = Path.home() / ".cache" / "pfd_toolkit"
_CACHE_FILE: Final[Path] = _CACHE_DIR / _DATA_FILE
_DATAFRAME_CACHE_LOCK: Final[Lock] = Lock()
_DATAFRAME_CACHE_SIGNATURE: tuple[str, int, int] | None = None
_DATAFRAME_CACHE_FRAME: pd.DataFrame | None = None


def _ensure_dataset(force_download: bool = False) -> Path:
    """Download the dataset if not already cached and return its path.

    Parameters
    ----------
    force_download : bool, optional
        If ``True``, delete any cached file before downloading a fresh copy.
        Defaults to ``False``.
    """
    if force_download and _CACHE_FILE.exists():
        _CACHE_FILE.unlink()

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


def _reset_dataframe_cache() -> None:
    global _DATAFRAME_CACHE_SIGNATURE, _DATAFRAME_CACHE_FRAME
    with _DATAFRAME_CACHE_LOCK:
        _DATAFRAME_CACHE_SIGNATURE = None
        _DATAFRAME_CACHE_FRAME = None


def _load_base_reports(force_download: bool = False) -> pd.DataFrame:
    global _DATAFRAME_CACHE_SIGNATURE, _DATAFRAME_CACHE_FRAME

    csv_path = _ensure_dataset(force_download=force_download)
    stat = csv_path.stat()
    signature = (str(csv_path.resolve()), stat.st_mtime_ns, stat.st_size)

    with _DATAFRAME_CACHE_LOCK:
        if _DATAFRAME_CACHE_SIGNATURE != signature or _DATAFRAME_CACHE_FRAME is None:
            reports = pd.read_csv(csv_path)
            reports = reports.loc[:, ~reports.columns.str.startswith("Unnamed")]
            reports["date"] = pd.to_datetime(
                reports["date"], format="%Y-%m-%d", errors="coerce"
            )
            reports = (
                reports.dropna(subset=["date"])
                .sort_values("date", ascending=False)
                .reset_index(drop=True)
            )
            _DATAFRAME_CACHE_SIGNATURE = signature
            _DATAFRAME_CACHE_FRAME = reports

        return _DATAFRAME_CACHE_FRAME


def load_reports(
    start_date: str = "2000-01-01",
    end_date: str = "2050-01-01",
    n_reports: int | None = None,
    refresh: bool = True,
) -> pd.DataFrame:
    """Load Prevention of Future Death reports as a DataFrame.

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
    refresh : bool, optional
        If ``True`` (the default), force a fresh download of the dataset. Set to
        ``False`` to reuse the previously cached copy.

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
        If the dataset cannot be downloaded.

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

    reports = _load_base_reports(force_download=refresh)
    reports = reports.loc[
        (reports["date"] >= date_from) & (reports["date"] <= date_to)
    ].copy()
    reports.reset_index(drop=True, inplace=True)

    # Limit to n_reports
    if n_reports is not None:
        # .head(n) will return all rows if n > len(reports)
        reports = reports.head(n_reports).reset_index(drop=True)

    return reports

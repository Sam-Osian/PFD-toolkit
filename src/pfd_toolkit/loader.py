from __future__ import annotations

from typing import Final

from pathlib import Path
from threading import Lock
import requests

import pandas as pd
from dateutil import parser as _date_parser
from pfd_toolkit.collections import COLLECTION_COLUMNS as _COLLECTION_COLUMNS

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
_THEME_PREFIX: Final[str] = "theme_"
_THEME_COLLECTION_NAME_OVERRIDES: Final[dict[str, str]] = {
    "suicide_risk": "suicide",
    "care_home_safety": "care_home",
}
_COLLECTION_ALIASES: Final[dict[str, str]] = {
    "health_reg": "health_regulators",
    "suicide_risk": "suicide",
    "care_home_safety": "care_home",
}


def _normalise_theme_name(raw_theme: str) -> str:
    value = str(raw_theme or "").strip()
    if value.startswith(_THEME_PREFIX):
        return value[len(_THEME_PREFIX):]
    return value


def _normalise_collection_name(raw_value: str) -> str:
    value = _normalise_theme_name(raw_value).strip()
    return _COLLECTION_ALIASES.get(value, value)

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


def _get_thematic_collection_columns(reports: pd.DataFrame) -> dict[str, str]:
    """Return available thematic collections keyed by collection name."""
    theme_columns: dict[str, str] = {}
    collection_columns = set(_COLLECTION_COLUMNS.values())
    for column in reports.columns:
        if column.startswith(_THEME_PREFIX) and column not in collection_columns:
            theme_name = column[len(_THEME_PREFIX):]
            if theme_name:
                public_name = _THEME_COLLECTION_NAME_OVERRIDES.get(theme_name, theme_name)
                theme_columns[public_name] = column
    return theme_columns


def load_reports(
    start_date: str = "2000-01-01",
    end_date: str = "2050-01-01",
    n_reports: int | None = None,
    refresh: bool = True,
    collection: str | list[str] | None = None,
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
    collection : str | list[str] | None, optional
        Filter rows using packaged collections. This includes receiver-based
        collections (e.g. ``"nhs"``, ``"gov_department"``,
        ``"health_regulators"``) and thematic collections inferred from
        packaged ``theme_*`` columns (e.g. ``"medication_safety"``,
        ``"suicide"``, ``"care_home"``). Values may be supplied with or
        without the ``theme_`` prefix for thematic collections. Pass a single
        collection name or a list of collection names. When multiple
        collections are supplied, rows matching **any** requested collection are
        returned. When a single collection is supplied, its boolean helper
        column is dropped from the returned DataFrame. When two or more
        collections are supplied, the requested boolean columns are retained for
        comparison.

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
    ValueError
        If a requested collection is not present in the packaged dataset.

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

    if collection is not None:
        requested_collections_raw = (
            [collection] if isinstance(collection, str) else list(collection)
        )
        requested_collections = []
        for item in requested_collections_raw:
            cleaned = _normalise_collection_name(str(item or "").strip())
            if cleaned:
                requested_collections.append(cleaned)
        if not requested_collections:
            raise ValueError(
                "collection must contain at least one non-empty collection name."
            )

        thematic_collections = _get_thematic_collection_columns(reports)
        available_collections = {**_COLLECTION_COLUMNS, **thematic_collections}
        missing_collections = [
            item for item in requested_collections if item not in available_collections
        ]
        if missing_collections:
            available = ", ".join(sorted(available_collections)) or "none"
            raise ValueError(
                f"Unknown collection(s): {missing_collections}. Available collections are: {available}."
            )

        collection_columns = [available_collections[item] for item in requested_collections]
        missing_columns = [
            column for column in collection_columns if column not in reports.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Requested collection column(s) are not present in the packaged dataset: {missing_columns}."
            )

        collection_mask = (
            reports[collection_columns].fillna(False).astype(bool).any(axis=1)
        )
        reports = reports.loc[collection_mask].reset_index(drop=True)
        if len(requested_collections) == 1:
            reports = reports.drop(columns=collection_columns, errors="ignore")

    # Limit to n_reports
    if n_reports is not None:
        # .head(n) will return all rows if n > len(reports)
        reports = reports.head(n_reports).reset_index(drop=True)

    return reports

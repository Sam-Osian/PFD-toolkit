from __future__ import annotations

import importlib.resources as resources
from typing import Final

import pandas as pd
from dateutil import parser as _date_parser

# Path to the embedded CSV
_DATA_PACKAGE: Final[str] = "pfd_toolkit.data"
_DATA_FILE: Final[str] = "all_reports.csv"


def load_reports(
    category: str = "all",
    start_date: str = "2000-01-01",
    end_date: str = "2050-01-01",
    n_reports: int | None = None,
) -> pd.DataFrame:
    """Utility for loading the fully-cleaned **Prevention of Future Death**
    report dataset shipped with *pfd_toolkit* as a :class:`pandas.DataFrame`.

    Parameters
    ----------
    category : str, optional
        PFD category slug (e.g. ``"suicide"``) or ``"all"``.
    start_date : str, optional
        Inclusive lower bound for the **report date** in the ``YYYY-MM-DD``
        format.
    end_date : str, optional
        Inclusive upper bound for the **report date** in the ``YYYY-MM-DD``
        format.
    n_reports : int or None, optional
        If given, keep only the most recent *n_reports* (based on the “Date” column)
        after filtering by date. If `None` (the default), return all reports
        in the specified date range.

    Returns
    -------
    pandas.DataFrame
        Reports filtered by date, sorted newest-first, and (optionally) truncated
        to the first *n_reports* rows.

    Raises
    ------
    ValueError
        If *start_date* is after *end_date*.
    FileNotFoundError
        If the bundled CSV cannot be located (i.e. a package-level error).

    Examples
    --------
    >>> from pfd_toolkit import load_reports
    >>> df = load_reports(start_date="2020-01-01", end_date="2022-12-31", n_reports=1000)
    >>> df.head()
    """
    # Date param reading
    date_from = _date_parser.parse(start_date)
    date_to = _date_parser.parse(end_date)
    if date_from > date_to:
        raise ValueError("start_date must be earlier than or equal to end_date")

    # Read CSV
    csv_path = resources.files(_DATA_PACKAGE).joinpath(_DATA_FILE)
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            reports = pd.read_csv(fh)
            # Drop any Unnamed columns
            reports = reports.loc[:, ~reports.columns.str.startswith("Unnamed")]

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Bundled dataset {_DATA_FILE!r} not found in package " f"{_DATA_PACKAGE!r}"
        ) from exc

    # Cleaning
    # ...Parse the Date column, drop rows with invalid dates, and filter window
    reports["Date"] = pd.to_datetime(
        reports["Date"], format="%Y-%m-%d", errors="coerce"
    )
    reports = (
        reports.dropna(subset=["Date"])
        .loc[lambda df: (df["Date"] >= date_from) & (df["Date"] <= date_to)]
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )

    # Limit to n_reports
    if n_reports is not None:
        # .head(n) will return all rows if n > len(reports)
        reports = reports.head(n_reports).reset_index(drop=True)

    # Category filtering placeholder
    # _ = category.lower()

    return reports

# Loading report data

`load_reports()` is the quickest way to access PFD reports.  It loads a
clean CSV that ships with the package and returns a pandas
`DataFrame`. Each row represents a single report with columns mirroring
the main sections.

```python
from pfd_toolkit import load_reports

reports = load_reports(
    start_date="2024-01-01",
    end_date="2024-12-31",
    n_reports=None,
)
```

Pass a `start_date` and `end_date` to restrict the date range, and use
`n_reports` to trim the DataFrame to the most recent *n* entries.
Results are always sorted newest first.

The dataset is refreshed weekly.  Simply run
`pip install --upgrade pfd_toolkit` whenever a new snapshot is
published.

See the [API reference](../reference/loader.md) for a breakdown of the
output columns.

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

Pass a `start_date` and `end_date` to restrict the date range, and optionally use
`n_reports` to trim the DataFrame to the most recent *n* entries.
Results are always sorted newest first.

The dataset is refreshed weekly.  Simply run
`pip install --upgrade pfd_toolkit` whenever a new snapshot is
published.


## Caveats

To collect PFD reports, we run a scraping pipeline on the judiciary.uk website every week. 
Our scraping methods assume that the host website will not change its basic layout. Should 
the host change their website structure, our pipeline may fail to update its catelogue of 
reports. The existing catelogue of reports will be unaffected.

Should this happen, we will notify users at the top of the [Home page](../index.md) and provide
updates on when we can remedy the issue.
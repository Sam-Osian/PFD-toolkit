## Get reports

`load_reports()` is the quickest way to access PFD reports.  It loads a clean CSV and returns a pandas
`DataFrame`. Each row represents a single report, with columns reflect the main sections of the report.

```py
from pfd_toolkit import load_reports

# Load all PFD reports from January 2024 to May 2025
reports = load_reports(
    start_date="2024-01-01",
    end_date="2025-05-01")

reports.head()
```


| url                        | date       | coroner    | area                        | receiver                | investigation           | circumstances                 | concerns                   |
|----------------------------|------------|------------|-----------------------------|-------------------------|-------------------------|-------------------------------|----------------------------|
| [...]            | 2025-05-01 | A. Hodson  | Birmingham and...    | NHS England; The Rob... | On 9th December 2024... | At 10.45am on 23rd November...| To The Robert Jones... |
| [...]           | 2025-04-30 | J. Andrews | West Sussex, Br...| West Sussex C... | On 2 November 2024 I... | They drove their car into...   | The inquest was told t...  |
| [...]            | 2025-04-30 | A. Mutch   | Manchester Sou...            | Fluxton Road Medical... | On 1 October 2024 I...  | They were prescribed long...   | The inquest heard evide... |
| [...]            | 2025-04-25 | J. Heath   | North Yorkshire...   | Townhead Surgery        | On 4th June 2024 I...   | On 15 March 2024, Richar...    | When a referral docume...  |
| [...]            | 2025-04-25 | M. Hassell | Inner North Lo...          | The President Royal...  | On 23 August 2024, on...| They were a big baby and...    | With the benefit of a m... |

Pass a `start_date` and `end_date` to restrict the date range, and optionally use
`n_reports` to trim the DataFrame to the most recent *n* entries. For example...

```py
reports = load_reports(
    n_reports=1000)
```

...loads the 1000 latest reports.


---

## Refresh reports

Reports are updated once a week (Monday 1:00am, universal time). `load_reports()` caches reports for faster loading, so to retrieve the latest reports you'll need to set `clear_cache` to `True`:

```py
reports = load_reports(clear_cache=True)
```



!!! note
    The dataset loaded when you call `load_reports()` is cleaned and fully processed. This means spelling and grammatical errors have been corrected and boilerplate text removed.
    
    If you wish to load an uncleaned version of the dataset, we suggest running your own scrape via [`Scraper`](scraper.md).

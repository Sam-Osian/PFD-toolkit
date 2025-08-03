---
title: "Load live report data"
description: |
  Fetch PFD reports as a pandas DataFrame using load_reports().
---

## Load live report data

`load_reports()` is the quickest way to access PFD reports.  While before, researchers would have to manually download reports one-by-one, this function allows users to immediately access all reports.

The function returns a pandas `DataFrame`, with each row representing an individual report and the columns representing the main report sections.

```py
from pfd_toolkit import load_reports

# Load all PFD reports from January 2024 to May 2025
reports = load_reports(
    start_date="2024-01-01", # YYYY-MM-DD format
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

If you don't pass `start_date` or `end_date` parameters, `load_reports()` will pull the entire collection of PFD reports.

!!! note
    Please note that the date ranges denote when the report was published, not the date of death.

---

## Get *n* latest reports

Optionally, use `n_reports` to trim the DataFrame to the most recent *n* entries. For example...

```py
reports = load_reports(
    n_reports=1000)
```

...loads the 1000 latest reports.

You can combine this with the date parameters to get the most recent *n* entries within a given date range.


---

## Refresh reports

Reports are updated daily (1:00am, universal time) and `load_reports()` fetches
the newest dataset by default. If you want to reuse the previously cached copy
to avoid a download on subsequent calls, set `refresh` to `False`:

```py
reports = load_reports(refresh=False)
```


!!! note
    The dataset loaded when you call `load_reports()` is cleaned and fully processed. This means spelling and grammatical errors have been corrected and boilerplate text removed.
    
    If you wish to load an uncleaned version of the dataset, we suggest running your own scrape via [`Scraper`](scraper.md).
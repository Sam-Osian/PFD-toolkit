# Loading report data

`load_reports()` is the quickest way to access PFD reports.  It loads a
clean CSV that ships with the package and returns a pandas
`DataFrame`. Each row represents a single report with columns mirroring
the main sections.

```py
from pfd_toolkit import load_reports

# Load all PFD reports from January 2024 to May 2025
reports = load_reports(
    start_date="2024-01-01",
    end_date="2025-05-01")

reports.head(n=5)
```


| url                        | date       | coroner    | area                        | receiver                | investigation           | circumstances                 | concerns                   |
|----------------------------|------------|------------|-----------------------------|-------------------------|-------------------------|-------------------------------|----------------------------|
| [...]            | 2025-05-01 | A. Hodson  | Birmingham and...    | NHS England; The Rob... | On 9th December 2024... | At 10.45am on 23rd November...| To The Robert Jones... |
| [...]           | 2025-04-30 | J. Andrews | West Sussex, Br...| West Sussex C... | On 2 November 2024 I... | They drove their car into...   | The inquest was told t...  |
| [...]            | 2025-04-30 | A. Mutch   | Manchester Sou...            | Fluxton Road Medical... | On 1 October 2024 I...  | They were prescribed long...   | The inquest heard evide... |
| [...]            | 2025-04-25 | J. Heath   | North Yorkshire...   | Townhead Surgery        | On 4th June 2024 I...   | On 15 March 2024, Richar...    | When a referral docume...  |
| [...]            | 2025-04-25 | M. Hassell | Inner North Lo...          | The President Royal...  | On 23 August 2024, on...| They were a big baby and...    | With the benefit of a m... |


Pass a `start_date` and `end_date` to restrict the date range, and optionally use
`n_reports` to trim the DataFrame to the most recent *n* entries.
Results are always sorted newest first.

The dataset is refreshed weekly.  Simply run
`pip install --upgrade pfd_toolkit` whenever a new snapshot is
published.


!!! note
    The dataset loaded when you call `load_reports()` is cleaned and fully processed. If you wish to load an uncleaned version of the dataset, we suggest running your own scrape via [`Scraper`](scraper.md).


## Caveats

To collect PFD reports, we run a scraping pipeline on the judiciary.uk website every week. 
Our scraping methods assume that the host website will not change its basic layout. Should 
the host change their website structure, our pipeline may fail to update its catelogue of 
reports. The existing catelogue of reports will be unaffected.

Should this happen, we will notify users at the top of the [Home page](../index.md) and provide
updates on when we can remedy the issue.
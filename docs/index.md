![PFD Toolkit](assets/header.png)

A Python package for scraping and analysing UK Prevention of Future Death (PFD) reports.

*Developed by Samuel Osian and John Pytches*

```bash
pip install PFDtoolkit
```

## Features
- Easily load live PFD report data from the Judiciary website.

- Clean and structure messy data.

- Designed for reproducible, programmatic access to public data.


## Quick start

```py
from PFDtoolkit import Data

# Load PFD data
loader = Data(date_from='2015-01-01', # (YYYY-MM-DD)
                date_to='2025-01-01')

reports = loader.get_data()
```
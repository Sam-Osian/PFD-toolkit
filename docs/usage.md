# Getting started

## Installation

PFD Toolkit can be installed from pip as `pfd_toolkit`:

```bath
pip install pfd_toolkit
```

*Note: The package is not currently on Conda distributions.*

## Loading your first dataset

Once installed, you can quickly load a cleaned, up-to-date dataset of Prevention of Future Death (PFD) reports using just a few lines of code:

```py
from pfd_toolkit import load_reports

# Load all processed PFD reports (as a pandas DataFrame)
df = load_reports(processed=True)

# You can filter by category and date range
df_suicide = load_reports(
    processed=True,
    category='suicide',
    start_date='2014-01-01',
    end_date='2025-03-01'
)
```
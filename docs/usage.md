# Getting started

## Installation

PFD Toolkit can be installed from pip as `pfd_toolkit`:

```bath
pip install pfd_toolkit
```

*Note: The package is not currently on Conda distributions.*

## Loading your first dataset

The quickest way to get started is by loading a pre-processed dataset. These datasets are updated weekly, meaning you always have access to the latest reports with minimal setup.

```py
from pfd_toolkit import load_reports

# Load 'all' PFD reports from April 2025
reports = load_reports(
    category='all', 
    start_date="2025-04-01",
    end_date="2025-04-30"
    )

# Preview reports
reports.head()
```

`load_reports` returns a pandas DataFrame, and so accepts any pandas method.
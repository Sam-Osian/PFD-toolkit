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
    category = 'all', 
    start_date = "2025-04-01",
    end_date = "2025-04-30")

# Preview reports
reports.head()
```

`load_reports` returns a pandas DataFrame, and so accepts any pandas method.

## Identify relevant reports

With PFD Toolkit, you can query reports in plain English. This lets you identify reports that match your precise research questions, even when the terminology varies or is incorrectly tagged in the original data.

Instead of filtering by pre-set categories or keywords, simply describe the cases or themes you are interested in, and the toolkit will return a curated dataset of matching reports.

For this code to run, you must first set up an LLM client. Replace "XXXXXX" with your API key.

```py
# Set up LLM client
llm_client = LLM(api_key[XXXXXX], max_workers=30)

# Create a user query to filter reports
# ...this can be as broad or narrow as you like
user_query = "Deaths that occurred following ordering medications online"

# Set up the filtering engine
report_filter = Filter(llm = llm_client,
                        reports = reports,
                        user_query = user_query)

# And filter reports!
# ...`filtered_reports` is still a pandas DataFrame
filtered_reports = report_filter.filter_reports()
```
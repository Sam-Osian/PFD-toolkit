# Getting started

This page talks you through an example workflow using PFD Toolkit. It doesn't cover everything: for more, checkout the various pages on the left panel.

---

## Installation

PFD Toolkit can be installed from pip as `pfd_toolkit`:

```bath
pip install pfd_toolkit
```

*Note: The package is not currently on Conda distributions.*

---

## Load your first dataset

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

---

## Screen for relevant reports

PFD Toolkit lets you query reports in plain English — no need to know precise keywords or categories. Just describe the cases you care about, and the toolkit will return matching reports.

Note: Screening and other advanced features use AI models and require you to [set up an LLM client](llm_setup.md).

```py
# Set up LLM client
llm_client = LLM(api_key=YOUR-API-KEY) # Replace with actual API key

# Create a user query to filter reports
# ...this can be as broad or narrow as you like
user_query = "Deaths that occurred in police custody"

# Set up the filtering engine
screener = Screener(llm = llm_client,
                        reports = reports,
                        user_query = user_query)

# And filter reports!
filtered_reports = screener.screen_reports()
```
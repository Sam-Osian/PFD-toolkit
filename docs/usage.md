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
import pandas as pd

# Load 'all' PFD reports from April 2025
reports = load_reports(
    category = 'all', 
    start_date = "2025-04-01",
    end_date = "2025-04-30")

# Preview reports
reports.head()
```

`load_reports` will output a pandas DataFrame:

| URL                                   | ID         | Date       | CoronerName      | Area                           | Receiver                  | InvestigationAndInquest           | CircumstancesOfDeath       | MattersOfConcern         |
|----------------------------------------|------------|------------|------------------|--------------------------------|---------------------------|-----------------------------------|----------------------------|--------------------------|
| https://www.judiciary.uk/prevention...<br> | 2025-0207 | 2025-04-30 | Alison Mutch     | Manchester South               | Flixton Road...        | On 1st October...                 | Louise Danielle...         | During the course...     |
| https://www.judiciary.uk/prevention...<br> | 2025-0208 | 2025-04-30 | Joanne Andrews   | West Sussex...       | West Sussex County...     | On 02 November...                 | Mrs Turner drove...        | During the course...     |
| https://www.judiciary.uk/prevention...<br> | 2025-0120 | 2025-04-25 | Mary Hassell     | Inner North London             | The President...       | On 23 August...                   | Jan was a big baby...      | During the course...     |
| https://www.judiciary.uk/prevention...<br> | 2025-0206 | 2025-04-25 | Jonathan Heath   | North Yorkshire and York       | Townhead Surgery          | On 04 June...                     | On 15 March 2024...        | During the course...     |
| https://www.judiciary.uk/prevention...<br> | 2025-0199 | 2025-04-24 | Samantha Goward  | Norfolk                        | The Department...         | On 22 August...                   | In summary, on...          | During the course...     |




---

## Screen for relevant reports

PFD Toolkit lets you query reports in plain English — no need to know precise keywords or categories. Just describe the cases you care about, and the toolkit will return matching reports.

### Set up an LLM client

Screening and other advanced features use AI models and require you to first [set up an LLM client](llm_setup.md).

```python
from pfd_toolkit import LLM

# Set up LLM client
llm_client = LLM(api_key=YOUR-API-KEY) # Replace with actual API key
```

### Screen reports in plain English

Suppose you want to screen for reports based on a description, such as:

> "Deaths that occurred in police custody"

PFD Toolkit's `Screener` allows you to submit this as a user query, returning relevant reports:

```python
from pfd_toolkit import Screener

# Create a user query to filter reports
user_query = "Deaths that occurred in police custody"

# Set up the screening/filtering engine
screener = Screener(llm = llm_client,
                        reports = reports, # Reports that you loaded earlier
                        user_query = user_query) 

# And screen/filter reports!
filtered_reports = screener.screen_reports()
```

`filtered_reports` also returns a pandas DataFrame, but only contains reports that matched your query.
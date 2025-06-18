# Getting started

This page talks you through an example workflow using PFD Toolkit: loading a dataset and screening for relevant cases related to "detention under the Mental Health Act". 

This is just an example. PFD reports contain a breadth of information across a whole range of topics and domains.

It doesn't cover everything: for more, we strongly suggest browsing through the pages in the top panel.

---

## Installation

PFD Toolkit can be installed from pip as `pfd_toolkit`:

```bash
pip install pfd_toolkit
```

!!! Note
    PFD Toolkit is not currently available via Anaconda. If you'd like this to change, please make a [GitHub Issue](https://github.com/Sam-Osian/PFD-toolkit/issues). Personally, we love using [`uv`](https://docs.astral.sh/uv/concepts/projects/dependencies/) as an alternative to (Ana)conda for dependency management.

---

## Load your first dataset

First, you'll need to load a PFD dataset. These datasets are updated weekly, meaning you always have access to the latest reports with minimal setup.

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


---

## Screen for relevant reports

You're likely using PFD Toolkit because you want to answer a specific question. For example: "Do any PFD reports raise concerns related to detention under the Mental Health Act?"

*[detention]: Often referred to as 'being sectioned'

PFD Toolkit lets you query reports in plain English — no need to know precise keywords or categories. Just describe the cases you care about, and the toolkit will return matching reports.

### Set up an LLM client

Screening and other advanced features use AI, and require you to first set up an LLM client. You'll need to head to [platform.openai.com](https://platform.openai.com/docs/overview) and create an API key. Once you've got this, simply feed it to the `LLM`.

For more information, see [Settting up an LLM client](../llm_setup.md).

```python
from pfd_toolkit import LLM

# Set up LLM client
llm_client = LLM(api_key=YOUR-API-KEY) # Replace with actual API key
```

### Screen reports in plain English

Now, all we need to do is specify our `user_query` (the statement the LLM will use to filter reports), and set up our `Screener` engine.


```python
from pfd_toolkit import Screener

# Create a user query to filter
user_query = "Concerns related to detention under the Mental Health Act **only**"

# Screen reports
screener = Screener(llm = llm_client,
                        reports = reports) # Reports that you loaded earlier

filtered_reports = screener.screen_reports(user_query=user_query)
```

`filtered_reports` returns a DataFrame, only containing reports that matched your query.


For more information on Screening reports, see [Screening relevant reports](../screener/index.md).




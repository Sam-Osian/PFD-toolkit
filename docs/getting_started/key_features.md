---
title: "Key features"
description: |
  Short tour of the main capabilities of PFD Toolkit.
---

# Key features

PFD Toolkit turns raw PFD reports into structured insights. This page provides a short tour of the core features so you can hit the ground running. Each section links to more detailed guidance.

This is only an overview — for full walkthroughs start with the [Load live report data](../loader/load_reports.md) guide in the Explore section.

---

## Load live report data

Access the latest PFD reports in a single line. `load_reports()` returns a `DataFrame` containing the main sections from each report.

```python
from pfd_toolkit import load_reports

# Load reports from January 2024 onwards
reports = load_reports(
    start_date="2024-01-01" # YYYY-MM-DD format
)
```

Learn more on the [loading report data](../loader/load_reports.md) page.

---

## Set up an LLM client

Most features rely on an OpenAI model. Create an `LLM` client and supply your API key:

```python
from pfd_toolkit import LLM

llm_client = LLM(api_key=YOUR_API_KEY)
```

For more information on LLM setup (including how to get an OpenAI API key) see [Setting up an LLM](../llm_setup.md).


!!! warning
    OpenAI currently mandate that your account must be at least than 48 hours old before being able to run LLMs as normal. If you're setting up your account for this first time, you might have to wait a couple of days before using PFD Toolkit's advanced features.


### Hiding your API key

We heavily recommend keeping your API key safe by hiding it from your script. Create a separate file called `api.env` which contains:

```sh
OPENAI_API_KEY = [copy & paste your API key here]
```

...and import it via:

```python
from pfd_toolkit import LLM
from dotenv import load_dotenv
import os

# Load OpenAI API key
load_dotenv("api.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialise LLM client
llm_client = LLM(api_key=openai_api_key)
```


---

## Screen reports with natural language

Use `Screener` to filter thousands of reports according to a plain‑English query. For example, if you wanted to filter reports concerning deaths in police custody, you could run:

```python
from pfd_toolkit import Screener

query = "Deaths in police custody **only**."

screener = Screener(llm=llm_client, reports=reports)
police_reports = screener.screen_reports(search_query=query)
```

`police_reports` only contains reports that the LLM believed matched your query.

See [Filter reports with a query](../screener/index.md) for further tips and options.

---

## Create concise summaries

Long reports can be trimmed into short summaries for quick review. The `Extractor` class handles this via `summarise()`.

```python
from pfd_toolkit import Extractor

extractor = Extractor(llm=llm_client, reports=police_reports)
summary_df = extractor.summarise()
```

Read more on [producing summaries](../extractor/summarising.md).

---

## Discover recurring themes

Once summaries exist, you can instruct the LLM to find common topics across your dataset.

```python
IdentifiedThemes = extractor.discover_themes()

# Optionally, print the identified themes + description
#print(extractor.identified_themes)
```

The returned model can then be used to label each report:

```python
from pfd_toolkit import Extractor

extractor = Extractor(llm=llm_client, reports=police_reports)

labelled = extractor.extract_features(
    feature_model=IdentifiedThemes,
    force_assign=True,
    allow_multiple=True,
)
```

Detailed guidance is available under [discover recurring themes](../extractor/themes.md).

---

## Extract structured features

Finally, define your own variables with `pydantic` and pull them from the text.

```python
from pfd_toolkit import Extractor
from pydantic import BaseModel, Field

extractor = Extractor(llm=llm_client, reports=police_reports)

class MyFeatures(BaseModel):
    age: int = Field(description="Age of the deceased, if provided")
    sex: str = Field(description="Sex of the deceased (you can infer from pronouns)")

features = extractor.extract_features(
    feature_model=MyFeatures,
    allow_multiple=True,
)
```

`features` now contains `age` and `sex` columns.

See [pull out structured data](../extractor/basics.md) for advanced usage and configuration.

---

With these six steps — loading data, creating an LLM client, screening reports, summarising, discovering themes and extracting features — you can transform months of manual work into a streamlined workflow.
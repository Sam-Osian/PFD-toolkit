---
title: "Produce summaries of report text"
description: |
  Generate concise summaries to quickly scan large numbers of reports.
---

# Produce summaries of report text

Short, consistent summaries make it possible to skim hundreds or even of thousands of reports. You might use these summaries to quickly identify notable case studies, or to get an 'at a glance' understanding of the breadth contained within PFD reports.

Producing summaries also paves the way for automated theme discovery (see [Discover recurring themes](themes.md)). 

Use `summarise()` to condense each report into a short text snippet. The `trim_intensity` option controls how terse the summary should be. Calling `summarise` adds a `summary` column to your stored reports and keeps a copy on the instance under `extractor.summarised_reports` for later reuse.

---

## Getting started

You'll likely wish to screen/filter reports with `Screener` before generating summaries. For example:

```python
from pfd_toolkit import load_reports, LLM, Screener

# Load reports
reports = load_reports()

# Set up your LLM client
llm_client = LLM(api_key=YOUR-API-KEY)

# Screen reports by search query
search_query = "Deaths in police custody **only**."

screener = Screener(
    llm=llm_client,
    reports=reports
)

police_df = screener.screen_reports(
    search_query=search_query)

```

!!! note
    For more information on filtering reports, see [Filter reports with a query](../screener/index.md).

Following this, we can generate summaries of our screened/filtered reports:

```python
from pfd_toolkit import Extractor

# Set up Extractor
extractor = Extractor(
    llm=llm_client,
    reports=reports
)

summary_df = extractor.summarise(trim_intensity="medium")
```

The resulting DataFrame contains a new column (default name `summary`). 

You can specify a different column name via `result_col_name` if desired. You can also set a different `trim_intensity` (options range from `low` to `very high`) if desired.


## Specify which sections to summarise

By default, the `summarise()` method will trim the Investigation, Circumstances of Death and Coroner's Concerns sections. You can override this by setting the `include_*` flags. For example:

```py
# Set up Extractor
extractor = Extractor(
    llm=llm_client,
    reports=reports,

    # Decide which sections to include:
    include_investigation=False,
    include_circumstances=False,
    include_concerns=True
)

summary_df = extractor.summarise(trim_intensity="medium")
```

#### All options and defaults

<table>
  <thead>
    <tr>
      <th style="width:22%">Flag</th>
      <th>Report section</th>
      <th>What it's useful for</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>include_coroner</code></td>
      <td>Coroner’s name</td>
      <td>Simply the name of the coroner.</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>include_area</code></td>
      <td>Coroner’s area</td>
      <td>The local area the coroner operates within.</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>include_receiver</code></td>
      <td>Receiver(s) of the report</td>
      <td>The recipient(s) of the reports.</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>include_investigation</code></td>
      <td>“Investigation &amp; Inquest” section</td>
      <td>Contains procedural detail about the inquest.</td>
      <td><code>True</code></td>
    </tr>
    <tr>
      <td><code>include_circumstances</code></td>
      <td>“Circumstances of Death” section</td>
      <td>Describes what actually happened; holds key facts about the death.</td>
      <td><code>True</code></td>
    </tr>
    <tr>
      <td><code>include_concerns</code></td>
      <td>“Coroner’s Concerns” section</td>
      <td>Lists the issues the coroner believes should be addressed.</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>


---


## Estimating token counts

Token usage is important when working with paid APIs. The `estimate_tokens()` helper provides a quick approximation of how many tokens a text column will consume.

```python
total = extractor.estimate_tokens()
print(f"Total tokens in summaries: {total}")
```

`estimate_tokens` defaults to the summary column, but you can pass any text series via `col_name`. Set `return_series=True` to get a per-row estimate instead of the total.

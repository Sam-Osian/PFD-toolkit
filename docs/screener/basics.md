# Getting started

Natural-language filtering is one of the headline features of PFD Toolkit. The `Screener` class lets you describe a topic in plain English – e.g. "deaths in police custody" – and have an LLM screen reports, delivering you a curated dataset.

To use the `Screener` you'll first need to [set up an LLM client](../llm_setup.md) if you haven't already.

---

## A minimal example

First, import the necessary modules, load reports and set up an `LLM` client:

```python
from pfd_toolkit import load_reports, LLM, Screener

# Grab all reports from 2024
reports = load_reports(start_date="2023-01-01",
                       end_date="2023-12-31")

# Set up your LLM client
llm_client = LLM(api_key=YOUR-API-KEY)
```

Then define a `user_query` which describes the reports you're interested in. Pass this query as an argument to `screen_reports()` and you'll be given a filtered dataset containing only reports which the LLM judged to have matched your query.

```python
user_query = "Deaths in police custody **only**."

screener = Screener(
    llm=llm_client,
    reports=reports
)

police_df = screener.screen_reports(
    user_query=user_query)
```

`police_df` will now only contain reports related to your query.

---

## Why not just have a "normal" search function?

A keyword search is only as good as the exact words you type. Coroners, however, don't always follow a shared vocabulary. The same idea can surface in wildly different forms:

* *Under-staffing* might be written as **"staff shortages," "inadequate nurse cover,"** or even **"resource constraints."**
* *Suicide in prisons* may masquerade as **"self-inflicted injury while remanded,"** **"ligature event in cell,"** or may not even appear together in the same sentence.

A keyword filter misses these variants unless you identify every synonym in advance. By contrast, an `LLM` understands the context behind your query and links the phrasing for you, which is exactly what `Screener` taps into.
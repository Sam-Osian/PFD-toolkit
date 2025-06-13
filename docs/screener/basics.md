# Getting started

Natural-language filtering is one of the headline features of PFD Toolkit. The `Screener` class lets you describe a topic in plain English – e.g. "deaths in police custody" – and have an LLM screen reports, delivering you a curated dataset.

You do not have to use `Screener` to benefit from the toolkit. If the built-in category tags are good enough, the `load_reports()` helper will give you a ready-made dataset without any LLM calls. `Screener` is offered for those projects that need something more nuanced than the provided broad category tags.

To use the `Screener` you'll need to [set up an LLM client](../llm_setup.md).

---

## A minimal example

First, import the necessary modules, load reports and set up an `LLM` client:

```python
from pfd_toolkit import load_reports, LLM, Screener

# Grab the pre-processed April 2025 dataset
reports = load_reports(category="all",
                       start_date="2025-04-01",
                       end_date="2025-04-30")

# Set up your LLM client (see “Creating an LLM client” for details)
llm_client = LLM(api_key="YOUR-API-KEY")
```

Then describe the reports you're interested in, pass the query to `Screener` and you'll be given a filtered dataset containing matching reports.

```python
user_query = "deaths in police custody"

screener = Screener(
    llm=llm_client,
    reports=reports
)

police_df = screener.screen_reports(user_query=user_query, filter_df=True)

print(f"{len(police_df)} reports matched.")
>> "13 reports matched."
```

---

## Why not just have a normal "search" function?

A keyword search is only as good as the exact words you type. Coroners, however, don't always follow a shared vocabulary. The same idea can surface in wildly different forms:

* *Under-staffing* might be written as **"staff shortages," "inadequate nurse cover,"** or even **"resource constraints."**
* *Suicide in prisons* may masquerade as **"self-inflicted injury while remanded,"** **"ligature event in cell,"** or appear across separate sentences.

A keyword filter misses these variants unless you guess every synonym in advance. By contrast, an `LLM` understands the context behind your query and links the phrasing for you, which is exactly what `Screener` taps into.

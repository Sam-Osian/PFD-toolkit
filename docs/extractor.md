# Extract information from reports

The `Extractor` class uses an `LLM` to pull structured information from Prevention of Future Death (PFD) reports.  You define the fields you want by passing a `pydantic` model and `Extractor` handles prompting the LLM and adding the results to your dataset.

In addition to factual extraction, `Extractor` can label reports with themes so you can create your own taxonomy.  The themes can be manually specified via a feature model or automatically discovered using the summarisation utilities (described later in this guide).

---

## Basic usage

Start by defining a feature model with `pydantic`.  Each attribute represents a piece of information you want to pull out of the report.  `Extractor` accepts any valid `BaseModel`, so feel free to mix strings, numbers or more complex types:

```python
from pydantic import BaseModel, Field
from pfd_toolkit import load_reports, LLM, Extractor

# Define feature model with pydantic
class MyFeatures(BaseModel):
    age: int
    cause_of_death: str
```

Next, load some report data and [set up your LLM client](llm_setup.md).  You then pass the feature model, the reports and the LLM client to an `Extractor` instance and call `.extract_features()`:

```python
reports = load_reports(category="all", start_date="2024-01-01", end_date="2024-12-31")
llm_client = LLM(api_key="YOUR-API-KEY")

extractor = Extractor(
    feature_model=MyFeatures,
    reports=reports,
    llm=llm_client
)

result_df = extractor.extract_features()
```

`result_df` now contains the new ``age`` and ``cause_of_death`` columns.  You can repeat the call with a different feature model to extract further information – the cached results mean previously processed rows will not be re-sent to the LLM unless you clear the cache with `.reset()` (see below).

---

## Provide 'themes' to categorise the reports

`Extractor` can also be used to tag reports with your own themes.  Each field on the feature model represents a potential tag.  In the model below the ``falls_in_custody`` field indicates whether a death occurred in police custody.

Set ``force_assign=True`` so the LLM always returns either ``True`` or ``False`` for each field.  ``allow_multiple=True`` lets a single report be marked with more than one theme if required.


```python
# For themes, we recommend always using the `bool` flag
class Themes(BaseModel):
    falls_in_custody: bool = Field(description="Death occurred in police custody")
    medication_error: bool = Field(description="Issues with medication or dosing")

extractor = Extractor(
    llm=llm_client,
    feature_model=Themes,
    reports=reports,
    force_assign=True,
    allow_multiple=True,
)

labelled = extractor.extract_features()
```

The returned DataFrame includes a boolean column for each theme.

---

## Choosing which sections the LLM reads

`Extractor` lets you decide exactly which parts of the report are presented to the model.  Each ``include_*`` flag mirrors one of the columns loaded by ``load_reports``.  Turning fields off reduces the amount of text sent to the LLM which often speeds up requests and lowers token usage.

```python
extractor = Extractor(
    llm=llm_client,
    reports=reports,
    include_investigation=True,
    include_circumstances=True,
    include_concerns=False,  # Skip coroner's concerns if not relevant
)
```

In this example only the investigation and circumstances sections are provided to the LLM.  The coroner's concerns are omitted entirely.  Limiting the excerpt like this often improves accuracy and drastically reduces token costs. However, be careful you're not turning 'off' a report section which is genuinely useful for your query.

---

## Summarising reports

Use `.summarise()` to condense each report into a short text snippet.  The ``trim_intensity`` option controls how terse the summary should be.  Calling ``summarise`` adds a ``summary`` column to your stored reports and keeps a copy on the instance under ``extractor.summarised_reports`` for later reuse.

```python
summary_df = extractor.summarise(trim_intensity="medium")
summary_df[["summary"]].head()
```

The resulting DataFrame contains a new column (default name ``summary``).  You can specify a different column name via ``result_col_name`` if desired.

---

## Discovering themes automatically

Instead of having a prescribed list of themes ahead of time, you may wish to 
automatically discover themes contained within your selection of reports.

Once summaries are available you can call `.discover_themes()` to let the
LLM propose a list of recurring themes.  ``.discover_themes()`` reads the
``summary`` column created by `.summarise()` (if you skip summarisation the
method will raise an error).

The function returns a ``pydantic`` model describing the discovered themes.  You
can immediately feed that model back into :meth:`extract_features` to label each
report.

```python
IdentifiedThemes = extractor.discover_themes()

# Optionally, inspect the newly identified themes:
# print(IdentifiedThemes)

assigned_reports = extractor.extract_features(
                              feature_model=IdentifiedThemes
                            # Recommended: set below parameters to True
                              force_assign=True, 
                              allow_multiple=True)
```


``theme_df`` will include a boolean column for each discovered theme.

``discover_themes`` accepts several parameters:

* ``warn_exceed`` and ``error_exceed`` – soft and hard limits for the estimated
  token count of the combined summaries.  Exceeding ``error_exceed`` raises an
  exception. You may wish to set either of these to the 'context window' of 
  your chosen LLM model.
* ``max_themes`` / ``min_themes`` – bound the number of themes the model should
  return.
* ``seed_topics`` – either a string, list or ``BaseModel`` of starter topics.  The
  LLM will incorporate these into the final list.
* ``extra_instructions`` – free‑form text appended to the prompt, allowing you
  to steer the LLM towards particular areas of interest.

For example, a more advanced call might look like:

```python
feature_model = extractor.discover_themes(
                            max_themes=10, min_themes=5,
                            seed_topics="medication deaths; bad risk assessment",
                            extra_instructions="Themes must be practical/operational - not too broad")

assigned_reports = extractor.extract_themes()
```

---

## Estimating token counts

Token usage is important when working with paid APIs. The `estimate_tokens()` helper provides a quick approximation of how many tokens a text column will consume.

```python
total = extractor.estimate_tokens()
print(f"Total tokens in summaries: {total}")
```

``estimate_tokens`` defaults to the summary column, but you can pass any text
series via ``col_name``.  Set ``return_series=True`` to get a per-row estimate
instead of the total.

---

## Caching and exporting results

`Extractor` caches every LLM response so repeated calls with the same prompt
reuse previous results.  Export the cache before you shut down and import it in
a future session to avoid paying for the same completions twice.

```python
extractor.export_cache("my_cache.pkl")
...
extractor.import_cache("my_cache.pkl")
```

If you want to start fresh, call ``reset()`` to clear cached feature values and
token estimates.  This is useful when you wish to re-run ``extract_features`` on
the same DataFrame with a different feature model.  ``reset`` returns the
instance so you can immediately chain another call:

```python
clean_df = extractor.reset().extract_features(feature_model=NewModel)
```

The returned DataFrame contains your newly extracted features and an empty cache
ready for further runs.


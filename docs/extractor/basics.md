# Basic usage

Start by defining a feature model with `pydantic`. Each attribute represents a piece of information you want to pull out of the report. `Extractor` accepts any valid `BaseModel`, so feel free to mix strings, numbers or more complex types:

```python
from pydantic import BaseModel, Field
from pfd_toolkit import load_reports, LLM, Extractor

# Define feature model with pydantic
class MyFeatures(BaseModel):
    age: int
    cause_of_death: str
```

Next, load some report data and [set up your LLM client](../llm_setup.md). You then pass the feature model, the reports and the LLM client to an `Extractor` instance and call `.extract_features()`:

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

`result_df` now contains the new `age` and `cause_of_death` columns. You can repeat the call with a different feature model to extract further information – the cached results mean previously processed rows will not be re-sent to the LLM unless you clear the cache with `.reset()`.

---

## Choosing which sections the LLM reads

`Extractor` lets you decide exactly which parts of the report are presented to the model. Each `include_*` flag mirrors one of the columns loaded by `load_reports`. Turning fields off reduces the amount of text sent to the LLM which often speeds up requests and lowers token usage.

```python
extractor = Extractor(
    llm=llm_client,
    reports=reports,
    include_investigation=True,
    include_circumstances=True,
    include_concerns=False  # Skip coroner's concerns if not relevant
)
```

In this example only the investigation and circumstances sections are provided to the LLM. The coroner's concerns are omitted entirely. Limiting the excerpt like this often improves accuracy and drastically reduces token costs. However, be careful you're not turning 'off' a report section which is genuinely useful for your query.

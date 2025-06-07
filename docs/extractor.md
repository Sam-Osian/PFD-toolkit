# Extract information from reports

The `Extractor` class uses an `LLM` to pull structured information from Prevention of Future Death (PFD) reports. You define the fields you want by passing a `pydantic` model and `Extractor` handles prompting the LLM and adding the results to your dataset.

Beyond pulling out factual details, `Extractor` can also assign themes to each report. Simply create a model with your desired categories and the class will label reports for you. This is useful if you want to create your own thematic taxonomy instead of relying on the built‑in categories.

---

## Basic usage

Start by defining a feature model with `pydantic`:

```python
from pydantic import BaseModel
from pfd_toolkit import load_reports, LLM, Extractor

class MyFeatures(BaseModel):
    age: int
    cause_of_death: str
```

Next, load some report data and [set up your LLM client](llm_setup.md).

Pass the feature model, the reports, and the LLM client to your instance of `Extractor` and call `extract_features()`.

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

`result_df` now contains the new `age` and `cause_of_death` columns.

---

## Thematic assignment

You can repurpose `Extractor` to tag reports with your own themes. Define a model where the field represents your label. 

In your feature model (`Themes`), we recommend setting the 'type' to `bool` so that the model outputs True or False for each theme. Set `force_assign=True` so the LLM does not return missing data when choosing a category (and instead provides 'False'). 

Use `allow_multiple=True` if a report may fall under several themes.


```python
class Themes(BaseModel):
    category: str = Field(description="Chosen theme for this report")

extractor = Extractor(
    llm=llm_client,
    feature_model=Themes,
    reports=reports,
    force_assign=True,
    allow_multiple=True,
)

labelled = extractor.extract_features()
```

This produces a `category` column with your thematic assignment.

---
title: "Pull out structured data from reports"
description: |
  Learn how the Extractor converts free text into structured columns.
---

# Pull out structured data from reports

`Extractor` lets you turn free text into structured columns. This is useful for running statistics, tracking patterns, and comparing cases at scale.

Start by defining a feature model with `pydantic`. 

Each feature attribute should contain three pieces of information: the variable name (e.g. `"age"`), its type (e.g. `int`) and a short description to aid the LLM in extracting the desired information:

```python
from pydantic import BaseModel, Field
from pfd_toolkit import load_reports, LLM, Extractor

# Define feature model with pydantic
class MyFeatures(BaseModel):
    age: int = Field(description="The age of the deceased")
    sex: str = Field(description="The sex of the deceased. You may infer sex from pronouns (e.g. 'He', 'Her', etc.)")
    cause_of_death: str = Field(description="A one-sentence summary of the cause of death")
```

!!! note
    As per the example above, your BaseModel instance must specify each of the field `types`. Use `int` for numbers, `str` for strings, and `bool` for binary values.

    `Extractor` accepts any valid BaseModel configuration. For more customisation, please read [Pydantic's documentation](https://docs.pydantic.dev/latest/concepts/fields/).


Next, load some report data and [set up your LLM](../llm_setup.md). You then pass the feature model, the reports and the LLM client to an `Extractor` instance and call `extract_features()`:

```python
reports = load_reports(start_date="2024-01-01", end_date="2024-12-31")
llm_client = LLM(api_key=YOUR-API-KEY)

extractor = Extractor(
    reports=reports,
    llm=llm_client
)

result_df = extractor.extract_features(
        feature_model=MyFeatures,
        allow_multiple=True, 
        force_assign=False
)
```

`result_df` now contains the new `age`, `sex`, and `cause_of_death` columns. 


!!! note
    Where the model was unable to extract any given piece of structured data, it will output missing data. Setting `force_assign` to `True` forces the model to output a value for each feature, even if it cannot be found. 

    In general, this is only recommended if you are working with binary values (of type `bool`). For example:

    ```py
    class MyFeatures(BaseModel):
    care_home: bool = Field(description="Whether or not the death took place in a care home")

    result_df = extractor.extract_features(
        feature_model=MyFeatures,
        allow_multiple=True, 
        force_assign=True  #  <-- force the model to output either `True` or `False`
    )
    ```



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

In this example only the investigation and circumstances sections are provided to the LLM. The coroner's concerns are omitted entirely. Limiting the excerpt like this may improve accuracy and reduce token costs. However, be careful you're not turning 'off' a report section which could contain information relevant for one of your features.

## Re-run the extraction

By default, the `Extractor` class won't run on the same data with the same configuration twice. 

If you want to start fresh, call `reset()` to clear cached feature values, and chain it into a new `extract_features()` call:

```python
clean_df = extractor.reset().extract_features(feature_model=NewModel)
```

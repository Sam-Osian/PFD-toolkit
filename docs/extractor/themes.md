---
title: "Searching for themes"
description: |
  Discover recurring topics across many PFD reports.
---

# Searching for themes

Spotting common themes across many reports helps reveal systemic problems and policy gaps. `Extractor` can be used to identify & label these themes easily.

!!! important
    Extracting themes works best if you've already screened for reports that are relevant to your research. For more information, see the guide on [search for matching cases](../screener/index.md).

---

## Discovering themes

The `discover_themes()` method allows you to identify recurring topics contained within a selection of PFD reports. 

Once summaries are available, you can instruct the LLM to identify a list of recurring themes. This method expects that the `summary` column has already been created by `summarise()` (see [Produce summaries of report text](summarising.md)).


```python

from pfd_toolkit import Extractor, LLM

# Set up Extractor
extractor = Extractor(
    llm=llm_client,
    reports=reports
)

summary_df = extractor.summarise(trim_intensity="medium")

IdentifiedThemes = extractor.discover_themes()

# Optionally, inspect the themes that the model has identified:
#print(extractor.identified_themes)
```

`IdentifiedThemes` is essentially a set of detailed instructions that you can pass to the LLM via `extract_features()`:

```python
assigned_reports = extractor.extract_features(
                              feature_model=IdentifiedThemes,
                              force_assign=True,
                              allow_multiple=True)
```

`assigned_reports` now contains your original dataset, along with new fields denoting whether the LLM assigned each report to a particular theme or not.  

## Tabulate themes

To create a table containing counts and percentages for each of your themes, run:

```python
extractor.tabulate()
```

---

## More customisation

`Extractor` contains a suite of options to help you customise the thematic discovery process.


### Choosing which sections the LLM reads

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

### Guided topic modelling

Guided topic modelling is a strategy of discovering themes where you provide a number of topics that are sure to be in your selection of reports. 

We can set one or more `seed_topics`, which the model will draw from while *also* discovering new themes. For example:


```python
# Set up Extractor
extractor = Extractor(
    llm=llm_client,
    reports=reports
)

summary_df = extractor.summarise(trim_intensity="medium")

IdentifiedThemes = extractor.discover_themes(
    seed_topics="Risk assessment failures; understaffing; information sharing failures"
)
```

Above, we provide 3 seed topics. The model will attempt to identify these topics in the text, while also searching for other, unspecified topics.

---

### Providing additional instructions

You can also provide additional instructions to help guide the model. This is somewhat similar to the guided topic modelling above, except instead of providing examples of themes, we can provide other kinds of guidance. For example:


```python
summary_df = extractor.summarise(trim_intensity="medium")

extra_instructions="""
My research question is: What are the various consequences of transitioning from youth to adult mental health services?"
"""

IdentifiedThemes = extractor.discover_themes(
    extra_instructions=extra_instructions
)
```

Above, we guide the model by specifying our specific area of interest. This will help to keep themes focused around our core research question. 


### Controlling the number of themes

You can control how many themes the model discovers through `min_themes` and `max_themes` arguments:

```python
summary_df = extractor.summarise(trim_intensity="medium")

IdentifiedThemes = extractor.discover_themes(
    min_theme=8,
    max_theme=12
)
```

`discover_themes` will now produce at least 8 themes, but not more than 12, themes.



### Manual topic modelling

Finally, you can bypass `discover_themes()` altogether by providing a complete set of themes to extract via a feature model. Here, the model only *assigns* the themes; it does not *identify* the themes.

For each of your themes, you should provide 3 pieces of information: the column name (e.g. `falls_in_custody`), its type (e.g. `bool`) and a brief description.

Set `force_assign=True` so the LLM always returns either `True` or `False` for each field. `allow_multiple=True` lets a single report be marked with more than one theme if required.


```python
from pydantic import BaseModel, Field

# For themes, we recommend always setting the type to `bool`
class Themes(BaseModel):
    falls_in_custody: bool = Field(description="Death occurred in police custody")
    medication_error: bool = Field(description="Issues with medication or dosing")

extractor = Extractor(
    llm=llm_client,
    reports=reports,
)

labelled = extractor.extract_features(
    feature_model=Themes,
    force_assign=True,
    allow_multiple=True)

```

!!! note
    Tip: always select type `bool` for your themes for more reliable performance.


The returned DataFrame includes a boolean column for each of your themes.


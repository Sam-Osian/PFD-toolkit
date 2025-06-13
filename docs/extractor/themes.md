# Tagging reports with themes

`Extractor` can be used to label reports with your own themes. Each field on the feature model represents a potential tag. In the model below the `falls_in_custody` field indicates whether a death occurred in police custody.

Set `force_assign=True` so the LLM always returns either `True` or `False` for each field. `allow_multiple=True` lets a single report be marked with more than one theme if required.

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

## Discovering themes automatically

Instead of having a prescribed list of themes ahead of time, you may wish to automatically discover themes contained within your selection of reports.

Once summaries are available you can call `.discover_themes()` to let the LLM propose a list of recurring themes. `.discover_themes()` reads the `summary` column created by `.summarise()` (see [Summaries & token counts](summarising.md)).

The function returns a `pydantic` model describing the discovered themes. You can immediately feed that model back into `extract_features` to label each report.

```python
IdentifiedThemes = extractor.discover_themes()

# Optionally, inspect the newly identified themes:
# print(IdentifiedThemes)

assigned_reports = extractor.extract_features(
                              feature_model=IdentifiedThemes,
                              force_assign=True,
                              allow_multiple=True)
```

`discover_themes` accepts several parameters:

* `warn_exceed` and `error_exceed` – soft and hard limits for the estimated token count of the combined summaries. Exceeding `error_exceed` raises an exception.
* `max_themes` / `min_themes` – bound the number of themes the model should return.
* `seed_topics` – either a string, list or `BaseModel` of starter topics. The LLM will incorporate these into the final list.
* `extra_instructions` – free‑form text appended to the prompt, allowing you to steer the LLM towards particular areas of interest.

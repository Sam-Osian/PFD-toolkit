# Summaries & token counts

Use `.summarise()` to condense each report into a short text snippet. The `trim_intensity` option controls how terse the summary should be. Calling `summarise` adds a `summary` column to your stored reports and keeps a copy on the instance under `extractor.summarised_reports` for later reuse.

```python
summary_df = extractor.summarise(trim_intensity="medium")
summary_df[["summary"]].head()
```

The resulting DataFrame contains a new column (default name `summary`). You can specify a different column name via `result_col_name` if desired.

---

## Estimating token counts

Token usage is important when working with paid APIs. The `estimate_tokens()` helper provides a quick approximation of how many tokens a text column will consume.

```python
total = extractor.estimate_tokens()
print(f"Total tokens in summaries: {total}")
```

`estimate_tokens` defaults to the summary column, but you can pass any text series via `col_name`. Set `return_series=True` to get a per-row estimate instead of the total.

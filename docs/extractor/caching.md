# Caching and exporting results

`Extractor` caches every LLM response so repeated calls with the same prompt reuse previous results. Export the cache before you shut down and import it in a future session to avoid paying for the same completions twice.

```python
extractor.export_cache("my_cache.pkl")
...
extractor.import_cache("my_cache.pkl")
```

If you want to start fresh, call `reset()` to clear cached feature values and token estimates. This is useful when you wish to re-run `extract_features` on the same DataFrame with a different feature model. `reset` returns the instance so you can immediately chain another call:

```python
clean_df = extractor.reset().extract_features(feature_model=NewModel)
```

The returned DataFrame contains your newly extracted features and an empty cache ready for further runs.

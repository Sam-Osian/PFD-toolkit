# Analysing PFD reports

PFD Toolkit ships with an `Extractor` class to pull "features" (i.e. key pieces of information) from Prevention of Future Death (PFD) reports. 

These features could be recurring themes, or more specific bits of information (e.g. age, sex, cause of death, etc.).

The guides below walk through the main features:

- [Basic usage](basics.md) – create a basic feature model to identify features from report data.
- [Summaries & token counts](summarising.md) – generate short summaries and estimate the token cost of your data.
- [Tagging reports with themes](themes.md) – automatically discover recurring themes or label reports with your own taxonomy.
- [Capturing text spans](spans.md) – keep short excerpts ("spans") showing where each feature came from.
- [Caching and exporting results](caching.md) – reuse completions to save time and API costs.
---
title: "Cleaning scraped data"
description: |
  Polish scraped reports using the Cleaner for consistent data.
---

# Cleaning scraped data

`Cleaner` provides an optional step for polishing scraped reports with the same
LLM used in the fallback stage. It loops over selected columns, sends field
specific prompts in batches and writes the corrected text back into a copy of
your DataFrame.

## Basic usage

```python
from pfd_toolkit import Cleaner

cleaner = Cleaner(df, llm, include_receiver=False)
clean_df = cleaner.clean_reports(anonymise=True)
```

The class works field by field so you decide which columns to modify. Use the
boolean flags like `include_receiver` or `include_area` to toggle each field.
Custom prompt strings allow advanced users to steer how the LLM rewrites the
text.

## Anonymising sensitive details

When `anonymise=True` the investigation, circumstances and concerns fields are
automatically de‑identified by swapping names and pronouns for gender‑neutral
placeholders.

The underlying logic constructs a field-specific prompt for every text snippet,
sends them to the LLM in batches and writes the results back into a copy of your
DataFrame.

Any prompts that return an error marker are ignored so the original text remains
untouched. Call `generate_prompt_template()` to preview the finalised prompts
before you run the clean.

## Prompt templates and error handling

`Cleaner` exposes the final prompt for each field via
`generate_prompt_template()`. This lets you inspect or tweak the wording before
sending anything to the model. If the LLM returns a recognised error marker, the
class reverts to the original text rather than introducing blank values.

See the [API reference](../reference/cleaner.md) for a detailed breakdown of
all arguments and attributes.

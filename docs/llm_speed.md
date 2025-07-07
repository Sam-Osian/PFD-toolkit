---
title: "Speed up your LLM"
description: |
  Increase throughput by running requests in parallel.
---

## Speed up your LLM

Process more reports in parallel by increasing the `max_workers` parameter. By default, this is set to `8`, but larger values can lead to faster run-times.

```python
llm_client = LLM(
    api_key=openai_api_key,
    max_workers=20      # Increase parallelisation
)
```

!!! note
    OpenAI enforces rate limits for each account and model. If you set `max_workers` too high, you may hit these limits and see errors or slowdowns. PFD Toolkit will automatically pause and retry if a rate limit is reached, but it’s best to keep `max_workers` within a reasonable range (usually 8 to 20 for most users).

    Your exact rate limit may depend on the 'tier' of your OpenAI account as well as the model you're using. If you need higher limits, you may be able to apply for an increase in your OpenAI account settings.

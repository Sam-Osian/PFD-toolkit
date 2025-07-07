---
title: "Change the model"
description: |
  Swap the underlying OpenAI model for a different version.
---

## Change the model

By default, PFD Toolkit uses `gpt-4.1`. We love this model as it balances cost, speed, and accuracy. We also recommend its smaller equivalent, `gpt-4.1-mini`, which offers decent performance at a lower API cost.

```python
llm_client = LLM(
    api_key=openai_api_key,
    model="gpt-4.1"     # Set model here
)
```

See OpenAI's [documentation](https://platform.openai.com/docs/models) for a complete list of their models.

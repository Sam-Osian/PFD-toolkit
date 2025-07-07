---
title: "Advanced usage"
description: |
  Configure custom endpoints, temperature, seed and other advanced settings.
---

## Use a custom endpoint

You can set a custom endpoint (e.g. for Azure, Ollama, etc.). Your endpoint must support the OpenAI SDK. 

```python
llm_client = LLM(
    api_key=openai_api_key,
    base_url="https://..."   # Set your custom endpoint
)
```

## Set temperature and seed

LLMs are highly stochastic, and may produce slightly different outputs for identical calls. 

We can reduce variation in responses through supplying `temperature` and `seed` parameters:

```py
llm_client = LLM(api_key=openai_api_key, 
                 temperature=0, 
                 seed=123)
```

`temperature` defaults to 0. To increase variation, you may increment its value to something like `0.2`. 

The `seed` parameter works similarly to random seed parameters used for other packages; it acts as a starting point for the model's random number generator during text generation.

!!! warning
    While we can reduce randomness in LLM responses, these models are non-deterministic and therefore may still produce varying responses even when the `seed` is set and `temperature` reduced to `0`.


## Set timeout

The `timeout` parameters dictates the maximum number of seconds the LLM should spend on processing each report. Essentially, if the LLM hasn't provided a response by the specified `timeout` value, then it gives up and moves on to the next.


```python
llm_client = LLM(api_key=openai_api_key, 
                 timeout=60)
```

# Creating an LLM Client

PFD Toolkit uses a Large Language Model (LLM) client for advanced features. This page explains how to set up your LLM, what an API key is, and why you might need these features.

---

## Setting up your LLM client

To use AI-powered features, you need to create an LLM client and supply your OpenAI API key ([how to get one below](#how-do-i-get-an-openai-api-key)). You do *not* need an LLM client to simply load report data (using `load_reports`).


*Basic setup:*

```python
from pfd_toolkit import LLM

llm_client = LLM(api_key=YOUR-API-KEY) # Replace YOUR-API-KEY with your real API key
```

You can now use LLM-powered features! For example, to screen for reports about medication purchased online:

```python
from pfd_toolkit import Screener

query = "Deaths that followed ordering medication(s) online."

screener = Screener(llm=llm_client, reports=reports)
online_med_reports = screener.screen_reports(user_query=query)
```

---

## How do I get an OpenAI API key?

1. Sign up or log in at [OpenAI Platform](https://platform.openai.com).
2. Go to [API Keys](https://platform.openai.com/api-keys).
3. Click “Create new secret key” and copy the string.
4. Store your key somewhere safe. **Never** share or publish it.
5. Add credit to your account (just $5 is enough for most research uses).

For more information about usage costs, see [OpenAI pricing](https://openai.com/api/pricing/).
---

## Speed up your LLM

Process more reports in parallel by increasing the `max_workers` parameter. By default, this is set to `8`, but larger values can lead to faster run-times.

```python
llm_client = LLM(
    api_key=openai_api_key,
    max_workers=30      # Increase parallelisation
)
```

!!! note
    OpenAI enforces rate limits for each account and model. If you set `max_workers` too high, you may hit these limits and see errors or slowdowns. PFD Toolkit will automatically pause and retry if a rate limit is reached, but it’s best to keep `max_workers` within a reasonable range (usually 8 to 20 for most users). 
    
    Your exact rate limit may depend on the 'tier' of your OpenAI account as well as the model you're using. If you need higher limits, you may be able to apply for an increase in your OpenAI account settings.

---

## Change your model

By default, PFD Toolkit uses `gpt-4.1-mini`. We love this model as it balances cost, speed, and accuracy. We also recommend its larger equivalent, `gpt-4.1`, which may offer improved performance, though with additional API costs and less forgiving rate limits.


```python
llm_client = LLM(
    api_key=openai_api_key,
    model="gpt-4.1"     # Set model here
)
```

See OpenAI's [documentation](https://platform.openai.com/docs/models) for a complete list of their models.

## Use a custom endpoint

You can set a custom endpoint (e.g. for Azure, Ollama, etc.) if it supports the OpenAI SDK:

```python
llm_client = LLM(
    api_key=openai_api_key,
    base_url="https://..."   # Set your custom endpoint
)
```
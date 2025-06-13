# Creating an LLM Client

PFD Toolkit uses a Large Language Model (LLM) client for advanced features. This page explains why you might need an LLM, how to set it up, and what you should know about API keys and costs.


---

## Why do I need an LLM client?

Many toolkit features — like advanced cleaning, screening reports, and assigning themes — depend on AI. These tasks aren’t reliable (or sometimes possible) with rule-based scripts alone.

To use these features, you’ll need to create an **LLM client** and pass it to the `Screener`, `Cleaner`, `Scraper`, or `Categoriser` objects. You do *not* need an LLM client to simply load report data (with `load_reports`).

We appreciate that not everyone using this package will have worked with API keys before, so we've made setup extra simple.

---

## Setting up your LLM client

Import the `LLM` class and provide your API key ([see below for details](#what-is-an-api-key-why-do-i-need-one)):

```py
from pfd_toolkit import LLM

llm_client = LLM(api_key=YOUR-API-KEY) # Replace YOUR-API-KEY with actual API key
```

And that's it — you can now use LLM-powered features! For example, to screen for reports about medication purchased online:

```py
from pfd_toolkit import Screener

query = "Deaths that followed ordering medication(s) online."

screener = Screener(llm=llm_client, # Assign llm client here
                        reports = reports)

online_med_reports = screener.screen_reports(user_query=query)

```

### Added security

It's important to never share your API key. This means making sure you don't commit your key to GitHub or similar services.

For added security, we recommend storing your API in a `.env` file (e.g. `api.env`) and importing it through `load_dotenv`. For example:

```py
# In your .env file (never commit this to GitHub!)
OPENAI_API_KEY=YOUR-API-KEY
```

Then, load the key in your script:

```py
from dotenv import load_dotenv
import os
from pfd_toolkit import LLM

load_dotenv("api.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm_client = LLM(api_key=openai_api_key)
```


---

## What is an API key?

All LLM features in PFD Toolkit use OpenAI’s servers. An API key is like a secret code that identifies you to OpenAI.

You pay OpenAI for usage, not PFD Toolkit. The toolkit is free and open source.
See [OpenAI pricing](https://openai.com/api/pricing/) for more — costs are based on tokens used (roughly, 1 word ≈ 1 token).


---

## How do I get an OpenAI API key?

1. Sign up or login at [OpenAI Platform](https://platform.openai.com).
2. Go to [API Keys](https://platform.openai.com/api-keys).
3. Click "Create new secret key", and copy the string.
4. Store this somewhere safe. **Never** share your API key or commit it to public code.
5. Add credit to your account (just $5 goes a long way for most research use).

---

## Advanced options

### Speed up your LLM

The LLM client supports parallelisation via the `max_workers` parameter. This controls the number of concurrent tasks the LLM can complete at once (each row/report is its own 'task'). For most workflows, set `max_workers` between 10-30.


```py
llm_client = LLM(
    api_key=openai_api_key,
    max_workers=30      # <--- increase parallelisation
)
```

OpenAI does impose [rate limits](https://platform.openai.com/docs/guides/rate-limits), however, so setting `max_workers` to an extremely high value may result in errors or slowdowns. 

PFD Toolkit tries to handle rate limit errors by briefly pausing the script once a rate limit has been exceeded. However, it's still good practice to set the parameter to a reasonable value to avoid errors.

---

### Change the model

By default, the LLM client will use `gpt-4.1-mini`. Our testing found that this offered the best balance between cost, speed and accuracy. However, you can change this to any supported [OpenAI model](https://platform.openai.com/docs/models).

```py
llm_client = LLM(
    api_key=openai_api_key,
    model="o4-mini"     # <--- change model to o4-mini
)
```

---

### Use a custom endpoint

You can redirect the LLM to any custom endpoint (e.g. Azure, OpenRouter), provided they support the OpenAI SDK.

```py
llm_client = LLM(
    api_key=openai_api_key,
    base_url="https://...",   # <--- Set custom endpoints
)
```


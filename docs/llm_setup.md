# Creating an LLM Client

PFD Toolkit uses a Large Language Model (LLM) client for advanced features. This page explains why you might need an LLM, how to set it up, and what you should know about API keys and costs.


---

## Why do I need an LLM client?

Many toolkit features — like advanced cleaning, screening reports, and assigning themes — depend on AI. These tasks aren’t reliable (or sometimes possible) with rule-based scripts alone.

To use these features, you’ll need to create an **LLM client** and pass it to the `Screener`, `Cleaner`, `Scraper`, or `Categoriser` objects.

You do *not* need an LLM client to simply load report data (with `load_reports`).

**Not used an LLM via API before?** Don’t worry, setup is simple.

---

## Setting up your LLM client

Import the `LLM` class and provide your *API key* ([see below for details](#what-is-an-api-key-why-do-i-need-one)):

```py
from pfd_toolkit import LLM

llm_client = LLM(api_key=YOUR-API-KEY) # Replace YOUR-API-KEY with actual API key
```

Now you can use LLM-powered features! For example, to screen for reports about medication purchased online:

```py
from pfd_toolkit import Screener

query = "Deaths that followed ordering medication(s) online."

screener = Screener(llm=llm_client, # Assign llm client here
                        reports = reports,
                        user_query=query)

online_med_reports = screener.screen_reports()

```

**Tip:** For security, store your API key in a `.env` file (e.g. `api.env`):

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

## What is an API key? Why do I need one?

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

You can customise the client with additional parameters:

```py
llm_client = LLM(
    api_key=openai_api_key,
    model="gpt-4.1",           # Use another named model if you prefer. We use gpt-4.1-mini by default.
    base_url="https://...",   # For custom endpoints (e.g. Azure, OpenRouter)
    max_workers=30            # Controls parallelism
)
```

Performance tips:

 - For most workflows, set `max_workers` between 10-30.

 - Higher values process more reports at once, but increase the risk of OpenAI rate limits. The toolkit contains retry logic, but you may still want to lower this if you see errors or slowdowns.


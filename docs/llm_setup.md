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

## How do I get an OpenAI API key?

1. Sign up or log in at [OpenAI Platform](https://platform.openai.com).
2. Go to [API Keys](https://platform.openai.com/api-keys).
3. Click “Create new secret key” and copy the string.
4. Store your key somewhere safe. **Never** share or publish it.
5. Add credit to your account (just $5 is enough for most research uses).

For more information about usage costs, see [OpenAI pricing](https://openai.com/api/pricing/).


## Why do I need an LLM client anyway?

Many toolkit features - like advanced cleaning, screening, and assigning themes - rely on AI, which goes far beyond what’s possible with rule-based scripts. If you want to use these features, you’ll need to set up an LLM client as described above.

We’ve made the setup as simple as possible, especially if you’re new to APIs. If you get stuck, please [reach out](contact.md): we’re happy to help.
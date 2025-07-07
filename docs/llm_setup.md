---
title: "Setting up an LLM"
description: |
  Create an LLM client and obtain an API key for advanced features.
---

# Setting up an LLM

PFD Toolkit uses a Large Language Model (LLM) client for advanced features. This page explains how to set up your LLM, including how to get an API key.

---

## Setting up your LLM client

To use AI-powered features, you need to create an LLM client and supply your OpenAI API key ([how to get one below](#how-do-i-get-an-openai-api-key)). You do *not* need an LLM client to simply load report data (i.e. using `load_reports`).


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
online_med_reports = screener.screen_reports(search_query=query)
```

---

## How do I get an OpenAI API key?

1. Sign up or log in at [OpenAI Platform](https://platform.openai.com).
2. Go to [API Keys](https://platform.openai.com/api-keys).
3. Click “Create new secret key” and copy the string.
4. Store your key somewhere safe. **Never** share or publish it.
5. Add credit to your account (just $5 is enough for most research uses).

For more information about usage costs, see [OpenAI pricing](https://openai.com/api/pricing/).

!!! warning
    OpenAI currently mandate that your account must be at least than 48 hours old before being able to run LLMs as normal. If you're setting up your account for this first time, you might have to wait a couple of days before using PFD Toolkit's advanced features.

---

## Hide your API key

It's important that you never share your API key. This includes making sure you don't commit any code containing your API to GitHub. 

The below code shows you how to stealthily import your API key without ever printing it in your code or console:

```python
from pfd_toolkit import LLM
from dotenv import load_dotenv
import os

# Load OpenAI API key from file
load_dotenv("api.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialise LLM client
llm_client = LLM(api_key=openai_api_key)
```

This code assumes you've got a file called `api.env` in the same directory as your script. `api.env` should look like this:

```
OPENAI_API_KEY = [...]
```

where `[...]` is your API key.

Finally, create a file called `.gitignore` at the front of your directory. Inside it, simply provide the directory of your `api.env` file:

```
path/to/your/api.env/file
```

This tells GitHub not to commit your `api.env` file, keeping it protected.

For more information on `.gitignore`, see [here](https://www.w3schools.com/git/git_ignore.asp).

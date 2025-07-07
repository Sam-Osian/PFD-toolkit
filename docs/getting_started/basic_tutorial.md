---
title: "Tutorial: Detention under the Mental Health Act"
description: |
  Step-by-step example showing how to load data, screen cases and analyse themes.
---

# Tutorial: Detention under the Mental Health Act

This page talks you through an example workflow using PFD Toolkit. Here, we will load a dataset and screen for relevant cases related to "detention under the Mental Health Act" (often referring to as 'being sectioned'). 

We will also discover themes to understand more about the issues coroners keep raising.

This is just an example. PFD reports contain a breadth of information across a whole range of topics and domains. Through this workflow, we hope to give you a sense of how the toolkit can be used, and how it might support your own project.


---

## Load your first dataset

First, you'll need to load a PFD dataset. As this is just an example workflow, we'll only load reports between January 2024 and May 2025.

```py
from pfd_toolkit import load_reports

# Load all PFD reports from January 2024 to May 2025
reports = load_reports(
    start_date="2024-01-01",
    end_date="2025-05-01")

reports.head(n=5)
```


| url                        | date       | coroner    | area                        | receiver                | investigation           | circumstances                 | concerns                   |
|----------------------------|------------|------------|-----------------------------|-------------------------|-------------------------|-------------------------------|----------------------------|
| [...]            | 2025-05-01 | A. Hodson  | Birmingham and...    | NHS England; The Rob... | On 9th December 2024... | At 10.45am on 23rd November...| To The Robert Jones... |
| [...]           | 2025-04-30 | J. Andrews | West Sussex, Br...| West Sussex C... | On 2 November 2024 I... | They drove their car into...   | The inquest was told t...  |
| [...]            | 2025-04-30 | A. Mutch   | Manchester Sou...            | Fluxton Road Medical... | On 1 October 2024 I...  | They were prescribed long...   | The inquest heard evide... |
| [...]            | 2025-04-25 | J. Heath   | North Yorkshire...   | Townhead Surgery        | On 4th June 2024 I...   | On 15 March 2024, Richar...    | When a referral docume...  |
| [...]            | 2025-04-25 | M. Hassell | Inner North Lo...          | The President Royal...  | On 23 August 2024, on...| They were a big baby and...    | With the benefit of a m... |


---


## Set up an LLM client

Before exploring some of the other features of PFD toolkit, we first need to set up an LLM client. This is the AI engine that powers all other features of the toolkit.

You'll need to head to [platform.openai.com](https://platform.openai.com/docs/overview) and create an API key. Once you've got this, simply feed it to the `LLM`.

```python
from pfd_toolkit import LLM

# Set up LLM client
llm_client = LLM(api_key=YOUR-API-KEY) # Replace with actual API key
```

!!! note
    For a more detailed guide on using LLMs in this toolkit, see [Setting up an LLM](../llm_setup.md).

---

## Screen for relevant reports

You're likely using PFD Toolkit because you want to answer a specific question. In our example, we're asking: "Do any PFD reports raise concerns related to detention under the Mental Health Act?"

PFD Toolkit lets you query reports in plain English — no need to know precise keywords or categories. Just describe the cases you care about, and the toolkit will return matching reports.


```python
from pfd_toolkit import Screener

# Create a search query to screen/filter reports by
search_query = "Concerns about detention under the Mental Health Act **only**"

# Set up & run our Screener
screener = Screener(llm = llm_client, # LLM client you set up above
                        reports = reports) # Reports that you loaded earlier

filtered_reports = screener.screen_reports(
    search_query=search_query)

# Optionally, count number of identified reports
len(filtered_reports)
```

```sh
>> 51
```

`filtered_reports` returns a filtered version of our original PFD `DataFrame`, containing the 51 reports the LLM believed matches our query. 


!!! note
    For more information on filtering reports, see [Filter reports with a query](../screener/index.md).

---

## Discover recurring themes

Now that we've loaded and screened our reports for relevance to being _detained under the Mental Health Act_, our next step is to discover recurring themes. In other words, concerns that coroners keep raising.

---

### Set up the Extractor

Before we get the model to generate a list of themes for us, we first need to set up our `Extractor`. This class dictates how the model interacts with your filtered list of PFD reports.

Each `include_*` flag controls whether a specific section of the report are sent to the LLM for analysis. 

For example, if we were only interested in patterns related to coroner's _concerns_, we would set the `include_concerns` flag to `True`:


```python
from pfd_toolkit import Extractor

extractor = Extractor(
    llm=llm_client,             # The same client you created earlier
    reports=filtered_reports,   # Your screened reports

    include_date=False,
    include_coroner=False,
    include_area=False,
    include_receiver=False,
    include_investigation=False,
    include_circumstances=False,
    include_concerns=True       # <--- Only supply the 'concerns' text
)
```

!!! note
    The main reason why we're hiding all reports sections other than the coroners' concerns is to help keep the LLM's instructions short & focused. LLMs often perform better when they are given only relevant information.

    Your own research question might be different. For example, you might be interested in discovering recurring themes related to 'cause of death', in which case you'll likely want to set `include_investigation` and `include_circumstances` to `True`.
    
    To understand more about what information is contained within each of the report sections, please see [About the data](../pfd_reports.md#what-do-pfd-reports-look-like).

---

### Summarise reports

Some PFD reports can be _long_. Because of this, we need to summarise reports *before* we discover themes:


```python
# Create short summaries of the concerns
extractor.summarise(trim_intensity="medium")
```

---

### Get a list of themes

Now that we've done this, we can run the `discover_themes` method and assign the result to a new class, which we've named `ThemeInstructions`:

```python
# Ask the LLM to propose recurring themes
ThemeInstructions = extractor.discover_themes(
    max_themes=6,  # Limit the list to keep things manageable
)
```

!!! note
    `discover_themes()` will warn you if the word count of your summaries is still too high. In these cases, you might want to set your `trim_intensity` to `high` or `very high` (though please note that the more we trim, the more detail we lose).



To print our list of themes, run:

```python
print(extractor.identified_themes)
```

...which gives us:

```json
{
  "bed_shortage": "Insufficient availability of inpatient mental health beds or suitable placements, leading to delays, inappropriate care environments, or patients being placed far from home.",

  "staff_training": "Inadequate staff training, knowledge, or awareness regarding policies, risk assessment, clinical procedures, or the Mental Health Act.",

  "record_keeping": "Poor, inconsistent, or falsified documentation and record keeping, including failures in care planning, observation records, and communication of key information.",

  "policy_gap": "Absence, inconsistency, or lack of clarity in policies, protocols, or guidance, resulting in confusion or unsafe practices.",

  "communication_failures": "Breakdowns in communication or information sharing between staff, agencies, families, or across systems, impacting patient safety and care continuity.",

  "risk_assessment": "Failures or omissions in risk assessment, escalation, or monitoring, including inadequate recognition of suicide risk, self-harm, or other patient safety concerns."
}
```

---

### Tag the reports with our themes

Above, we've only _identified_ a list of themes: we haven't yet assigned these themes to each of our reports.

Here, we take `ThemeInstructions` that we created earlier and pass it back into the extractor to assign themes to reports via `extract_features()`:

```python
labelled_reports = extractor.extract_features(
    feature_model=ThemeInstructions,
    force_assign=True,  # (Force the model to make a decision)
    allow_multiple=True  # (A single report might touch on several themes)
)

labelled_reports.head()
```

The resulting `DataFrame` now contains our existing columns along with a suite of new ones: each filled with either `True` or `False`, depending on whether the theme was present.

| url | id | date | coroner | area | receiver | investigation | circumstances | concerns | bed_shortage | staff_training | record_keeping | policy_gap | communication_failures | risk_assessment |
|-----|----|------|---------|------|----------|---------------|---------------|----------|--------------|----------------|----------------|------------|------------------------|-----------------|
| […] | 2025-0172 | 2025-04-07 | S. Reeves | South London | South London and Maudsley NHS … | On 21 March 2023 an inquest … | Christopher McDonald was … | The evidence heard … | False | True | False | False | False | True |
| […] | 2025-0144 | 2025-03-17 | S. Horstead | Essex | Chief Executive Officer of Essex … | On 31 October 2023 I … | On the 23rd September 2023 … | (a) Failures in care … | False | False | True | False | True | True |
| […] | 2025-0104 | 2025-03-13 | A. Harris | South London | Oxleas NHS Foundation Trust; … | On 15 January 2020 an … | Mr Paul Dunne had a … | Individual mental health … | False | True | True | True | True | True |
| […] | 2025-0124 | 2025-03-06 | D. Henry | Coventry | Chair of the Coventry and … | On 13 August 2021 I … | Mr Gebrsselasié on the 2nd … | The inquest explored issues … | False | False | False | True | False | True |
| […] | 2025-0119 | 2025-03-04 | L. Hunt | Birmingham and Solihull | Birmingham and Solihull Mental … | On 20 July 2023 I … | Mr Lynch resided in room 1 … | To Birmingham and Solihull … | False | True | True | True | True | True |

---

### Tabulate reports

Finally, we can count how often a theme appears in our collection of reports:


```python
extractor.tabulate()
```


| Category              | Count | Percentage |
|-----------------------|-------|------------|
| bed_shortage          | 14    | 27.5       |
| staff_training        | 22    | 43.1       |
| record_keeping        | 13    | 25.5       |
| policy_gap            | 35    | 68.6       |
| communication_failures| 19    | 37.3       |
| risk_assessment       | 34    | 66.7       |


That's it! You've gone from a mass of PFD reports, to a focused set of cases relating to being detained under the Mental Health Act to a theme‑tagged dataset ready for deeper exploration.

From here, you might want to export your curated dataset to a .csv for any final qualitative/manual analysis:

```python
labelled_reports.to_csv()
```

Alternatively, you might want to check out the other analytical features that PFD Toolkit offers.

For a deeper dive into each capability, head over to the [Explore](../loader/load_reports.md) section.
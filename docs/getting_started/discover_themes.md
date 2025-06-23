# Discover themes in your filtered dataset

With our subset of reports screened for Mental Health Act detention concerns (see previous page), we will now uncover underlying themes contained within these reports. This lets you see 'at a glance' what issues the coroners keep raising.

This page shows you how to use the `Extractor` class of `pfd_toolkit` to automatically identify themes from the reports. We'll then assign these themes to the reports, and tabulate them to see their prevalence.


---

## Set up the Extractor

The Extractor reads the text from the screened reports you provide. Each `include_*` flag controls whether a specific section of the report are sent to the LLM for analysis. 

In this example, we are only interested in the coroner's *concerns*, so we set `include_concerns` to `True`, while everything else is set to `False`:

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

## Summarise reports & discover themes

Before discovering themes, we first need to summarise each report. 

We do this because the length of PFD reports varies from coroner to coroner. By summarising the reports, we're centering on the key messages, keeping the prompt short for the LLM. This may improve performance and increase speed.

The report sections that are summarised depend on the `include_*` flags you set earlier. In this tutorial, we are only summarising the *concerns* section.


```python
# Create short summaries of the concerns
extractor.summarise(trim_intensity="medium")
```

Now that we've done this, we can run the `discover_themes` method and assign the result to a new class, which we've named `ThemeInstructions`:

```python
# Ask the LLM to propose recurring themes
ThemeInstructions = extractor.discover_themes(
    max_themes=6,  # Limit the list to keep things manageable
)
```

!!! note
    `Extractor` will warn you if the word count of your summaries is too high. In these cases, you might want to set your `trim_intensity` to `high` or `very high` (though please note that the more we trim, the more detail we lose).


`ThemeInstructions` is a Pydantic model containing a set of detailed instructions for the LLM. We use this to assign each report with our list of themes.

But first, you'll likely want to see which themes the model has identified. To see the list of themes (plus a short, automatically generated description for each) we can run:

```python
print(extractor.identified_themes)
```

...which gives us a JSON with our themes & descriptions for each:

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

## Tag the reports with our themes

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


## Tabulate reports

Finally, we can count how often a theme appears in our collection of reports:


```python
extractor.tabulate()
```

```
| Category              | Count | Percentage |
|-----------------------|-------|------------|
| bed_shortage          | 14    | 27.5       |
| staff_training        | 22    | 43.1       |
| record_keeping        | 13    | 25.5       |
| policy_gap            | 35    | 68.6       |
| communication_failures| 19    | 37.3       |
| risk_assessment       | 34    | 66.7       |

```
That's it! You've gone from a mass of PFD reports, to a focused set of cases relating to Mental Health Act detention, to a theme‑tagged dataset ready for deeper exploration.

From here, you might want to export your curated dataset to a .csv for qualitative analysis:

```python
labelled_reports.to_csv()
```

Alternatively, you might want to check out the other analytical features that PFD Toolkit offers.
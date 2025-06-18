# Discover themes in your filtered dataset

With your subset of reports screened for Mental Health Act detention concerns, the next step is to uncover the underlying themes. This lets you see 'at a glance' what issues the coroners keep raising.

We'll use the `Extractor` class to automatically identify themes from the *concerns* section of each report.

---

## Set up the Extractor

The Extractor reads the text from the reports you provide. Each `include_*` flag controls which report columns are sent to the LLM. Here we are only interested in the coroner's concerns, so we turn everything else off:

```python
from pfd_toolkit import Extractor

extractor = Extractor(
    llm=llm_client,             # The same client you created earlier
    reports=filtered_reports,   # Your screened DataFrame
    include_date=False,
    include_coroner=False,
    include_area=False,
    include_receiver=False,
    include_investigation=False,
    include_circumstances=False,
    include_concerns=True       # Only supply the 'concerns' text
)
```

Keeping the prompt focused on the coroner's concerns reduces cost and may result in more cohesive themes.

---

## Summarise then discover themes

Before discovering themes, we need to summarise each report. 

We do this because the length of PFD report varies from coroner-to-coroner. By summarising the reports, we're centering on the key messages, keeping the prompt short for the LLM.
    


```python
# Create short summaries of the concerns
extractor.summarise(trim_intensity="medium")

# Ask the LLM to propose recurring themes
IdentifiedThemes = extractor.discover_themes(
    max_themes=6,  # Limit the list to keep things manageable
)
```

!!! note
    `Extractor` will warn you if the word count of your summaries is too high. In these cases, you might want to set your `trim_intensity` to `high` or `very high` (though please note that the more we trim, the more detail we lose).


`IdentifiedThemes` is a Pydantic model containing our list of themes.

It is not printable in itself, but it is internally replicated as a JSON which we can print:

```python
print(extractor.identified_themes)
```

This gives us a record of each proposed theme with an accompanying description:

```json
{
  "bed_shortage": {
    "type": "bool",
    "description": "Shortage of inpatient mental health beds causing prolonged waits, inappropriate placements, and increased risks."
  },
  "risk_assessment": {
    "type": "bool",
    "description": "Failures or inadequacies in assessing, documenting, and managing patient risks including suicide, self-harm, and violence."
  },
  "communication_failures": {
    "type": "bool",
    "description": "Breakdowns in communication and information sharing between healthcare staff, agencies, families, and police."
  },
  "staff_training": {
    "type": "bool",
    "description": "Insufficient or inconsistent training of staff on policies, clinical knowledge, risk management, and emergency procedures."
  },
  "policy_implementation": {
    "type": "bool",
    "description": "Lack of or poor adherence to policies, protocols, and guidance leading to unsafe practices and delays."
  },
  "observation_monitoring": {
    "type": "bool",
    "description": "Failures in patient observation practices, including inadequate monitoring, falsification of records, and unclear procedures."
  }
}
```

---

## Tag the reports

Above, we've only identified the themes: we haven't assigned these themes to each of our the reports.

Once you have the theme model, pass it back into the extractor to assign themes to every report in the dataset:

```python
labelled_reports = extractor.extract_features(
    feature_model=IdentifiedThemes,
    force_assign=True,
    allow_multiple=True  # A single report might touch on several themes
)
```

The resulting DataFrame now contains a column for each discovered theme, filled with `True` or `False` depending on whether that theme was present in the coroner's concerns.

Finally, we can count how often a theme appears in our collection of reports:


```python
from pfd_toolkit import _tabulate

_tabulate(labelled_reports, columns=[
    "bed_shortage",
    "risk_assessment",
    "communication_failures",
    "staff_training",
    "policy_implementation",
    "observation_monitoring"])
```

```
| Category              | Count | Percentage |
|-----------------------|-------|------------|
| risk_assessment       | 69    | 70.41      |
| information_sharing   | 50    | 51.02      |
| bed_shortage          | 18    | 18.37      |
| staff_training        | 49    | 50.00      |
| policy_compliance     | 58    | 59.18      |
| environmental_safety  | 17    | 17.35      |
```

That's it! You've gone from a mass of PFD reports, to a focused set of cases relating to Mental Health Act detention, to a theme‑tagged dataset ready for deeper exploration.

From here we can either save our `labelled_reports` dataset via `pandas` for qualitative analysis, or we can use *even more* analytical features of PFD Toolkit.

```python
labelled_reports.to_csv()
```

!!! note
    On our machine, the entire workflow contained within this page and [Load & screen reports](load_and_screen.md) took just 1 minute and 5 seconds. Adjust the `max_workers` parameter in the `LLM` class to control concurrency, but note that higher values could result in rate limit errors.

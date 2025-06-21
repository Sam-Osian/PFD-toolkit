# Discover themes in your filtered dataset

With our subset of reports screened for Mental Health Act detention concerns (see previous page), we will now uncover underlying themes contained within these reports. This lets you see 'at a glance' what issues the coroners keep raising.

We'll use the `Extractor` class to automatically identify themes from the *concerns* section of each report.

---

## Set up the Extractor

The Extractor reads the text from the reports you provide. Each `include_*` flag controls which sections of the reports are sent to the LLM for analysis. In this example, we are only interested in the coroner's concerns, so we turn everything else off:

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
    include_concerns=True       # <--- Only supply the 'concerns' text
)
```

!!! note
    The main reason why we're turning 'off' all reports sections other than the coroners' concerns is to help keep the LLM's instructions short & focused. LLMs often perform better when they are only given relevant information.

    The sections you'll want to draw from will depend on your specific research question. To understand more about what information is contained within each of the report sections, please see: [About the data](../pfd_reports.md#what-do-pfd-reports-look-like).


---

## Summarise then discover themes

Before discovering themes, we first need to summarise each report. 

We do this because the length of PFD reports vary from coroner to coroner. By summarising the reports, we're centering on the key messages, keeping the prompt short for the LLM. This improves performance and increases speed.

The report sections that are summarised depend on the `include_*` flags you set earlier. So, in our example, we are only summarising the *concerns* section.



```python
# Create short summaries of the concerns
extractor.summarise(trim_intensity="low")

# Ask the LLM to propose recurring themes
ThemeInstructions = extractor.discover_themes(
    max_themes=6,  # Limit the list to keep things manageable
)
```

!!! note
    `Extractor` will warn you if the word count of your summaries is too high. In these cases, you might want to set your `trim_intensity` to `medium`, `high` or `very high` (though please note that the more we trim, the more detail we lose).


`ThemeInstructions` is a Pydantic model containing a set of detailed instructions for the LLM. We'll need this later to categorise each of the reports by theme.

But first, you'll likely want to see which themes the model has identified. To see the list of themes (plus a short, automatically generated description for each) we can run:

```python
print(extractor.identified_themes)
```

...which gives us a JSON with our themes & descriptions:

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

## Tag the reports with our themes

Above, we've only _identified_ a list of themes: we haven't yet assigned these themes to each of our reports.

Here, we take `ThemeInstructions` that we created earlier and pass it back into the extractor to assign themes to every report in the dataset:

```python
labelled_reports = extractor.extract_features(
    feature_model=ThemeInstructions,
    force_assign=True,
    allow_multiple=True  # (A single report might touch on several themes)
)

labelled_reports.head()
```

The resulting DataFrame now contains a column for each discovered theme, filled with `True` or `False` depending on whether that theme was present in the coroner's concerns:

| url    | date       | coroner    | area              | receiver                   | investigation            | circumstances           | concerns               | bed_shortage | risk_assessment | communication_failures | staff_training | policy_implementation | observation_monitoring |
|--------|------------|------------|-------------------|----------------------------|--------------------------|-------------------------|------------------------|--------------|-----------------|-----------------------|----------------|-----------------------|-----------------------|
| [...]  | 2025-04-24 | S. Marsh   | Somerset          | Somerset Foundation Trust... | On sixth December...     | Anne first presented... | Anne was not sent...   | False        | True            | True                  | True           | False                | False                |
| [...]  | 2025-04-07 | S. Reeves  | South London      | South London and Maudsley... | On 21 March 2023...      | Christopher McDonald... | The evidence heard...  | False        | False           | False                 | True           | True                 | False                |
| [...]  | 2025-03-25 | F. Wilcox  | Inner West London | Commissioner of the Police... | From third March...      | Mr Omishore had been... | That there is an...    | False        | True            | True                  | True           | False                | False                |
| [...]  | 2025-03-24 | T. Rawden  | South Yorkshire West | South West Yorkshire Partnership... | On 27 September...     | Claire Louise Driver... | The inquest heard...   | False        | True            | True                  | True           | False                | False                |
| [...]  | 2025-03-17 | S. Horstead| Essex             | Chief Executive Officer...   | On 31 October 2023...    | On the 23rd September... | (a) Failures in care... | False        | True            | True                  | False          | True                 | False                |


## Tabulate reports

Finally, we can count how often a theme appears in our collection of reports:


```python
extractor.tabulate()
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
That's it! You've gone from a mass of PFD reports, to a focused set of cases relating to Mental Health Act detention, to a theme‑tagged dataset ready for deeper exploration - all in a matter of minutes.

From here, you might want to export your curated dataset to a .csv for qualitative analysis:

```python
labelled_reports.to_csv()
```

Alternatively, you might want to check out the other analytical features that PFD Toolkit offers.
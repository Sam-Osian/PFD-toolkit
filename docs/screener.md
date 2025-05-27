# Screen reports for relevancy

Natural-language filtering is one of the headline features of PFD Toolkit. The `Screener` class lets you describe a topic in plain English – e.g. “deaths in police custody”, “medication purchased online”, “deaths in A&E” – and have an `LLM` screen reports, delivering you a curated dataset.

You do not have to use `Screener` to benefit from the toolkit. If the built-in category tags are good enough, the `load_reports()` helper will give you a ready-made dataset without any `LLM` calls. `Screener` is offered for those projects that need something more nuanced than the provided broad category tags.

To use the `Screener` you'll need to [set up an LLM client](llm_setup.md).

---

## A minimal example

First, import the necessary modules, load reports and set up an `LLM` client:

```python
from pfd_toolkit import load_reports, LLM, Screener

# Grab the pre-processed April 2025 dataset
reports = load_reports(category="all",
                       start_date="2025-04-01",
                       end_date="2025-04-30")

# Set up your LLM client (see “Creating an LLM client” for details)
llm_client = LLM(api_key="YOUR-API-KEY")
```

Then set your user query (the reports you're interested in), pass it to `Screener` along with the `LLM` client, and you'll be given a filtered dataset containing reports that match your query.

```python
# Describe the topic you're interested in
user_query = "deaths in police custody"

# Set up Screener
screener = Screener(
    llm=llm_client,
    reports=reports,
    user_query=user_query,
    filter_df=True
)

police_df = screener.screen_reports()

print(f"{len(police_df)} reports matched.")
>> "13 reports matched."
```

---

## Why not just have a normal “search” function?

A keyword search is only as good as the exact words you type. Coroners, however, don't always follow a shared vocabulary. The same idea can surface in wildly different forms:

*  *Under-staffing* might be written as **“staff shortages,” “inadequate nurse cover,”** or even **“resource constraints.”**  
*  *Suicide in prisons* may masquerade as **“self-inflicted injury while remanded,”** **“ligature event in cell,”** or appear across separate sentences — one mentioning suicide, another describing the prison setting — so a simple keyword filter never links the two.

A keyword filter misses these variants unless you guess every synonym in advance. By contrast, an `LLM` understands the context behind your query and links the phrasing for you, which is exactly what `Screener` taps into.


---
## Additional options

### Match leniency
The `match_leniency` flag nudges the `LLM` when it's unsure whether a report meets your query or not.

In "strict" mode (the default), a marginal case is excluded and is not added to your curated list; in "liberal" mode, the benefit of the doubt gears towards inclusion. 

If you are sweeping widely then tightening by hand, start liberal; if you want only high-precision hits for immediate analysis, stay strict.

```py
screener = Screener(
    llm=llm_client,
    reports=reports,
    user_query=user_query,
    match_leniency='liberal'    # <--- or 'strict' (default)
)
```

Under the hood, this is what the `LLM` sees. For "strict" mode (default):

> Your match leniency should be strict. This means that if you are on the fence as to whether a report matches the user query, you should respond "No".

And for "liberal" mode:

> Your match leniency should be liberal. This means that if you are on the fence as to whether a report matches the user query, you should respond "Yes".

---


### Annotation vs. filtering

If `filter_df` is True (the default) `Screener` returns a trimmed DataFrame that contains only the reports the `LLM` marked as relevant to your query. 

Setting it to False activates annotate mode: every report/row from your original DataFrame is kept, and a boolean column is added denoting whether the report met your query or not. You can also rename this column with `result_col_name`. 

A common workflow is to screen once with `filter_df=False`, inspect a few borderline cases, then rerun with `filter_df=True` once you trust the settings.

```py
screener = Screener(
    llm=llm_client,
    reports=reports,
    user_query=user_query,
    filter_df=False,    # <--- create annotation column; don't filter out
    result_col_name='custody_match'     # <--- name of annotation column
)
```

---


### Choosing which columns the LLM 'sees'

By default the `LLM` model reads the narrative heavyweight sections of each report: *investigation*, *circumstances* and *concerns*. You can expose or hide any field with `include_*` flags. 

For example, if you are screening based on a specific *cause of death*, then you should consider setting `include_concerns` to False, as including this won't benefit your search. 

By contrast, if you are searching for a specific concern, then setting `include_investigation` and `include_circumstances` to False may improve accuracy, speed up your code, and lead to cheaper `LLM` calls.

```py
user_query = "Death from insulin overdose due to misprogrammed insulin pumps."

screener = Screener(
    llm=llm_client,
    reports=reports,
    user_query=user_query,
    include_concerns=False    # <--- Our query doesn't need this section
)
```

In another example, let's say we are only interested in reports sent to a *Member of Parliament*. We'll want to turn off all default sections and only read from the receiver column.

```py
user_query = "Whether the report was sent to a Member of Parliament (MP)"

screener = Screener(
    llm=llm_client,
    reports=reports,
    user_query=user_query,

    # Turn off the defaults...
    include_investigation=False,
    include_circumstances=False,
    include_concerns=False,

    include_receiver=True       # <--- Read from receiver section
)
```

#### All options and defaults

<table>
  <thead>
    <tr>
      <th style="width:22%">Flag</th>
      <th>Report section</th>
      <th>What it's useful for</th>
      <th>Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>include_coroner</code></td>
      <td>Coroner’s name</td>
      <td>Simply the name of the coroner. Rarely needed for screening.</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>include_area</code></td>
      <td>Coroner’s area</td>
      <td>Useful for geographic questions, e.g.&nbsp;deaths in South-East England.</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>include_receiver</code></td>
      <td>Receiver(s) of the report</td>
      <td>Great for accountability queries, e.g. reports sent to NHS Wales.</td>
      <td><code>False</code></td>
    </tr>
    <tr>
      <td><code>include_investigation</code></td>
      <td>“Investigation &amp; Inquest” section</td>
      <td>Contains procedural detail about the inquest.</td>
      <td><code>True</code></td>
    </tr>
    <tr>
      <td><code>include_circumstances</code></td>
      <td>“Circumstances of Death” section</td>
      <td>Describes what actually happened; holds key facts about the death.</td>
      <td><code>True</code></td>
    </tr>
    <tr>
      <td><code>include_concerns</code></td>
      <td>“Coroner’s Concerns” section</td>
      <td>Lists the issues the coroner wants addressed — ideal for risk screening.</td>
      <td><code>True</code></td>
    </tr>
  </tbody>
</table>



---

## Tips for writing a good user query


1. **Stick to one core idea.**  Give the `LLM` a single, clear subject: “falls from hospital beds,” “carbon-monoxide poisoning at home.” In general, the shorter the prompt, the less room for misinterpretation.

2. **Avoid nested logic.**  Complex clauses like “suicide *and* medication error *but not* in custody” dilute the signal. Run separate screens (suicide; medication error; in custody) and combine or subtract results later with pandas.

3. **Let the model handle synonyms.**  You don’t need *“defective, faulty, malfunctioning”* all in the same query; *“malfunctioning defibrillators”* is enough.

4. **Use positive phrasing.**  Negations (e.g. “not related to COVID-19”) can flip the model’s reasoning. Screen positively, set `filter_df` to False, then drop rows in pandas.

5. **Keep it readable.**  If your query needs multiple commas or parentheses, break it up. A one-line statement without side notes usually performs best.


Examples:


| Less-effective query                                                                                                                     | Why it struggles                                                                               | Better query               |
|------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------|
| “Deaths where someone slipped or fell in hospital corridors or patient rooms and maybe had fractures but **not** clinics”               | Too long, multiple settings, negative clause                                                   | “Falls on inpatient wards”           |
| “Fires or explosions causing death at home including gas leaks but **not** industrial accidents”                                         | Mixes two ideas (home vs. industrial) plus a negation                                          | “Domestic gas explosions”            |
| “Cases involving children and allergic reactions to nuts during school outings”                                                          | Several concepts (age, allergen, setting)                                                      | “Fatal nut allergy on school trip”   |
| “Railway incidents that resulted in death due to being hit by train while trespassing **or** at crossings”                               | Two scenarios joined by “or”; verbose                                                          | “Trespasser struck by train”         |
| “Patients dying because an ambulance was late **or** there was delay in emergency services arrival **or** they couldn't get one”         | Chain of synonyms and clauses                                                                  | “Death from delayed ambulance”       |
| “Errors in giving anaesthesia, like too much anaesthetic, wrong drug, problems with intubation, **etc.**”                                  | Long list invites confusion; “etc.” is vague                                                   | “Anaesthesia error”                  |

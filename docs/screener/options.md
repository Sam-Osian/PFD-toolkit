# Additional options

### Match leniency
The `match_leniency` flag nudges the LLM when it's unsure whether a report meets your query or not.

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

Under the hood, this is what the LLM sees. For "strict" mode (default):

> Your match leniency should be strict. This means that if you are on the fence as to whether a report matches the user query, you should respond "No".

And for "liberal" mode:

> Your match leniency should be liberal. This means that if you are on the fence as to whether a report matches the user query, you should respond "Yes".

---

### Annotation vs. filtering

If `filter_df` is True (the default) `Screener` returns a trimmed DataFrame that contains only the reports the LLM marked as relevant to your query.

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

By default the LLM model reads the narrative heavyweight sections of each report: *investigation*, *circumstances* and *concerns*. You can expose or hide any field with `include_*` flags.

For example, if you are screening based on a specific *cause of death*, then you should consider setting `include_concerns` to False, as including this won't benefit your search.

By contrast, if you are searching for a specific concern, then setting `include_investigation` and `include_circumstances` to False may improve accuracy, speed up your code, and lead to cheaper LLM calls.

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

---
title: "Additional options"
description: |
  Control annotation, filtering and other Screener behaviour.
---

# Additional options

### Annotation vs. filtering

If `filter_df` is True (the default) `Screener` returns a trimmed DataFrame that contains only the reports the LLM marked as relevant to your query.

Setting it to False activates annotate mode: every report/row from your original DataFrame is kept, and a boolean column is added denoting whether the report met your query or not. You can also rename this column with `result_col_name`.

Annotate mode is useful where you want to add a column denoting whether the report matched your query, but you don't want to lose the non-matching reports from your dataset.


```py
screener = Screener(
    llm=llm_client,
    reports=reports,
)

annotated = screener.screen_reports(
    search_query=search_query,
    filter_df=False,    # <--- create annotation column instead of filtering
    result_col_name='custody_match'     # <--- name of annotation column
)
```

---

### Choosing which columns the LLM 'sees'

By default the LLM model reads the narrative heavyweight sections of each report: *investigation*, *circumstances* and *concerns*. You can expose or hide any field with `include_*` flags.

For example, if you are screening based on a specific *cause of death*, then you should consider setting `include_concerns` to False, as including this won't benefit your search.

By contrast, if you are searching for a specific concern, then setting `include_investigation` and `include_circumstances` to False may improve accuracy, speed up your code, and lead to cheaper LLM calls.

```py
search_query = "Death from insulin overdose due to misprogrammed insulin pumps."

screener = Screener(
    llm=llm_client,
    reports=reports,
    include_concerns=False    # <--- Our query doesn't need this section
)

result = screener.screen_reports(search_query=search_query)
```

In another example, let's say we are only interested in reports sent to a *Member of Parliament*. We'll want to turn off all default sections and only read from the receiver column.

```py
search_query = "Whether the report was sent to a Member of Parliament (MP)"

screener = Screener(
    llm=llm_client,
    reports=reports,

    # Turn off the defaults...
    include_investigation=False,
    include_circumstances=False,
    include_concerns=False,

    include_receiver=True       # <--- Read from receiver section
)

result = screener.screen_reports(search_query=search_query)
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

### Returning text spans

Set `produce_spans=True` when calling `screen_reports()` to capture the exact snippets from the report text that justified whether or not a report was returned as relevant or not. A new column called `spans_matches_topic` will be created containing these verbatim snippets. 

```python
screener = Screener(llm=llm_client, reports=reports)

filtered_reports = screener.screen_reports(
    search_query="Where the cause of death was determined to be suicide", 
    produce_spans=True, 
    drop_spans=False)

```

If you only want to use the spans internally, pass `drop_spans=True` to remove the column from the returned dataset after screening.

!!! note
    Producing but then dropping spans might seem a bit pointless, but it's actually likely a great way of improving performance. The LLM will generate these spans *before* deciding whether a report matches the query, allowing it to judge whether these spans truly capture the search criteria.
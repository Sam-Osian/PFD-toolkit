---
title: "Tips for writing a good search query"
description: |
  Best practices for crafting clear queries when screening reports.
---

# Tips for writing a good search query

1. **Stick to one core idea.** Give the LLM a single, clear subject: “falls from hospital beds,” “carbon-monoxide poisoning at home.” In general, the shorter the prompt, the less room for misinterpretation.
2. **Avoid nested logic.** Complex clauses like “suicide *and* medication error *but not* in custody” dilute the signal. Consider running separate screens (suicide; medication error; in custody) and combine or subtract results later with pandas.
3. **Let the model handle synonyms.** You don’t need “defective, faulty, malfunctioning” all in the same query; “malfunctioning defibrillators” is enough.
4. **Use positive phrasing.** Negations (e.g. “not related to COVID-19”) can flip the model’s reasoning. Screen positively, set [`filter_df`](http://127.0.0.1:8000/pfd-toolkit/screener/options/#annotation-vs-filtering) to False, then drop rows in pandas.
5. **Keep it readable.** If your query needs multiple commas or parentheses, break it up. A one-line statement without side notes usually performs best.
6. **Use limiting words to restrict scope.** Words like “only" can focus the LLM on a specific case, reducing the chance it will infer extra details. For example, “falls from beds \*\*only\*\*” signals to the model not to include corridor or bathroom falls. However, avoid overusing them; too many limiters can make the model miss relevant edge cases.

Examples:

| :material-close: Less-effective query | Why it struggles | :material-check: Better query |
|---|---|---|
| “Deaths where someone slipped or fell in hospital corridors or patient rooms and maybe had fractures but **not** clinics” | Too long, multiple settings, negative clause | “Falls on inpatient wards” |
| “Fires or explosions causing death at home including gas leaks but **not** industrial accidents” | Mixes two ideas (home vs. industrial) plus a negation | “Domestic gas explosions” |
| “Cases involving children and allergic reactions to nuts during school outings” | Several concepts (age, allergen, setting) | “Fatal nut allergy on school trip” |
| “Railway incidents that resulted in death due to being hit by train while trespassing **or** at crossings” | Two scenarios joined by “or”; verbose | “Trespasser struck by train” |
| “Patients dying because an ambulance was late **or** there was delay in emergency services arrival **or** they couldn't get one” | Chain of synonyms and clauses | “Death from delayed ambulance” |
| “Errors in giving anaesthesia, like too much anaesthetic, wrong drug, problems with intubation, **etc.**” | Long list invites confusion; “etc.” is vague | “Anaesthesia error” |

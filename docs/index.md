---
title: PFD Toolkit
description: AI toolkit for analysing Prevention of Future Deaths (PFD) reports
image: assets/header.png
---

# Home

![PFD Toolkit: Open-source software for Prevention of Future Deaths reports](assets/header.png)

**PFD Toolkit** is an open-source Python package that lets you automatically extract, structure, and analyse Prevention of Future Deaths (PFD) reports from coroners in England and Wales — turning an unwieldy archive into a clean, research-ready dataset in minutes.

PFD reports are vital public warnings, but until now, they’ve been a nightmare to analyse: inconsistent formats, missing metadata, no way to mass download report content into a neat, tabular dataset – among many other issues.

PFD Toolkit solves this by using AI (including LLMs and Vision models) to read, clean, and standardise every report, whether typed or scanned.

What does this mean in practice? Instead of weeks of manual coding, you get instant access to:

- **Bulk-download structured datasets** of PFD reports
- **Screen and filter** reports by any research question (e.g. “road safety”, “suicide”, or any other query)
- **Generate concise summaries** and extract key variables, themes, or concerns
- **Automatically spot patterns** and recurring issues across reports
- **Output ready-to-use tables** for charts, analysis, or policy work



---

Here’s a sample of the PFD dataset you can load:

| url                        | date       | coroner    | area                        | receiver                | investigation           | circumstances                 | concerns                   |
|----------------------------|------------|------------|-----------------------------|-------------------------|-------------------------|-------------------------------|----------------------------|
| [...]            | 2025-05-01 | A. Hodson  | Birmingham and...    | NHS England; The Rob... | On 9th December 2024... | At 10.45am on 23rd November...| To The Robert Jones... |
| [...]           | 2025-04-30 | J. Andrews | West Sussex, Br...| West Sussex C... | On 2 November 2024 I... | They drove their car into...   | The inquest was told t...  |
| [...]            | 2025-04-30 | A. Mutch   | Manchester Sou...            | Fluxton Road Medical... | On 1 October 2024 I...  | They were prescribed long...   | The inquest heard evide... |
| [...]            | 2025-04-25 | J. Heath   | North Yorkshire...   | Townhead Surgery        | On 4th June 2024 I...   | On 15 March 2024, Richar...    | When a referral docume...  |
| [...]            | 2025-04-25 | M. Hassell | Inner North Lo...          | The President Royal...  | On 23 August 2024, on...| They were a big baby and...    | With the benefit of a m... |


Each row is an individual report, while each column reflects a section of the report. For more information on each of these columns, see [here](pfd_reports.md#what-do-pfd-reports-look-like).

---

## Why does this matter? 

Despite being public warnings, PFD reports are chronically underused. That’s because they’ve historically been a pain to work with: scattered formats, inconsistent categorisation, and a lack of easy access for researchers. As a result, it’s been almost impossible to spot trends or respond to risks quickly.

**PFD Toolkit changes this.** By automating the messy admin typically involved in a PFD research project, it transforms unstructured coroners’ text into neat datasets — making it finally practical to do timely systematic research, policy analysis, or audit work on preventable deaths.


---

## Installation

You can install PFD Toolkit using pip:

```bash
pip install pfd_toolkit
```

To update, run:

```bash
pip install -U pfd_toolkit

```

---

## Licence

This project is distributed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/Sam-Osian/PFD-toolkit?tab=AGPL-3.0-1-ov-file).


!!! note
    * You are welcome to use, modify, and share this code under the terms of the AGPL-3.0.
    * If you use this code to provide a networked service, you are required to make the complete source code available to users of that service.
    * Some project dependencies may have their own licence terms, which could affect certain types of use (e.g. commercial use).

---

## Contribute

PFD Toolkit is designed as a research-enabling tool, and we’re keen to work with the community to make sure it genuinely meets your needs. If you have feedback, ideas, or want to get involved, head to our [Feedback & contributions](contribute.md) page.


---

## How to cite

If you use PFD Toolkit in your research, please cite the archived release:

> Osian, S., & Pytches, J. (2025). PFD Toolkit: Unlocking Prevention of Future Death Reports for Research (Version 0.3.7) [Software]. Zenodo. https://doi.org/10.5281/zenodo.15729717

Or, in BibTeX:

```bibtex
@software{osian2025pfdtoolkit,
  author       = {Sam Osian and Jonathan Pytches},
  title        = {PFD Toolkit: Unlocking Prevention of Future Death Reports for Research},
  year         = {2025},
  version      = {0.3.7},
  doi          = {10.5281/zenodo.15729717},
  url          = {https://github.com/sam-osian/PFD-toolkit}
}
```

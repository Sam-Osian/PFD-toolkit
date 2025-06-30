---
title: "PFD Toolkit – Turn PFD reports into structured insights"
description: |
  A Python Package for Accessing and Analysing Prevention of Future Death (PFD) Reports
---

![PFD Toolkit: Open-source software for Prevention of Future Death reports](assets/header.png)

*PFD Toolkit* is an open-source Python package for bulk extraction and analysis of Prevention of Future Death (PFD) reports from coroners in England and Wales. Designed for researchers, policymakers, and analysts, it transforms manual, fragmented report analysis into an automated workflow.

What used to take months (or even years!) of manual work can now be reduced to a matter of minutes. PFD Toolkit lets you:

1. Download structured PFD report datasets
2. Filter reports to identify research-relevant cases (just type in a phrase such as "road safety")
3. Generate short summaries of reports
4. Automatically discover recurring topics or themes
5. Output clean tables ready for charts or further analysis

---

Here is a sample of the PFD dataset:

| url                        | date       | coroner    | area                        | receiver                | investigation           | circumstances                 | concerns                   |
|----------------------------|------------|------------|-----------------------------|-------------------------|-------------------------|-------------------------------|----------------------------|
| [...]            | 2025-05-01 | A. Hodson  | Birmingham and...    | NHS England; The Rob... | On 9th December 2024... | At 10.45am on 23rd November...| To The Robert Jones... |
| [...]           | 2025-04-30 | J. Andrews | West Sussex, Br...| West Sussex C... | On 2 November 2024 I... | They drove their car into...   | The inquest was told t...  |
| [...]            | 2025-04-30 | A. Mutch   | Manchester Sou...            | Fluxton Road Medical... | On 1 October 2024 I...  | They were prescribed long...   | The inquest heard evide... |
| [...]            | 2025-04-25 | J. Heath   | North Yorkshire...   | Townhead Surgery        | On 4th June 2024 I...   | On 15 March 2024, Richar...    | When a referral docume...  |
| [...]            | 2025-04-25 | M. Hassell | Inner North Lo...          | The President Royal...  | On 23 August 2024, on...| They were a big baby and...    | With the benefit of a m... |


Each row is an individual report, while each column reflects a section of the report. For more information on the structure of these reports, see [here](pfd_reports.md#what-do-pfd-reports-look-like).

---

## Why use PFD Toolkit for PFD Report Analysis?

PFD reports have long served as urgent public warnings — issued when coroners identified risks that could, if ignored, lead to further deaths. Yet despite being freely available, these reports are chronically underused. 

This is for one simple reason: PFD reports are a _pain_ to analyse. 

Common issues include:

 * No straightforward way to download report content in bulk

 * No reliable way of querying reports to find cases relevant to a specific research question

 * Reports being inconsistent in format (e.g. many reports are low quality digital scans)

 * No system for surfacing recurring issues raised across multiple reports

 * Widespread miscategorisation of reports, creating research limitations


As a result, research involving PFD reports demanded months, or even years, of manual admin. Researchers were forced to sift through hundreds/thousands of reports one-by-one, wrestle with absent metadata, and code themes by hand. 

PFD Toolkit offers a solution to each of these issues, helping researchers load, screen and analyse PFD report data - all in a matter of minutes.

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

> Osian, S., & Pytches, J. (2025). PFD Toolkit: Unlocking Prevention of Future Death Reports for Research (Version 0.3.3) [Software]. Zenodo. https://doi.org/10.5281/zenodo.15729717

Or, in BibTeX:

```bibtex
@software{osian2025pfdtoolkit,
  author       = {Sam Osian and Jonathan Pytches},
  title        = {PFD Toolkit: Unlocking Prevention of Future Death Reports for Research},
  year         = {2025},
  version      = {0.3.3},
  doi          = {10.5281/zenodo.15729717},
  url          = {https://github.com/sam-osian/PFD-toolkit}
}
```

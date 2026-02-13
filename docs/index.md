---
title: PFD Toolkit
description: Analyse Prevention of Future Deaths reports with AI
image: assets/header.png
---

# Start page

## Using the Python package

PFD Toolkit is **also available as a Python package** for using the toolkit programmatically, rather than through the interactive web app.

This route gives coders direct control over the full analysis pipeline, including data loading, screening logic, LLM setup, extraction behaviour, and downstream outputs.

Use this section to install the Python package, run workflows in your own environment, and integrate PFD analysis into reproducible scripts and projects.

---

Just like the web app, you can use the Python API to load a DataFrame of PFD Reports similar to the below:

| url                        | date       | coroner    | area                        | receiver                | investigation           | circumstances                 | concerns                   |
|----------------------------|------------|------------|-----------------------------|-------------------------|-------------------------|-------------------------------|----------------------------|
| [...]            | 2025-05-01 | A. Hodson  | Birmingham and...    | NHS England; The Rob... | On 9th December 2024... | At 10.45am on 23rd November...| To The Robert Jones... |
| [...]           | 2025-04-30 | J. Andrews | West Sussex, Br...| West Sussex C... | On 2 November 2024 I... | They drove their car into...   | The inquest was told t...  |
| [...]            | 2025-04-30 | A. Mutch   | Manchester Sou...            | Fluxton Road Medical... | On 1 October 2024 I...  | They were prescribed long...   | The inquest heard evide... |
| [...]            | 2025-04-25 | J. Heath   | North Yorkshire...   | Townhead Surgery        | On 4th June 2024 I...   | On 15 March 2024, Richar...    | When a referral docume...  |
| [...]            | 2025-04-25 | M. Hassell | Inner North Lo...          | The President Royal...  | On 23 August 2024, on...| They were a big baby and...    | With the benefit of a m... |


Each row is an individual report, while each column reflects a section of the report. For more detail on these columns, see [About the data](pfd_reports.md#what-do-pfd-reports-look-like).

---

## Installation

You can install the Python package with pip:

```bash
pip install pfd_toolkit
```

To update the package, run:

```bash
pip install -U pfd_toolkit

```

---

## Licence

The **PFD Toolkit Python package** is distributed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/Sam-Osian/PFD-toolkit?tab=AGPL-3.0-1-ov-file).


!!! note
    * You are welcome to use, modify, and share the Python package code under the terms of the AGPL-3.0.
    * If you use the package to provide a networked service, you are required to make the complete source code available to users of that service.
    * Some package dependencies may have their own licence terms, which could affect certain types of use (e.g. commercial use).

---

## Contribute

The Python package is designed as a research-enabling tool, and we’re keen to work with the community to make sure it genuinely meets your needs. If you have feedback, ideas, or want to get involved, head to our [Feedback & contributions](contribute.md) page.


---

## How to cite

If you use the PFD Toolkit Python package in your research, please cite the archived release:

> Osian, S., & Pytches, J. (2025). PFD Toolkit: Unlocking Prevention of Future Death Reports for Research (Version 0.4.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.15729717

Or, in BibTeX:

```bibtex
@software{osian2025pfdtoolkit,
  author       = {Sam Osian and Jonathan Pytches},
  title        = {PFD Toolkit: Unlocking Prevention of Future Death Reports for Research},
  year         = {2025},
  version      = {0.4.0},
  doi          = {10.5281/zenodo.15729717},
  url          = {https://github.com/sam-osian/PFD-toolkit}
}
```

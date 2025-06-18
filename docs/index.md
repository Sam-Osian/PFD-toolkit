![PFD Toolkit](assets/header.png)

---

## Background

*PFD Toolkit* is an open-source Python package created to transform how researchers, policymakers, and analysts access and analyse Prevention of Future Death (PFD) reports from coroners in England and Wales.

### The problem

PFD reports have long served as urgent public warnings — issued when coroners identified risks that could, if ignored, lead to further deaths. Yet despite being freely available, these reports are chronically underused. This is for one simple reason: PFD reports are a pain to analyse. 

Common issues include:

 * No straightforward way to download report content in bulk

 * Wildly inconsistent formats, making traditional web scraping unreliable

 * No reliable way of automatically screening/filtering reports based on a custom query

 * No system for surfacing recurring themes, or extracting other key pieces of information

 * Widespread miscategorisation of reports, creating research limitations


As a result, valuable insights often ends up buried beneath months or even years of manual admin. Researchers are forced to sift through thousands of reports, one-by-one, wrestle with absent metadata, and code themes by hand. 


### Our solution

PFD Toolkit acts as a one-stop-shop for extracting, screening and analysing PFD report data.

The package brings together every PFD report (currently just under 6000) and makes them available in a single, downloadable dataset, ready for instant analysis. 

Here is a sample of the PFD dataset:

| url                        | date       | coroner    | area                        | receiver                | investigation           | circumstances                 | concerns                   |
|----------------------------|------------|------------|-----------------------------|-------------------------|-------------------------|-------------------------------|----------------------------|
| [...]            | 2025-05-01 | A. Hodson  | Birmingham and...    | NHS England; The Rob... | On 9th December 2024... | At 10.45am on 23rd November...| To The Robert Jones... |
| [...]           | 2025-04-30 | J. Andrews | West Sussex, Br...| West Sussex C... | On 2 November 2024 I... | They drove their car into...   | The inquest was told t...  |
| [...]            | 2025-04-30 | A. Mutch   | Manchester Sou...            | Fluxton Road Medical... | On 1 October 2024 I...  | They were prescribed long...   | The inquest heard evide... |
| [...]            | 2025-04-25 | J. Heath   | North Yorkshire...   | Townhead Surgery        | On 4th June 2024 I...   | On 15 March 2024, Richar...    | When a referral docume...  |
| [...]            | 2025-04-25 | M. Hassell | Inner North Lo...          | The President Royal...  | On 23 August 2024, on...| They were a big baby and...    | With the benefit of a m... |



---
PFD Toolkit was built to break down every major barrier to PFD report analysis. Out of the box, you can:

1. Load live PFD data in seconds

2. Query and filter reports with natural language

3. Summarise reports to highlight key messages

4. Automatically discover recurring themes

5. Extract other kinds of information, such as age, sex and cause of death


---

## Installation

You can install PFD Toolkit using pip:

```bash
pip install pfd_toolkit
```

---

## Contribute

PFD Toolkit is designed as a research-enabling tool, and we’re keen to work with the community to make sure it genuinely meets your needs. If you have feedback, ideas, or want to get involved, head to our [Feedback & contributions](contribute.md) page.
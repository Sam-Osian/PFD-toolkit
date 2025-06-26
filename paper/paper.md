---
title: 'PFD Toolkit: Unlocking Prevention of Future Death Reports for Research'
tags: 
  - Python
  - Prevention of Future Deaths
  - Coroners' Reports
  - Open Data
  - United Kingdom
  - Public health
  - Public safety
  - Natural Language Processing (NLP)
  - Large Language Models (LLMs)
  - Web scraping
authors:
 - name: Sam Osian
   corresponding: true
   orcid: 0009-0000-7644-6218
   affiliation: "1, 2, 3"
 - name: Jonathan Pytches
   orcid: 0009-0008-3336-7104
   affiliation: 4
affiliations:
 - name: Department of Primary Care and Mental Health, University of Liverpool, United Kingdom
   index: 1
 - name: Mental Health Research for Innovation Centre (M-RIC), Liverpool, United Kingdom
   index: 2
 - name: Civic Health Innovation Labs (CHIL), Liverpool, United Kingdom
   index: 3
 - name: Independent Researcher, United Kingdom
   index: 4
date: 3 June 2025
bibliography: paper.bib
---


# Summary

Prevention of Future Death (PFD) reports are used by coroners in England and Wales 
to alert individuals and organisations to circumstances that may lead to further deaths. 
Each report outlines the context, contributing factors, and concerns that, if addressed, 
could prevent similar tragedies.

Although these documents offer a rare, ground-level view of public safety risks, research 
involving PFD reports has been notoriously challenging. Assembling a usable dataset requires 
extensive manual effort: downloading, screening, and extracting information from thousands of 
inconsistently formatted documents. This has placed effective analysis out of reach for many.

*PFD Toolkit* is a Python package designed to overcome these barriers, automating the 
abundance of manual tasks involved in analysing PFD data. Using large language models (LLMs) 
and Vision-LLMs (supporting both proprietary and open source models), it extracts 
structured information from text and scanned PFD reports. Users can perform natural language 
searches of reports, discover recurring themes, and extract features at scale for rapid analysis.


# Statement of need

Despite repeated calls for reform, PFD reports remain persistently underused in research 
and policy. Governmental and Parliamentary assessments have identified the current system as 
"under-utilised" for public safety, with no existing mechanisms to track recurring issues
[@home_office_hillsborough_2023; @HCJustice2021_CoronerService]. Families of the deceased have 
called PFD reports "nothing more than a paper exercise," highlighting the lack of systematic
tracking of coroner concerns [@IAPDC_2023_pfdimpact].

This underuse is largely driven by a series of practical and technical barriers on the 
[Courts and Tribunals Judiciary website](https://www.judiciary.uk/prevention-of-future-death-reports/), 
where PFD reports are published: users cannot mass-download reports, formats are inconsistent, 
metadata is incomplete, and many reports are low-resolution scans that lack embedded text. 
Around 73% of reports lack thematic labels, and those that exist are inconsistently applied [@zhang_lessons_2023; @anthony_preventable_2023].

Researchers must screen and extract information from potentially thousands of reports individually, 
an effort that may require months or even years of manual labour [@bremner_systematic_2023]. 
One study on opioid deaths manually screened as many as 3,897 reports by hand [@dernie_preventable_2023].

The [*Preventable Deaths Tracker*](https://preventabledeathstracker.net/) is an existing resource 
that provides valuable summary statistics and metadata on PFD reports, offering the only centralised resource of its 
kind. However, it lacks report-text searching, thematic discovery, and custom information extraction. 
A critical gap remains in the infrastructure for supporting scalable analysis and reducing the manual burden of PFD research.

Researchers have called for technology to automate data collection, enhance data quality, and 
surface information from unstructured text [@bremner_systematic_2023; @zhang_lessons_2023].

*PFD Toolkit* addresses each of these longstanding challenges by enabling researchers to load 
report data; automate the discovery of recurring themes; and extract structured information at 
scale. This makes possible – for the first time – timely, comprehensive, and flexible analysis 
across the entire national archive of PFD reports.


# Key features


1. **Rapid data access.** Instantly loads the latest Prevention of Future Deaths data, 
updated weekly.
2. **Three-layer scraping.** Handles HTML, .pdf, and scanned image reports using 
vision-enabled large language models (V-LLMs) for comprehensive data extraction.
3. **Automated text cleaning.** Corrects spelling, grammar, and formatting issues across 
diverse report structures.
4. **Flexible searching.** Allows users to automatically screen and filter thousands of 
reports using natural language queries (e.g. "medication errors", "deaths in police custody").
5. **Topic modelling.** Discovers recurring themes contained within a given selection of 
PFD reports.
6. **Custom feature extraction.** Pulls structured fields and variables from unstructured 
report text.
7. **Fast run-time.** Supports parallel processing of LLM tasks.


# Example usage

The example below showcases key features #1 and #4. With just a few lines of code and
minutes-long runtime, users can automate what would otherwise take months of manual 
effort: downloading and screening thousands of reports into a research-ready dataset.


```python
from pfd_toolkit import load_reports, LLM, Screener

# -- Load reports into a pandas DataFrame --
reports = load_reports(
  start_date="2013-01-01", end_date="2025-07-01")

# -- Set up an LLM client --
llm_client = LLM(api_key=<YOUR_OPENAI_API_KEY>)

# -- Screen/filter reports by a natural language query --
query="""
Reports explicitly mentioning detention 
under the Mental Health Act
"""

screener = Screener(reports=reports, llm=llm_client)

filtered_df = screener.screen_reports(
                          user_query=query)
```


# Availability and documentation

*PFD Toolkit* is available on PyPI (via `pip install pfd_toolkit`). Full source code is available on [GitHub](https://github.com/Sam-Osian/PFD-toolkit) while package documentation is available [online](https://sam-osian.github.io/PFD-toolkit/).

We welcome community contributions, feedback and feature requests. Please see our [contributions page](https://sam-osian.github.io/PFD-toolkit/contribute/) for more.

The package is unit tested with 74% coverage and is actively maintained. 


# References

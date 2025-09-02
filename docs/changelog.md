---
title: "Changelog"
description: |
  Record of notable changes and updates to the PFD Toolkit project.
---

# 📆 Changelog

Welcome to the project changelog. All notable changes to this project will be documented below.


### 0.3.7 - <small>2025-09-02</small>
* In August 2025, the judiciary.uk website made some subtle changes that broke PFD Toolkit's scraper, meaning that we were unable to collect newly published reports. This issue has now been resolved, and all previously missed reports have now been added.

### 0.3.6 - <small>2025-08-03</small>
* Improve reliability and performance of the Scraper and Cleaner modules. 
* The Cleaner module now standardises each report 'area' to one of 77 official jurisdictions (e.g. "Liverpool and the Wirral"), so minor variations and typos are automatically corrected for consistent regional filtering.
* `load_reports()` now refreshes the dataset by default. Pass `refresh=False` to use a previously cached copy instead of downloading again.


### 0.3.5 - <small>2025-07-07</small>
* Fixed issue where PFD Toolkit refused to run in Google Colab

### 0.3.4 - <small>2025-07-07</small>
* Deprecated `user_query` in `Screener` in favour of `search_query`. `user_query` will be removed in a future release.
* Dropping spans in `extract_features()` no longer removes spans added during screening.
* Downgraded pandas from 2.3.0 to 2.2.2
* Fixed text cleaning bug that expanded dates and removed paragraph spacing.
* Added tests covering span removal behaviour.

### 0.3.3 - <small>2025-06-25</small>
* Improved package installation time
* Changed default LLM model from GPT-4.1-mini to GPT-4.1

### 0.3.2 - <small>2025-06-23</small>
* You no longer need to manually update the `pfd_toolkit` package to get access to freshly published reports. Instead, run `load_reports(refresh=True)`.
* Improve robustness of Scraping module in handling missing data between different scraping strategies.
* Fixed typos and improve documentation.

### 0.3.1 - <small>2025-06-19</small>
* Improved reliability of weekly dataset top-ups.


### 0.3.0 - <small>2025-06-18</small>
First public release! ✨


<!-- 
## [0.3.0] – 2025-07-01

=== "✨ Highlights"
    - 🖇️ Refactored API for more modular LLM integration.
    - 🐛 Fixed intermittent crash on empty PFD report uploads.

=== "📝 Details"
    - **Added:** New `produce_spans` flag for detailed span extraction during LLM-powered feature extraction.
    - **Changed:** Unified the feature extraction and theme assignment APIs—breaking change, see migration below.
    - **Fixed:** Empty DataFrame uploads now return a user-friendly error instead of crashing.
    - **Docs:** Improved developer guide for custom extractors.

!!! Important
    **Breaking change in 0.3.0:**  
    The feature extraction API now requires explicit column selection. Old scripts may fail.


???+ note "Migration Guidance"
    Update your function calls from:
    ```python
    extractor.extract_features(reports)
    ```
    to:
    ```python
    extractor.extract_features(reports, include_date=True, include_concerns=True)
    ```
    See the [API reference](api.md) for details.

---

## [0.2.0] – 2025-05-20

- 🧱 Initial LLM feature extraction  
- 📑 Thematic assignment proof-of-concept  
- 🛠️ Improved error handling for malformed reports

---

## [0.1.0] – 2025-04-14

- 🎉 First release: dataset loader, basic extraction, manual theme labelling

--- -->
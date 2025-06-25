# 📆 Changelog

Welcome to the project changelog. All notable changes to this project will be documented below.

---

### 0.3.3 - <small>2025-06-25</small>
* Improve package installation time
* Changed default LLM model from GPT-4.1-mini to GPT-4.1

### 0.3.2 - <small>2025-06-23</small>
* You no longer need to manually update the `pfd_toolkit` package to get access to freshly published reports. Instead, run `load_reports(refresh=True)`.
* Improve robustness of Scraping module in handling missing data between different scraping strategies.
* Fix typos and improve documentation.

### 0.3.1 - <small>2025-06-19</small>
* Improve reliability of weekly dataset top-ups.


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
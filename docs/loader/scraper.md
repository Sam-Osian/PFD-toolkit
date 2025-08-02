---
title: "Scraping module"
description: |
  Download reports directly from the judiciary website with full control.
---

# Scraping module

`Scraper` lets you download PFD reports straight from the judiciary website and control each step of the extraction process. For most projects [`load_reports()`](load_reports.md) is sufficient, but the scraping module gives you full transparency over how reports are gathered and how missing values are filled in. Use it when you need to customise request behaviour, adjust fallback logic or troubleshoot tricky reports.

## Creating a scraper

```python
from pfd_toolkit import Scraper

scraper = Scraper(
    category="suicide",           # judiciary.uk slug or "all"
    llm=llm_client,         # assumes you've already set up your LLM client
    start_date="2024-01-01",
    end_date="2024-12-31",
    scraping_strategy=[1, 2, 3],   # html → pdf → llm
    max_workers=10,
    delay_range=(1, 2),
)
```

Pass in a category slug (or use `"all"`), a date range and any optional settings such as worker count, request delay or timeout. The `scraping_strategy` list defines which stages run and in what order. Each entry refers to the HTML, PDF and LLM steps respectively – set an index to `-1` to skip a step entirely.

!!! note
    For example, setting `scraping_strategy` to `[2, 1, -1]` runs HTML scraping first, .pdf scraping second, and disables Vision-LLM scraping. Setting it to `[2, -1, 1]` runs Vision-LLM scraping first, HTML scraping second, and disables .pdf scraping. 
    
    This latter configuration is exactly what PFD Toolkit uses under the hood to construct the dataset you see when you call `load_reports()`.

### A closer look at the pipeline

1. **HTML scraping** collects data directly from the report landing page. This is the fastest approach and usually recovers most metadata fields (e.g. coroner name, area, receiver) but struggles where the HTML make up of a given report differs, even slightly, from the majority of reports.
2. **.pdf scraping** downloads the report .pdf and extracts text with *PyMuPDF*. This approach also recovers most fields, but will often scrape page numbers, footnotes and other .pdf 'juice'. It will fail where a report uses a non-standard heading (e.g. uses just "Concerns" instead of the more common "Coroner's concerns").
3. **Vision-LLM scraping** is by far the most reliable method, but also the longest. The LLM understands the reports _in context_, meaning it doesn't matter if a report has unusual formatting or different section headings.


The stages cascade automatically — if HTML scraping gathers everything you need, the PDF and LLM steps are skipped. You can reorder or disable steps entirely by tweaking `scraping_strategy`.

### Running a scrape

After initialisation, call `scrape_reports()` to run the full scrape:

```python
df = scraper.scrape_reports()
```

The results are cached on `scraper.reports` as a pandas DataFrame. This cache lets you rerun individual stages without hitting the network again. If more reports are published later you can update the existing DataFrame with `top_up()`:

```python
updated = scraper.top_up(existing_df=df, end_date="2025-01-31", clean=True)
```

`top_up()` only fetches new pages, meaning you avoid repeating work and keep the original ordering intact. When `clean=True` the new and existing rows are passed through `Cleaner.clean_reports()` for optional LLM-powered tidying.

### Applying the LLM fallback separately

Sometimes you may want to review scraped results before running the LLM stage. `run_llm_fallback()` accepts a DataFrame (typically the output of `scrape_reports()` or `top_up()`) and attempts to fill any remaining blanks using your configured language model:

```python
llm_df = scraper.run_llm_fallback(df)
```

### Cleaning scraped data

To tidy up scraped fields using the same language model, see
the dedicated [Cleaner](cleaner.md) page. It explains how to batch correct
scraped text, anonymise personal information and fine‑tune prompts for each
column.

### Threading and polite scraping

`Scraper` uses a thread pool to speed up network requests. The `max_workers` and `delay_range` settings let you tune throughput and avoid overloading the server. The default one–two second delay between requests mirrors human browsing behaviour and greatly reduces the risk of your IP address being flagged.

### Inspecting results

Every scrape writes a timestamp column when `include_time_stamp=True`. This can be useful for auditing your scraping pipeline. 

All fields that could not be extracted are set to missing values, making gaps explicit in the final dataset.


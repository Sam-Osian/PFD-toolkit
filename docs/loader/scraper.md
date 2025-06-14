# Scraping module

`Scraper` lets you download PFD reports straight from the judiciary website and control each step of the extraction process.  For most projects `load_reports()` is enough, but when you need custom behaviour you can run the pipeline yourself.

## Creating a scraper

```python
from pfd_toolkit import Scraper

scraper = Scraper(
    category="suicide",           # judiciary slug or "all"
    start_date="2024-01-01",
    end_date="2024-12-31",
    scraping_strategy=[1, 2, 3],   # html → pdf → llm
    max_workers=10,
    delay_range=(1, 2),
)
```

Provide a category slug (or use `"all"`), a date range and any optional
settings such as worker count, request delay or timeout.  The
`scraping_strategy` list defines which stages run and in what order.

### How it works

1. **HTML scraping** collects metadata from the web page.
2. **PDF fallback** fetches the report PDF and fills missing fields.
3. **LLM fallback** uses an `LLM` client to recover anything still blank.

You can disable or reorder stages by tweaking `scraping_strategy`.

After initialisation, call `scrape_reports()` to run a full scrape.  The
results are cached on `scraper.reports` as a pandas DataFrame.  If you
want to check for newly published reports later on, call `top_up()` with
an updated date range.  `run_llm_fallback()` can be used to apply the
LLM stage after the fact.

See the [API reference](../reference/scraper.md) for every argument and
attribute.

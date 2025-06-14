# Scraping module

`Scraper` lets you download PFD reports straight from the judiciary website and control each step of the extraction process. For most projects [`load_reports()`](load_reports.md) is sufficient, but the scraping module gives you full transparency over how reports are gathered and how missing values are filled in. Use it when you need to customise request behaviour, adjust fallback logic or troubleshoot tricky reports.

## Why run a custom scrape?

The weekly datasets shipped with the package cover the majority of use cases. However there are two scenarios when direct scraping may be is preferable:

- **Rapid updates** – the official dataset lags up to a week behind new publications. Running your own scrape means you can see the newest reports immediately.
- **Custom logic** – while the dataset bundled with the package is a product of Vision-LLM scraping, you may also wish to enable HTML and .pdf scraping.

## Creating a scraper

```python
from pfd_toolkit import Scraper

scraper = Scraper(
    category="suicide",           # judiciary.uk slug or "all"
    start_date="2024-01-01",
    end_date="2024-12-31",
    scraping_strategy=[1, 2, 3],   # html → pdf → llm
    max_workers=10,
    delay_range=(1, 2),
)
```

Pass in a category slug (or use `"all"`), a date range and any optional settings such as worker count, request delay or timeout. The `scraping_strategy` list defines which stages run and in what order. Each entry refers to the HTML, PDF and LLM steps respectively – set an index to `-1` to skip a step entirely.

### A closer look at the pipeline

1. **HTML scraping** collects metadata directly from the web page. This is the fastest approach and usually recovers most fields.
2. **PDF fallback** downloads the report PDF and extracts text with *PyMuPDF*. Missing fields from the HTML stage are filled in here if possible.
3. **LLM fallback** hands off any unresolved blanks to an `LLM` client. This final pass is optional but can be invaluable for tricky reports where automated parsing fails.

The stages cascade automatically—if HTML scraping gathers everything you need, the PDF and LLM steps are skipped. You can reorder or disable steps entirely by tweaking `scraping_strategy`.

### Running a scrape

After initialisation, call `scrape_reports()` to run the full scrape:

```python
df = scraper.scrape_reports()
```

The results are cached on `scraper.reports` as a pandas DataFrame. This cache lets you rerun individual stages without hitting the network again. If more reports are published later you can update the existing DataFrame with `top_up()`:

```python
updated = scraper.top_up(existing_df=df, end_date="2025-01-31")
```

`top_up()` only fetches new pages, meaning you avoid repeating work and keep the original ordering intact.

### Applying the LLM fallback separately

Sometimes you may want to review scraped results before running the LLM stage. `run_llm_fallback()` accepts a DataFrame (typically the output of `scrape_reports()` or `top_up()`) and attempts to fill any remaining blanks using your configured language model:

```python
llm_df = scraper.run_llm_fallback(df)
```

### Threading and polite scraping

`Scraper` uses a thread pool to speed up network requests. The `max_workers` and `delay_range` settings let you tune throughput and avoid overloading the server. The default one–two second delay between requests mirrors human browsing behaviour and greatly reduces the risk of your IP address being flagged.

### Inspecting results

Every scrape writes a timestamp column when `include_time_stamp=True`. This can be useful for auditing or for merging multiple scrapes. All fields that could not be extracted are set to mising values, making gaps explicit in the final dataset. Use standard pandas operations to analyse or filter the DataFrame.

### Caveats

Since scraping relies on the judiciary.uk website, any changes to layout could easily break parsers. The toolkit aims to handle common edge cases, but if you rely on scraping for production work you should keep an eye on logs and be ready to adapt your strategy. Also remember that running the LLM stage incurs API costs if you use a paid provider.

See the [API reference](../reference/scraper.md) for a detailed breakdown of every argument and attribute.

from pfd_toolkit.scraper.scraper import Scraper



def test_assemble_report_respects_include_flags():
    scraper = Scraper(
        category="all",
        start_date="2024-01-01",
        end_date="2024-01-02",
        max_workers=1,
        max_requests=1,
        delay_range=(0, 0),
        scraping_strategy=[1, -1, -1],
        include_receiver=False,
        include_time_stamp=False,
    )
    fields = {
        "id": "1",
        "date": "2024-05-01",
        "receiver": "someone",
        "coroner": "cor",
        "area": "area",
        "investigation": "inv",
        "circumstances": "circ",
        "concerns": "conc",
    }
    report = scraper._assemble_report("http://example.com", fields)
    assert scraper.COL_RECEIVER not in report
    assert report[scraper.COL_URL] == "http://example.com"
    assert report[scraper.COL_ID] == "1"


def test_warn_if_suboptimal_config(monkeypatch):
    from pfd_toolkit.scraper import scraper as scraper_module

    warnings = []
    monkeypatch.setattr(scraper_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    Scraper(
        category="all",
        start_date="2024-01-01",
        end_date="2024-01-02",
        max_workers=1,  # low
        max_requests=2,  # low
        delay_range=(0, 0),
        scraping_strategy=[1, -1, -1],
    )

    joined = "\n".join(warnings).lower()
    assert "only html scraping is enabled" in joined
    assert "max_workers is set to a low value" in joined
    assert "delay_range has been disabled" in joined


def test_parse_scraping_strategy_warns(monkeypatch):
    from pfd_toolkit.scraper import scraper as scraper_module

    warnings = []
    monkeypatch.setattr(scraper_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    Scraper(
        category="all",
        start_date="2024-01-01",
        end_date="2024-01-02",
        max_workers=1,
        max_requests=1,
        delay_range=(0, 0),
        scraping_strategy=[2, 2, -1],
    )

    assert any("unexpected scraping_strategy" in w.lower() for w in warnings)


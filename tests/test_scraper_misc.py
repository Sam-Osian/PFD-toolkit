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


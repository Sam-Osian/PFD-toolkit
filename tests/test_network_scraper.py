from unittest.mock import Mock, patch

from pathlib import Path
from bs4 import BeautifulSoup

from pfd_toolkit.scraper.html_extractor import HtmlExtractor
from pfd_toolkit.scraper.pdf_extractor import PdfExtractor
from pfd_toolkit.scraper.scraper import Scraper
from pfd_toolkit.config import ScraperConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> bytes:
    with open(FIXTURE_DIR / name, "rb") as f:
        return f.read()



def test_fetch_report_page():
    cfg = ScraperConfig()
    extractor = HtmlExtractor(cfg, timeout=1, id_pattern=None, not_found_text="N/A")
    html = _load_fixture("sample_report.html").decode()
    mock_resp = Mock(status_code=200, content=html.encode(), text=html)
    mock_resp.raise_for_status = Mock()
    with patch.object(cfg.session, "get", return_value=mock_resp) as mock_get:
        soup = extractor.fetch_report_page("https://example.com/report")
    assert isinstance(soup, BeautifulSoup)
    assert soup.find("p").text.strip() == "Sample page"
    mock_get.assert_called_once_with("https://example.com/report", timeout=1)


def test_fetch_pdf_bytes():
    cfg = ScraperConfig()
    extractor = PdfExtractor(cfg, timeout=1, not_found_text="N/A")
    page_html = _load_fixture("sample_report.html").decode()
    pdf_bytes = _load_fixture("sample_pdf.pdf")

    def get_side_effect(url, timeout):
        if url.endswith(".pdf"):
            resp = Mock(status_code=200, content=pdf_bytes)
        else:
            resp = Mock(status_code=200, content=page_html.encode())
        resp.raise_for_status = Mock()
        return resp

    with patch.object(cfg.session, "get", side_effect=get_side_effect) as mock_get:
        result = extractor.fetch_pdf_bytes("https://example.com/report")

    assert result == pdf_bytes
    assert mock_get.call_count == 2


def test_get_report_href_values():
    scraper = Scraper(
        category="all",
        start_date="2024-01-01",
        end_date="2024-01-02",
        delay_range=(0, 0),
        max_workers=1,
        max_requests=1,
        scraping_strategy=[-1, 1, -1],
    )
    search_html = (
        "<html><body>"
        "<a class='card__link' href='https://example.com/r1'></a>"
        "<a class='card__link' href='https://example.com/r2'></a>"
        "</body></html>"
    )
    mock_resp = Mock(status_code=200, text=search_html, content=search_html.encode())
    mock_resp.raise_for_status = Mock()
    with patch.object(scraper.cfg.session, "get", return_value=mock_resp):
        links = scraper._get_report_href_values("https://example.com/search")
    assert links == ["https://example.com/r1", "https://example.com/r2"]


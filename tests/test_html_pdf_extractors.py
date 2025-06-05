from bs4 import BeautifulSoup
from pfd_toolkit.scraper.html_extractor import HtmlExtractor, HtmlFieldConfig
from pfd_toolkit.scraper.pdf_extractor import PdfExtractor
from pfd_toolkit.config import ScraperConfig


def make_html_extractor():
    cfg = ScraperConfig()
    return HtmlExtractor(cfg, timeout=1, id_pattern=None, not_found_text="N/A")


def test_extract_html_paragraph():
    extractor = make_html_extractor()
    html = "<p>Ref: 2024-1234</p>"
    soup = BeautifulSoup(html, 'html.parser')
    text = extractor.extract_html_paragraph_text(soup, ['Ref:'])
    assert text == 'Ref: 2024-1234'


def test_extract_html_section():
    extractor = make_html_extractor()
    html = "<strong>SECTION</strong> content here"
    soup = BeautifulSoup(html, 'html.parser')
    text = extractor.extract_html_section_text(soup, ['SECTION'])
    assert 'content here' in text


def test_extract_pdf_section():
    extractor = PdfExtractor(ScraperConfig(), timeout=1, not_found_text='N/A')
    sample = 'start text KEY1 middle KEY2 end'
    result = extractor.extract_pdf_section(sample, ['KEY1'], ['KEY2'])
    assert result.strip() == ' middle '

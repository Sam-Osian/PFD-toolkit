from bs4 import BeautifulSoup
from pfd_toolkit.scraper.html_extractor import HtmlExtractor
from pfd_toolkit.scraper.pdf_extractor import PdfExtractor
from pfd_toolkit.config import ScraperConfig, GeneralConfig


def make_html_extractor():
    cfg = ScraperConfig()
    return HtmlExtractor(cfg, timeout=1, id_pattern=None, not_found_text=GeneralConfig.NOT_FOUND_TEXT)


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
    extractor = PdfExtractor(ScraperConfig(), timeout=1, not_found_text=GeneralConfig.NOT_FOUND_TEXT)
    sample = 'start text KEY1 middle KEY2 end'
    result = extractor.extract_pdf_section(sample, ['KEY1'], ['KEY2'])
    assert result.strip() == 'middle'


def test_extract_fields_from_html_id_and_date():
    cfg = ScraperConfig()
    extractor = HtmlExtractor(cfg, timeout=1, id_pattern=GeneralConfig.ID_PATTERN, not_found_text=GeneralConfig.NOT_FOUND_TEXT)
    html = """
        <p>Ref: 2024-1234</p>
        <p>Date of report: 1 May 2024</p>
    """
    soup = BeautifulSoup(html, 'html.parser')
    fields = {}
    flags = {f.key: True for f in cfg.html_fields}
    extractor.extract_fields_from_html(soup, fields, flags)
    assert fields['id'] == '2024-1234'
    assert fields['date'] == '2024-05-01'


def test_apply_pdf_fallback_updates_fields():
    cfg = ScraperConfig()
    extractor = PdfExtractor(cfg, timeout=1, not_found_text=GeneralConfig.NOT_FOUND_TEXT)
    pdf_text = (
        "INVESTIGATION and INQUEST some investigation text that is quite long and clearly exceeds thirty characters CIRCUMSTANCES "
    )
    fields = {'investigation': GeneralConfig.NOT_FOUND_TEXT}
    flags = {'investigation': True}
    extractor.apply_pdf_fallback(pdf_text, fields, flags)
    assert fields['investigation'].startswith('some investigation text')


def test_get_pdf_link_new_structure():
    cfg = ScraperConfig()
    extractor = PdfExtractor(cfg, timeout=1, not_found_text=GeneralConfig.NOT_FOUND_TEXT)
    html = (
        '<div data-wp-interactive="core/file" class="wp-block-file">'
        '<a href="https://example.com/new.pdf" class="wp-block-file__button">Download</a>'
        '</div>'
    )
    soup = BeautifulSoup(html, 'html.parser')
    assert extractor.get_pdf_link(soup) == 'https://example.com/new.pdf'


def test_get_pdf_link_old_structure():
    cfg = ScraperConfig()
    extractor = PdfExtractor(cfg, timeout=1, not_found_text=GeneralConfig.NOT_FOUND_TEXT)
    html = (
        '<a class="govuk-button" href="https://example.com/old.pdf">Download PDF</a>'
    )
    soup = BeautifulSoup(html, 'html.parser')
    assert extractor.get_pdf_link(soup) == 'https://example.com/old.pdf'

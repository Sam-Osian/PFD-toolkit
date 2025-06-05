import pytest
from pfd_toolkit.config import ScraperConfig


def test_url_template_valid():
    cfg = ScraperConfig()
    tmpl = cfg.url_template('suicide')
    assert '{page}' in tmpl


def test_url_template_invalid():
    cfg = ScraperConfig()
    with pytest.raises(ValueError):
        cfg.url_template('unknown_category')

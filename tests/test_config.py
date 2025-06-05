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


def test_url_template_case_insensitive():
    cfg = ScraperConfig()
    assert cfg.url_template('SuIcIdE') == cfg.CATEGORY_TEMPLATES['suicide']


def test_apply_random_delay_calls_sleep(monkeypatch):
    cfg = ScraperConfig(delay_range=(0.5, 0.5))
    called = {}

    def fake_uniform(low, high):
        return 0.5

    def fake_sleep(secs):
        called['secs'] = secs

    monkeypatch.setattr('random.uniform', fake_uniform)
    monkeypatch.setattr('time.sleep', fake_sleep)
    cfg.apply_random_delay()
    assert called.get('secs') == 0.5


def test_apply_random_delay_disabled(monkeypatch):
    cfg = ScraperConfig(delay_range=(0, 0))

    def fake_sleep(secs):  # should not be called
        raise AssertionError('sleep should not be called')

    monkeypatch.setattr('time.sleep', fake_sleep)
    cfg.apply_random_delay()  # should not raise

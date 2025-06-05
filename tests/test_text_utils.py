import pytest
from pfd_toolkit import text_utils


def test_normalise_apostrophes():
    inp = "O\u2019Connor\u2018s"
    assert text_utils.normalise_apostrophes(inp) == "O'Connor's"


def test_clean_text():
    raw = "  Hello   world\u2019s  "
    assert text_utils.clean_text(raw) == "Hello world's"


def test_normalise_date_valid():
    assert text_utils.normalise_date("1 May 2024") == "2024-05-01"


def test_normalise_date_with_ref():
    out = text_utils.normalise_date("1 May 2024 Ref: 123")
    assert out == "2024-05-01"


def test_normalise_date_invalid():
    assert text_utils.normalise_date("not a date") == "not a date"


def test_process_extracted_field_date():
    text = "Date of report: 1 May 2024"
    result = text_utils.process_extracted_field(
        text,
        ["Date of report:"],
        "N/A",
        is_date=True,
    )
    assert result == "2024-05-01"


def test_process_extracted_field_min_len():
    text = "short"
    result = text_utils.process_extracted_field(
        text,
        [],
        "N/A",
        min_len=10,
    )
    assert result == "N/A"


def test_process_extracted_field_max_len():
    text = "This text is definitely too long"
    result = text_utils.process_extracted_field(
        text,
        [],
        "N/A",
        max_len=10,
    )
    assert result == "N/A"


def test_process_extracted_field_not_found_passthrough():
    result = text_utils.process_extracted_field(
        "N/A",
        [],
        "N/A",
        min_len=1,
    )
    assert result == "N/A"

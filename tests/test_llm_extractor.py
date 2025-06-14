import pandas as pd

from pfd_toolkit.scraper.llm_extractor import run_llm_fallback
from pfd_toolkit.config import GeneralConfig


class DummyLLM:
    def __init__(self):
        self.max_workers = 1
        self.called_with = []

    def _call_llm_fallback(self, pdf_bytes, missing_fields, report_url=None, verbose=False, tqdm_extra_kwargs=None):
        self.called_with.append((pdf_bytes, missing_fields, report_url))
        result = {}
        for key in missing_fields:
            if key == "date":
                result[key] = "1 May 2024"
            else:
                result[key] = f"VAL-{key}"
        return result


class DummyPdfExtractor:
    def __init__(self):
        self.called_urls = []

    def fetch_pdf_bytes(self, url: str):
        self.called_urls.append(url)
        return b"pdf-bytes"


def test_run_llm_fallback_basic():
    df = pd.DataFrame({
        GeneralConfig.COL_URL: ["https://example.com/report1"],
        GeneralConfig.COL_DATE: [GeneralConfig.NOT_FOUND_TEXT],
        GeneralConfig.COL_RECEIVER: [GeneralConfig.NOT_FOUND_TEXT],
    })

    llm = DummyLLM()
    pdf_extractor = DummyPdfExtractor()
    llm_field_config = [
        (True, GeneralConfig.COL_DATE, "date", "prompt-date"),
        (True, GeneralConfig.COL_RECEIVER, "receiver", "prompt-receiver"),
    ]
    llm_to_df_mapping = {"date": GeneralConfig.COL_DATE, "receiver": GeneralConfig.COL_RECEIVER}

    result = run_llm_fallback(
        df.copy(),
        llm=llm,
        pdf_extractor=pdf_extractor,
        llm_field_config=llm_field_config,
        llm_to_df_mapping=llm_to_df_mapping,
        col_url=GeneralConfig.COL_URL,
        not_found_text=GeneralConfig.NOT_FOUND_TEXT,
        llm_key_date="date",
        verbose=False,
    )

    assert pdf_extractor.called_urls == ["https://example.com/report1"]
    assert result[GeneralConfig.COL_DATE].iloc[0] == "2024-05-01"
    assert result[GeneralConfig.COL_RECEIVER].iloc[0] == "VAL-receiver"
    assert llm.called_with[0][1] == {"date": "prompt-date", "receiver": "prompt-receiver"}


def test_run_llm_fallback_empty_df():
    df = pd.DataFrame(columns=[GeneralConfig.COL_URL, GeneralConfig.COL_DATE])
    llm = DummyLLM()
    pdf_extractor = DummyPdfExtractor()

    result = run_llm_fallback(
        df,
        llm=llm,
        pdf_extractor=pdf_extractor,
        llm_field_config=[],
        llm_to_df_mapping={},
        col_url=GeneralConfig.COL_URL,
        not_found_text=GeneralConfig.NOT_FOUND_TEXT,
        llm_key_date="date",
        verbose=False,
    )

    assert result.empty
    assert llm.called_with == []
    assert pdf_extractor.called_urls == []

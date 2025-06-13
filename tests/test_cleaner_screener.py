import pandas as pd
import pytest
from pfd_toolkit.cleaner import Cleaner
from pfd_toolkit.screener import Screener, TopicMatch
from pfd_toolkit.extractor import Extractor
from pfd_toolkit.config import GeneralConfig


class DummyLLM:

    def __init__(self, keywords=None):
        self.keywords = [k.lower() for k in (keywords or [])]

    def generate(self, prompts, response_format=None, **kwargs):
        outputs = []
        for p in prompts:
            text = p.split("\n")[-1]
            if response_format is None:
                outputs.append(text.upper())
            else:
                match = any(kw in text.lower() for kw in self.keywords)
                val = "Yes" if match else "No"
                if hasattr(response_format, "model_fields") and "spans_matches_topic" in response_format.model_fields:
                    span = "span" if match else ""
                    outputs.append(
                        response_format(matches_topic=val, spans_matches_topic=span)
                    )
                else:
                    outputs.append(response_format(matches_topic=val))
        return outputs


def test_cleaner_basic():
    df = pd.DataFrame({"CoronerName": ["john doe"], "Area": ["area"], "Receiver": ["x"],
                       "InvestigationAndInquest": ["inv"], "CircumstancesOfDeath": ["circ"],
                       "MattersOfConcern": ["conc"]})
    cleaner = Cleaner(df, DummyLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned["CoronerName"].iloc[0] == "JOHN DOE"


def test_generate_prompt_template():
    df = pd.DataFrame({
        GeneralConfig.COL_CORONER_NAME: ["john doe"],
        GeneralConfig.COL_AREA: ["area"],
        GeneralConfig.COL_RECEIVER: ["x"],
        GeneralConfig.COL_INVESTIGATION: ["inv"],
        GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
        GeneralConfig.COL_CONCERNS: ["conc"],
    })
    cleaner = Cleaner(df, DummyLLM())
    tmpl = cleaner.generate_prompt_template()
    assert GeneralConfig.COL_CORONER_NAME in tmpl
    assert "[TEXT]" in tmpl[GeneralConfig.COL_CORONER_NAME]


def test_cleaner_missing_column_error():
    df = pd.DataFrame({"CoronerName": ["x"]})
    with pytest.raises(ValueError):
        Cleaner(df, DummyLLM(), include_area=True)


def test_screener_basic():
    data = {
        "InvestigationAndInquest": ["Contains needle text"],
        "CircumstancesOfDeath": ["other"],
        "MattersOfConcern": ["something"]
    }
    df = pd.DataFrame(data)
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df, user_query="needle")
    filtered = screener.screen_reports()
    assert len(filtered) == 1


def test_screener_add_column_no_filter():
    data = {
        "InvestigationAndInquest": ["foo"],
        "CircumstancesOfDeath": ["bar"],
        "MattersOfConcern": ["baz"],
    }
    df = pd.DataFrame(data)
    llm = DummyLLM(keywords=["zzz"])  # no match
    screener = Screener(llm=llm, reports=df, user_query="zzz", filter_df=False)
    result = screener.screen_reports()
    assert screener.result_col_name in result.columns
    assert result[screener.result_col_name].iloc[0] is False


def test_screener_produce_spans():
    df = pd.DataFrame({"InvestigationAndInquest": ["needle info"]})
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df, user_query="needle", filter_df=False)
    result = screener.screen_reports(produce_spans=True)
    assert "spans_matches_topic" in result.columns
    assert result["spans_matches_topic"].iloc[0] == "span"
    assert result[screener.result_col_name].iloc[0] is True


def test_screener_drop_spans():
    df = pd.DataFrame({"InvestigationAndInquest": ["needle info"]})
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df, user_query="needle", filter_df=False)
    result = screener.screen_reports(produce_spans=True, drop_spans=True)
    assert "spans_matches_topic" not in result.columns

def test_cleaner_summarise():
    df = pd.DataFrame({
        GeneralConfig.COL_CORONER_NAME: ["john"],
        GeneralConfig.COL_AREA: ["area"],
        GeneralConfig.COL_RECEIVER: ["x"],
        GeneralConfig.COL_INVESTIGATION: ["inv"],
        GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
        GeneralConfig.COL_CONCERNS: ["conc"],
    })
    extractor = Extractor(llm=DummyLLM(), reports=df)
    out = extractor.summarise()
    assert "summary" in out.columns
    assert len(out) == len(df)

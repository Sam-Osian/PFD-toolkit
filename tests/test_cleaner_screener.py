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
    df = pd.DataFrame({
        GeneralConfig.COL_CORONER_NAME: ["john doe"],
        GeneralConfig.COL_AREA: ["area"],
        GeneralConfig.COL_RECEIVER: ["x"],
        GeneralConfig.COL_INVESTIGATION: ["inv"],
        GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
        GeneralConfig.COL_CONCERNS: ["conc"],
    })
    cleaner = Cleaner(df, DummyLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_CORONER_NAME].iloc[0] == "JOHN DOE"


def test_cleaner_anonymise_prompts():
    df = pd.DataFrame({
        GeneralConfig.COL_CORONER_NAME: ["john"],
        GeneralConfig.COL_AREA: ["area"],
        GeneralConfig.COL_RECEIVER: ["x"],
        GeneralConfig.COL_INVESTIGATION: ["inv"],
        GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
        GeneralConfig.COL_CONCERNS: ["conc"],
    })

    captured = []

    class CaptureLLM(DummyLLM):
        def generate(self, prompts, *args, **kwargs):
            captured.extend(prompts)
            return super().generate(prompts, *args, **kwargs)

    cleaner = Cleaner(df, CaptureLLM())
    cleaner.clean_reports(anonymise=True)

    instruction = "replace all personal names and pronouns with they/them/their"
    for text in ["inv", "circ", "conc"]:
        relevant = [p for p in captured if p.strip().endswith(text)]
        assert relevant, f"no prompt captured for text {text}"
        prompt = relevant[0]
        lines = [line.strip().lower() for line in prompt.splitlines()]
        idx = lines.index("input text:")
        assert lines[idx - 2].startswith(instruction)


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
    df = pd.DataFrame({GeneralConfig.COL_CORONER_NAME: ["x"]})
    with pytest.raises(ValueError):
        Cleaner(df, DummyLLM(), include_area=True)


def test_screener_basic():
    data = {
        GeneralConfig.COL_INVESTIGATION: ["Contains needle text"],
        GeneralConfig.COL_CIRCUMSTANCES: ["other"],
        GeneralConfig.COL_CONCERNS: ["something"],
    }
    df = pd.DataFrame(data)
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df)
    filtered = screener.screen_reports(user_query="needle")
    assert len(filtered) == 1


def test_screener_add_column_no_filter():
    data = {
        GeneralConfig.COL_INVESTIGATION: ["foo"],
        GeneralConfig.COL_CIRCUMSTANCES: ["bar"],
        GeneralConfig.COL_CONCERNS: ["baz"],
    }
    df = pd.DataFrame(data)
    llm = DummyLLM(keywords=["zzz"])  # no match
    screener = Screener(llm=llm, reports=df)
    result = screener.screen_reports(user_query="zzz", filter_df=False, result_col_name="match")
    assert "match" in result.columns
    assert result["match"].iloc[0] is False


def test_screener_produce_spans():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["needle info"]})
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df)
    result = screener.screen_reports(user_query="needle", filter_df=False, produce_spans=True)
    assert "spans_matches_query" in result.columns
    assert result["spans_matches_query"].iloc[0] == "span"
    assert result["matches_query"].iloc[0] is True


def test_screener_drop_spans():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["needle info"]})
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df)
    result = screener.screen_reports(user_query="needle", filter_df=False, produce_spans=True, drop_spans=True)
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

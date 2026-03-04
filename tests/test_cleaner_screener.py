import pandas as pd
import pytest
from pfd_toolkit.cleaner import Cleaner, AreaModel
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
                fields = getattr(response_format, "model_fields", {})
                if "matches_topic" in fields:
                    match = any(kw in text.lower() for kw in self.keywords)
                    val = "Yes" if match else "No"
                    if "spans_matches_topic" in fields:
                        span = "span" if match else ""
                        outputs.append(
                            response_format(matches_topic=val, spans_matches_topic=span)
                        )
                    else:
                        outputs.append(response_format(matches_topic=val))
                else:
                    field_name = next(iter(fields))
                    if field_name == "area":
                        outputs.append(response_format(area="Other"))
                    else:
                        outputs.append(response_format(**{field_name: text.upper()}))
        return outputs


class EchoAreaLLM(DummyLLM):
    def generate(self, prompts, response_format=None, **kwargs):
        outputs = []
        for p in prompts:
            text = p.split("\n")[-1]
            if response_format is None:
                outputs.append(text.upper())
            else:
                field_name = next(iter(response_format.model_fields))
                if field_name == "area":
                    outputs.append(response_format(area=text))
                else:
                    outputs.append(response_format(**{field_name: text.upper()}))
        return outputs


def test_cleaner_basic():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john doe"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )
    cleaner = Cleaner(df, DummyLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_CORONER_NAME].iloc[0] == "JOHN DOE"


def test_cleaner_anonymise_prompts():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

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
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john doe"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )
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
    filtered = screener.screen_reports(search_query="needle")
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
    result = screener.screen_reports(
        search_query="zzz", filter_df=False, result_col_name="match"
    )
    assert "match" in result.columns
    assert result["match"].iloc[0] is False


def test_screener_produce_spans():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["needle info"]})
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df)
    result = screener.screen_reports(
        search_query="needle", filter_df=False, produce_spans=True
    )
    assert "spans_matches_query" in result.columns
    assert result["spans_matches_query"].iloc[0] == "span"
    assert result["matches_query"].iloc[0] is True


def test_screener_drop_spans():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["needle info"]})
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df)
    result = screener.screen_reports(
        search_query="needle", filter_df=False, produce_spans=True, drop_spans=True
    )
    assert "spans_matches_topic" not in result.columns


def test_screener_drop_spans_preserves_existing():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["needle info"],
            "spans_age": ["span"],
            "age": [30],
        }
    )
    llm = DummyLLM(keywords=["needle"])
    screener = Screener(llm=llm, reports=df)
    result = screener.screen_reports(
        search_query="needle",
        filter_df=False,
        produce_spans=True,
        drop_spans=True,
    )

    assert "spans_matches_query" not in result.columns
    assert "spans_age" in result.columns
    assert result["spans_age"].iloc[0] == "span"


def test_area_model_unknown_area_defaults_to_other():
    model = AreaModel(area="Imaginary Shire")
    assert model.area == "Other"


def test_cleaner_unrecognised_area_defaults_to_other():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    class UnknownAreaLLM(DummyLLM):
        def generate(self, prompts, response_format=None, **kwargs):
            outputs = []
            for p in prompts:
                text = p.split("\n")[-1]
                if response_format is None:
                    outputs.append(text.upper())
                else:
                    field_name = next(iter(response_format.model_fields))
                    if field_name == "area":
                        outputs.append(response_format(area="Atlantis"))
                    else:
                        outputs.append(response_format(**{field_name: text.upper()}))
            return outputs

    cleaner = Cleaner(df, UnknownAreaLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_AREA].iloc[0] == "Other"


def test_cleaner_area_prompt_includes_allowed_area_guidance():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["County Durham and Darlington"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    cleaner = Cleaner(df, DummyLLM())
    prompt = cleaner.area_prompt_template

    assert "Allowed area labels:" in prompt
    assert "County Durham and Darlington" in prompt
    assert "London West" in prompt
    assert "Return exactly one label from this list." in prompt
    assert "If no logical match exists, return exactly: <NA>." in prompt


@pytest.mark.parametrize(
    ("raw_area", "expected"),
    [
        ("West Yorkshire (E)", "Yorkshire West Eastern"),
        ("South Yorkshire (West)", "Yorkshire South West"),
        (
            "City of Kingston Upon Hull and the County of the East Riding of Yorkshire",
            "East Riding and Hull",
        ),
    ],
)
def test_cleaner_area_uses_deterministic_matching_for_near_matches(raw_area, expected):
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: [raw_area],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    cleaner = Cleaner(df, EchoAreaLLM())
    cleaned = cleaner.clean_reports()

    assert cleaned[GeneralConfig.COL_AREA].iloc[0] == expected


def test_cleaner_area_synonyms():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["West London"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    cleaner = Cleaner(df, EchoAreaLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_AREA].iloc[0] == "London West"


def test_cleaner_receiver_removes_chief_coroner_and_keeps_semicolon_formatting():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    class ReceiverLLM(DummyLLM):
        def generate(self, prompts, response_format=None, **kwargs):
            outputs = []
            for p in prompts:
                text = p.split("\n")[-1]
                if response_format is None:
                    if text == "x":
                        outputs.append("Chief Coroner; NHS England; Chief Coroner")
                    else:
                        outputs.append(text.upper())
                else:
                    field_name = next(iter(response_format.model_fields))
                    outputs.append(response_format(**{field_name: text.upper()}))
            return outputs

    cleaner = Cleaner(df, ReceiverLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_RECEIVER].iloc[0] == "NHS England"


def test_cleaner_receiver_strips_roles_but_preserves_capitalisation():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    class ReceiverLLM(DummyLLM):
        def generate(self, prompts, response_format=None, **kwargs):
            outputs = []
            for p in prompts:
                text = p.split("\n")[-1]
                if response_format is None:
                    if text == "x":
                        outputs.append("CEO, Cardinal Healthcare; Jane Smith, Chief Executive, NHS England")
                    else:
                        outputs.append(text.upper())
                else:
                    field_name = next(iter(response_format.model_fields))
                    outputs.append(response_format(**{field_name: text.upper()}))
            return outputs

    cleaner = Cleaner(df, ReceiverLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_RECEIVER].iloc[0] == "Cardinal Healthcare; NHS England"


def test_cleaner_receiver_normalises_articles_symbols_acronyms_and_departments():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    class ReceiverLLM(DummyLLM):
        def generate(self, prompts, response_format=None, **kwargs):
            outputs = []
            for p in prompts:
                text = p.split("\n")[-1]
                if response_format is None:
                    if text == "x":
                        outputs.append(
                            "The Department of Health & Social Care; "
                            "National Institute for Health and Care Excellence (NICE); "
                            "Chief Executive of NHS England; "
                            "The Care Quality Commission"
                        )
                    else:
                        outputs.append(text.upper())
                else:
                    field_name = next(iter(response_format.model_fields))
                    outputs.append(response_format(**{field_name: text.upper()}))
            return outputs

    cleaner = Cleaner(df, ReceiverLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_RECEIVER].iloc[0] == (
        "Department of Health and Social Care; "
        "National Institute for Health and Care Excellence; "
        "NHS England; "
        "Care Quality Commission"
    )


def test_cleaner_receiver_strips_low_risk_junk_and_qualifiers():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    class ReceiverLLM(DummyLLM):
        def generate(self, prompts, response_format=None, **kwargs):
            outputs = []
            for p in prompts:
                text = p.split("\n")[-1]
                if response_format is None:
                    if text == "x":
                        outputs.append(
                            "– Warwickshire County Council; "
                            "Secretary of State for the Department of Health and Social Care; "
                            "National Police Chiefs’ Council; "
                            'Nottingham University Hospitals NHS Trust ("the Trust")'
                        )
                    else:
                        outputs.append(text.upper())
                else:
                    field_name = next(iter(response_format.model_fields))
                    outputs.append(response_format(**{field_name: text.upper()}))
            return outputs

    cleaner = Cleaner(df, ReceiverLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_RECEIVER].iloc[0] == (
        "Warwickshire County Council; "
        "Secretary of State for Health and Social Care; "
        "National Police Chiefs' Council; "
        "Nottingham University Hospitals NHS Foundation Trust"
    )


def test_cleaner_receiver_merges_nhs_trust_and_foundation_trust_variants():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    class ReceiverLLM(DummyLLM):
        def generate(self, prompts, response_format=None, **kwargs):
            outputs = []
            for p in prompts:
                text = p.split("\n")[-1]
                if response_format is None:
                    if text == "x":
                        outputs.append(
                            "Barts Health NHS Trust; Barts Health NHS Foundation Trust"
                        )
                    else:
                        outputs.append(text.upper())
                else:
                    field_name = next(iter(response_format.model_fields))
                    outputs.append(response_format(**{field_name: text.upper()}))
            return outputs

    cleaner = Cleaner(df, ReceiverLLM())
    cleaned = cleaner.clean_reports()
    assert (
        cleaned[GeneralConfig.COL_RECEIVER].iloc[0]
        == "Barts Health NHS Foundation Trust"
    )


def test_cleaner_receiver_applies_explicit_canonical_mappings():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )

    class ReceiverLLM(DummyLLM):
        def generate(self, prompts, response_format=None, **kwargs):
            outputs = []
            for p in prompts:
                text = p.split("\n")[-1]
                if response_format is None:
                    if text == "x":
                        outputs.append(
                            "Department of Health; NHS England and NHS Improvement; Highways Agency; Highways England"
                        )
                    else:
                        outputs.append(text.upper())
                else:
                    field_name = next(iter(response_format.model_fields))
                    outputs.append(response_format(**{field_name: text.upper()}))
            return outputs

    cleaner = Cleaner(df, ReceiverLLM())
    cleaned = cleaner.clean_reports()
    assert cleaned[GeneralConfig.COL_RECEIVER].iloc[0] == (
        "Department of Health and Social Care; NHS England; National Highways"
    )


def test_area_model_normalises_formatting_but_returns_canonical_value():
    model = AreaModel(area="  east riding and hull  ")
    assert model.area == "East Riding and Hull"


def test_area_model_normalises_synonym_formatting_but_returns_canonical_value():
    model = AreaModel(area="cardiff & vale of glamorgan")
    assert model.area == "South Wales Central"


def test_area_model_recodes_legacy_area_to_current_canonical_value():
    model = AreaModel(area="Powys, Bridgend and Glamorgan")
    assert model.area == "South Wales Central"


@pytest.mark.parametrize(
    ("legacy_area", "current_area"),
    [
        ("Essex", "Essex and Thurrock"),
        ("Exeter & Greater Devon", "County of Devon, Plymouth and Torbay"),
        ("Lincolnshire", "Greater Lincolnshire"),
        ("North Lincolnshire & Grimsby", "Greater Lincolnshire"),
        ("North Northumberland", "Northumberland"),
        ("North Staffordshire and Stoke on Trent", "Staffordshire and Stoke-on-Trent"),
        ("Kent North West", "Kent and Medway"),
        ("Greater Manchester West", "Manchester West"),
        ("Nottinghamshire", "Nottinghamshire and Nottingham"),
        ("Newcastle and North Tyneside", "Newcastle Upon Tyne and North Tyneside"),
        ("Plymouth, Torbay & South Devon", "County of Devon, Plymouth and Torbay"),
        ("South Northumberland", "Northumberland"),
        ("South Staffordshire", "Staffordshire and Stoke-on-Trent"),
    ],
)
def test_area_model_recodes_additional_legacy_areas(legacy_area, current_area):
    model = AreaModel(area=legacy_area)
    assert model.area == current_area


def test_area_match_exact_then_fuzzy_recovers_typo():
    result = Cleaner.match_area("Londn West", strategy="exact_then_fuzzy")
    assert result.canonical_area == "London West"
    assert result.match_method == "fuzzy_allowed"


def test_area_match_fuzzy_only_recovers_typo():
    result = Cleaner.match_area("Londn West", strategy="fuzzy_only")
    assert result.canonical_area == "London West"
    assert result.match_method == "fuzzy_allowed"


def test_cleaner_summarise():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_CORONER_NAME: ["john"],
            GeneralConfig.COL_AREA: ["area"],
            GeneralConfig.COL_RECEIVER: ["x"],
            GeneralConfig.COL_INVESTIGATION: ["inv"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ"],
            GeneralConfig.COL_CONCERNS: ["conc"],
        }
    )
    extractor = Extractor(llm=DummyLLM(), reports=df)
    out = extractor.summarise()
    assert "summary" in out.columns
    assert len(out) == len(df)

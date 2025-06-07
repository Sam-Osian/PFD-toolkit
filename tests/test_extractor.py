import pandas as pd
from pydantic import BaseModel, Field
from pfd_toolkit.extractor import Extractor
from pfd_toolkit.config import GeneralConfig


class DummyLLM:
    def __init__(self, values=None):
        self.values = values or {}
        self.called = 0
        self.max_workers = 1

    def generate_batch(self, prompts, response_format=None, **kwargs):
        self.called += len(prompts)
        outputs = []
        for _ in prompts:
            if response_format is not None:
                outputs.append(response_format(**self.values))
            else:
                outputs.append(self.values)
        return outputs


class DemoModel(BaseModel):
    age: int = Field(..., description="Age")
    ethnicity: str = Field(..., description="Ethnicity")


def test_extractor_basic():
    df = pd.DataFrame(
        {"InvestigationAndInquest": ["text"], "CircumstancesOfDeath": ["other"]}
    )
    llm = DummyLLM(values={"age": 30, "ethnicity": "White"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result = extractor.extract_features()
    assert result["age"].iloc[0] == 30
    assert result["ethnicity"].iloc[0] == "White"
    assert llm.called == 1


def test_extractor_empty_df():
    df = pd.DataFrame(columns=["InvestigationAndInquest"])
    llm = DummyLLM(values={"age": 20, "ethnicity": "A"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
    )
    result = extractor.extract_features()
    assert result.empty
    assert llm.called == 0


def test_extractor_not_found_handling():
    df = pd.DataFrame({"InvestigationAndInquest": ["text"]})
    llm = DummyLLM(values={"age": GeneralConfig.NOT_FOUND_TEXT, "ethnicity": "B"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
    )
    result = extractor.extract_features()
    assert result["age"].iloc[0] == GeneralConfig.NOT_FOUND_TEXT
    assert result["ethnicity"].iloc[0] == "B"
    assert llm.called == 1


def test_extractor_force_assign():
    df = pd.DataFrame({"InvestigationAndInquest": ["text"]})
    llm = DummyLLM(values={"age": 40, "ethnicity": "C"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
        force_assign=True,
    )
    assert GeneralConfig.NOT_FOUND_TEXT not in extractor.prompt_template
    field_info = extractor._grammar_model.model_fields["age"]
    field_type = field_info.annotation
    assert str not in getattr(field_type, "__args__", (field_type,))



def test_extractor_allow_multiple_prompt_line():
    df = pd.DataFrame({"InvestigationAndInquest": ["text"]})
    llm = DummyLLM(values={"age": 10, "ethnicity": "E"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
        allow_multiple=True,
    )
    assert "multiple categories" in extractor.prompt_template


def test_feature_schema_full_and_minimal():
    llm = DummyLLM()
    extractor_full = Extractor(llm=llm, feature_model=DemoModel, schema_detail="full")
    extractor_min = Extractor(llm=llm, feature_model=DemoModel, schema_detail="minimal")

    expected_full = (
        "{\n"
        "  \"age\": {\n"
        "    \"description\": \"Age\",\n"
        "    \"title\": \"Age\",\n"
        "    \"type\": \"integer\"\n"
        "  },\n"
        "  \"ethnicity\": {\n"
        "    \"description\": \"Ethnicity\",\n"
        "    \"title\": \"Ethnicity\",\n"
        "    \"type\": \"string\"\n"
        "  }\n"
        "}"
    )

    expected_minimal = (
        "{\n"
        "  \"age\": {\n"
        "    \"type\": \"integer\",\n"
        "    \"description\": \"Age\"\n"
        "  },\n"
        "  \"ethnicity\": {\n"
        "    \"type\": \"string\",\n"
        "    \"description\": \"Ethnicity\"\n"
        "  }\n"
        "}"
    )

    assert extractor_full._feature_schema == expected_full
    assert extractor_min._feature_schema == expected_minimal


def test_extract_skip_if_present_default():
    df = pd.DataFrame(
        {
            "InvestigationAndInquest": ["text"],
            "CircumstancesOfDeath": ["other"],
            "age": [10],
            "ethnicity": ["Z"],
        }
    )
    llm = DummyLLM(values={"age": 99, "ethnicity": "Y"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result = extractor.extract_features()
    assert llm.called == 0
    assert result["age"].iloc[0] == 10


def test_extract_skip_if_present_partial_row():
    """Row is skipped when any feature column already contains data."""
    df = pd.DataFrame(
        {
            "InvestigationAndInquest": ["text"],
            "CircumstancesOfDeath": ["other"],
            "age": [GeneralConfig.NOT_FOUND_TEXT],
            "ethnicity": ["Cached"],
        }
    )
    llm = DummyLLM(values={"age": 88, "ethnicity": "New"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result = extractor.extract_features()
    # LLM should not be called because one feature value is already present
    assert llm.called == 0
    assert result["ethnicity"].iloc[0] == "Cached"
    assert result["age"].iloc[0] == GeneralConfig.NOT_FOUND_TEXT


def test_extract_skip_if_present_false():
    df = pd.DataFrame(
        {
            "InvestigationAndInquest": ["text"],
            "CircumstancesOfDeath": ["other"],
            "age": [1],
            "ethnicity": ["A"],
        }
    )
    llm = DummyLLM(values={"age": 55, "ethnicity": "B"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result = extractor.extract_features(skip_if_present=False)
    assert llm.called == 1
    assert result["age"].iloc[0] == 55


def test_extractor_caching():
    df = pd.DataFrame(
        {
            "InvestigationAndInquest": ["text"],
            "CircumstancesOfDeath": ["other"],
        }
    )
    llm = DummyLLM(values={"age": 21, "ethnicity": "C"})
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result1 = extractor.extract_features(skip_if_present=False)
    assert llm.called == 1

    df2 = pd.DataFrame(
        {
            "InvestigationAndInquest": ["text"],
            "CircumstancesOfDeath": ["other"],
            "age": [GeneralConfig.NOT_FOUND_TEXT],
            "ethnicity": [GeneralConfig.NOT_FOUND_TEXT],
        }
    )
    result2 = extractor.extract_features(df2, skip_if_present=False)
    assert llm.called == 1  # cached result used
    assert result2["age"].iloc[0] == 21
    assert result2["ethnicity"].iloc[0] == "C"


def test_export_import_cache(tmp_path):
    df = pd.DataFrame(
        {
            "InvestigationAndInquest": ["text"],
            "CircumstancesOfDeath": ["other"],
        }
    )
    llm1 = DummyLLM(values={"age": 50, "ethnicity": "D"})
    extractor1 = Extractor(
        llm=llm1,
        feature_model=DemoModel,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    extractor1.extract_features(skip_if_present=False)
    assert llm1.called == 1

    cache_dir = tmp_path / "cache"
    exported = extractor1.export_cache(cache_dir)
    assert cache_dir.exists()
    assert cache_dir.is_dir()
    assert exported

    llm2 = DummyLLM(values={"age": 99, "ethnicity": "X"})
    extractor2 = Extractor(
        llm=llm2,
        feature_model=DemoModel,
        include_investigation=True,
        include_circumstances=True,
    )
    extractor2.import_cache(cache_dir)

    df2 = pd.DataFrame(
        {
            "InvestigationAndInquest": ["text"],
            "CircumstancesOfDeath": ["other"],
            "age": [GeneralConfig.NOT_FOUND_TEXT],
            "ethnicity": [GeneralConfig.NOT_FOUND_TEXT],
        }
    )
    result = extractor2.extract_features(df2, skip_if_present=False)
    assert llm2.called == 0  # result from cache
    assert result["age"].iloc[0] == 50
    assert result["ethnicity"].iloc[0] == "D"


def test_prompt_additional_instructions():
    llm = DummyLLM()
    extractor = Extractor(
        llm=llm,
        feature_model=DemoModel,
        extra_instructions="Extra guidance",
    )
    assert "Extra guidance" in extractor.prompt_template

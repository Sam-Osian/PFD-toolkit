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
    field_info = extractor._llm_model.model_fields["age"]
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

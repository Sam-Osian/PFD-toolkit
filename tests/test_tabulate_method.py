import pandas as pd
import pytest
from pydantic import BaseModel

from pfd_toolkit.extractor import Extractor
from pfd_toolkit.config import GeneralConfig


class DummyLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.index = 0
        self.max_workers = 1

    def generate(self, prompts, response_format=None, **kwargs):
        results = []
        for _ in prompts:
            data = self.outputs[self.index]
            self.index += 1
            results.append(response_format(**data))
        return results

    def estimate_tokens(self, texts, model=None):
        if isinstance(texts, str):
            texts = [texts]
        return [len(t.split()) for t in texts]


class BoolModel(BaseModel):
    a: bool
    b: bool


class CatModel(BaseModel):
    cat: str | None = None


def test_tabulate_boolean_columns():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: [""] * 4})
    llm = DummyLLM([
        {"a": True, "b": False},
        {"a": False, "b": True},
        {"a": True, "b": True},
        {"a": False, "b": False},
    ])
    extractor = Extractor(llm=llm, reports=df, include_investigation=True)
    extractor.extract_features(feature_model=BoolModel, skip_if_present=False)

    table = extractor.tabulate(columns=["a", "b"], labels=["A", "B"])
    row_a = table[table["Category"] == "A"].iloc[0]
    row_b = table[table["Category"] == "B"].iloc[0]
    assert row_a["Count"] == 2
    assert row_b["Count"] == 2
    assert row_a["Percentage"] == 50.0
    assert row_b["Percentage"] == 50.0


def test_tabulate_categorical_column():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: [""] * 5})
    llm = DummyLLM([
        {"cat": "x"},
        {"cat": "y"},
        {"cat": "x"},
        {"cat": None},
        {"cat": "y"},
    ])
    extractor = Extractor(llm=llm, reports=df, include_investigation=True)
    extractor.extract_features(feature_model=CatModel, skip_if_present=False)

    table = extractor.tabulate(columns=["cat"], labels=["Category"])
    counts = dict(zip(table["Category"], table["Count"]))
    percents = dict(zip(table["Category"], table["Percentage"]))
    assert counts["Category: x"] == 2
    assert counts["Category: y"] == 2
    assert percents["Category: x"] == 40.0
    assert percents["Category: y"] == 40.0


def test_tabulate_label_length_mismatch():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["one"]})
    llm = DummyLLM([{"a": True, "b": False}])
    extractor = Extractor(llm=llm, reports=df, include_investigation=True)
    extractor.extract_features(feature_model=BoolModel, skip_if_present=False)
    with pytest.raises(ValueError):
        extractor.tabulate(columns=["a"], labels=["label1", "label2"])

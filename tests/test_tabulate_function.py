import pandas as pd
import pytest
from pfd_toolkit.extractor import tabulate


def test_tabulate_boolean_columns():
    df = pd.DataFrame({"a": [True, False, True, None], "b": [False, True, True, False]})
    table = tabulate(df, ["a", "b"], ["A", "B"])
    row_a = table[table["Category"] == "A"].iloc[0]
    row_b = table[table["Category"] == "B"].iloc[0]
    assert row_a["Count"] == 2
    assert row_b["Count"] == 2
    assert row_a["Percentage"] == 50.0
    assert row_b["Percentage"] == 50.0


def test_tabulate_categorical_column():
    df = pd.DataFrame({"cat": ["x", "y", "x", None, "y"]})
    table = tabulate(df, "cat", "Category")
    counts = dict(zip(table["Category"], table["Count"]))
    assert counts["Category: x"] == 2
    assert counts["Category: y"] == 2
    percents = dict(zip(table["Category"], table["Percentage"]))
    assert percents["Category: x"] == 40.0
    assert percents["Category: y"] == 40.0


def test_tabulate_label_length_mismatch():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        tabulate(df, ["a"], ["label1", "label2"])


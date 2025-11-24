import logging
import pandas as pd
import json
import pytest
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from pfd_toolkit.extractor import Extractor
from pfd_toolkit.config import GeneralConfig


class DummyLLM:
    def __init__(self, values=None):
        self.values = values or {}
        self.called = 0
        self.token_called = 0
        self.max_workers = 1
        self.model = "gpt-3.5-turbo"

    def generate(self, prompts, response_format=None, **kwargs):
        self.called += len(prompts)
        outputs = []
        for _ in prompts:
            if response_format is not None:
                vals = {k: (None if pd.isna(v) else v) for k, v in self.values.items()}
                outputs.append(response_format(**vals))
            else:
                if isinstance(self.values, dict):
                    outputs.append(json.dumps(self.values))
                else:
                    outputs.append(self.values)
        return outputs

    def estimate_tokens(self, texts, model=None):
        self.token_called += 1
        if isinstance(texts, str):
            texts = [texts]
        return [len((t or "").split()) for t in texts]


class DemoModel(BaseModel):
    age: int = Field(..., description="Age")
    ethnicity: str = Field(..., description="Ethnicity")


def test_extractor_basic():
    df = pd.DataFrame(
        {GeneralConfig.COL_INVESTIGATION: ["text"], GeneralConfig.COL_CIRCUMSTANCES: ["other"]}
    )
    llm = DummyLLM(values={"age": 30, "ethnicity": "White"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result = extractor.extract_features(feature_model=DemoModel)
    assert result["age"].iloc[0] == 30
    assert result["ethnicity"].iloc[0] == "White"
    assert llm.called == 1


def test_extract_produce_spans():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["text"]})
    llm = DummyLLM(
        values={
            "spans_age": "age 30",
            "age": 30,
            "spans_ethnicity": "white",
            "ethnicity": "White",
        }
    )
    extractor = Extractor(llm=llm, reports=df, include_investigation=True)
    result = extractor.extract_features(feature_model=DemoModel, produce_spans=True)

    assert list(extractor.feature_names) == [
        "spans_age",
        "age",
        "spans_ethnicity",
        "ethnicity",
    ]
    assert "quotation marks" in extractor.prompt_template
    assert result["spans_age"].iloc[0] == "age 30"
    assert result["ethnicity"].iloc[0] == "White"
    assert llm.called == 1


def test_extract_drop_spans():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["text"]})
    llm = DummyLLM(
        values={
            "spans_age": "age 30",
            "age": 30,
            "spans_ethnicity": "white",
            "ethnicity": "White",
        }
    )
    extractor = Extractor(llm=llm, reports=df, include_investigation=True)
    result = extractor.extract_features(
        feature_model=DemoModel, produce_spans=True, drop_spans=True
    )

    assert "spans_age" not in result.columns
    assert "spans_ethnicity" not in result.columns
    assert result["age"].iloc[0] == 30
    assert result["ethnicity"].iloc[0] == "White"
    assert llm.called == 1


def test_extract_drop_spans_warns():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["text"]})
    llm = DummyLLM(values={"age": 30, "ethnicity": "White"})
    extractor = Extractor(llm=llm, reports=df, include_investigation=True)
    with pytest.warns(UserWarning):
        result = extractor.extract_features(
            feature_model=DemoModel, produce_spans=False, drop_spans=True
        )

    assert "spans_age" not in result.columns
    assert "spans_ethnicity" not in result.columns
    assert result["age"].iloc[0] == 30
    assert result["ethnicity"].iloc[0] == "White"


def test_summarise_truncate_concatenation():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["one two three four five six"],
            GeneralConfig.COL_CIRCUMSTANCES: ["seven eight nine"],
            GeneralConfig.COL_CONCERNS: ["ten eleven"],
        }
    )
    llm = DummyLLM()
    extractor = Extractor(llm=llm, reports=df)

    summarised = extractor.summarise(trim_intensity="none", truncate=5)

    assert summarised["summary"].iloc[0] == "one two three four five"
    assert llm.called == 0


def test_count_table_words_threshold(capsys):
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["one two three", "one two", "one two three four five"],
            GeneralConfig.COL_CIRCUMSTANCES: ["", "", ""],
        }
    )
    extractor = Extractor(llm=DummyLLM(), reports=df)

    table = extractor.count(measure="words", as_="table", threshold=4, step=2)

    captured = capsys.readouterr().out
    assert "10 total words" in captured
    assert "5 total words within the threshold" in captured
    assert table.loc[table["threshold"] == 4, "cumulative_count"].iloc[0] == 2


def test_count_tokens_chart_ignores_step_info(caplog, capsys):
    df = pd.DataFrame(
        {GeneralConfig.COL_INVESTIGATION: ["alpha beta gamma"], GeneralConfig.COL_CIRCUMSTANCES: ["delta"]}
    )
    llm = DummyLLM()
    extractor = Extractor(llm=llm, reports=df)

    with caplog.at_level(logging.INFO):
        fig, _ = extractor.count(measure="tokens", as_="hist", threshold=3, step=25)

    captured = capsys.readouterr().out
    assert "4 total tokens" in captured
    assert "0 total tokens within the threshold" in captured
    assert "Ignoring step parameter" in caplog.text
    assert llm.token_called == 1
    plt.close(fig)


def test_count_stats_words_markdown(capsys, caplog):
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["one two three", "one two three four five"],
            GeneralConfig.COL_CIRCUMSTANCES: ["", ""],
        }
    )
    extractor = Extractor(llm=DummyLLM(), reports=df)

    with caplog.at_level(logging.INFO):
        markdown = extractor.count(measure="words", as_="stats", threshold=4, step=25)

    captured = capsys.readouterr().out
    assert captured == ""
    assert "Ignoring step parameter" in caplog.text
    assert "total_words" in markdown
    assert "within_threshold_words" in markdown
    assert "median" in markdown


def test_count_stats_tokens_logging(capsys, caplog):
    df = pd.DataFrame(
        {GeneralConfig.COL_INVESTIGATION: ["alpha beta", "gamma"], GeneralConfig.COL_CIRCUMSTANCES: ["", "delta"]}
    )
    llm = DummyLLM()
    extractor = Extractor(llm=llm, reports=df)

    with caplog.at_level(logging.INFO):
        markdown = extractor.count(measure="tokens", as_="stats", step=50)

    captured = capsys.readouterr().out
    assert captured == ""
    assert "total_tokens" in markdown
    assert "iqr" in markdown
    assert "Ignoring step parameter" in caplog.text
    assert llm.token_called == 1


def test_extract_drop_spans_preserves_existing():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["text"],
            "spans_matches_query": ["span"],
        }
    )
    llm = DummyLLM(
        values={
            "spans_age": "age 30",
            "age": 30,
            "spans_ethnicity": "white",
            "ethnicity": "White",
        }
    )
    extractor = Extractor(llm=llm, reports=df, include_investigation=True)
    result = extractor.extract_features(
        feature_model=DemoModel, produce_spans=True, drop_spans=True
    )

    assert "spans_matches_query" in result.columns
    assert result["spans_matches_query"].iloc[0] == "span"
    assert "spans_age" not in result.columns
    assert "spans_ethnicity" not in result.columns


def test_extract_spans_blank_replaced():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["text"]})
    llm = DummyLLM(
        values={
            "spans_age": " ",
            "age": 30,
            "spans_ethnicity": "",
            "ethnicity": "White",
        }
    )
    extractor = Extractor(llm=llm, reports=df, include_investigation=True)
    result = extractor.extract_features(feature_model=DemoModel, produce_spans=True)

    assert pd.isna(result["spans_age"].iloc[0])
    assert pd.isna(result["spans_ethnicity"].iloc[0])


def test_extractor_empty_df():
    df = pd.DataFrame(columns=[GeneralConfig.COL_INVESTIGATION])
    llm = DummyLLM(values={"age": 20, "ethnicity": "A"})
    extractor = Extractor(
        llm=llm,
        reports=df,
    )
    result = extractor.extract_features(feature_model=DemoModel)
    assert result.empty
    assert llm.called == 0


def test_extractor_not_found_handling():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["text"]})
    llm = DummyLLM(values={"age": GeneralConfig.NOT_FOUND_TEXT, "ethnicity": "B"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
    )
    result = extractor.extract_features(feature_model=DemoModel)
    assert pd.isna(result["age"].iloc[0])
    assert result["ethnicity"].iloc[0] == "B"
    assert llm.called == 1


def test_extractor_force_assign():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["text"]})
    llm = DummyLLM(values={"age": 40, "ethnicity": "C"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
    )
    extractor.extract_features(feature_model=DemoModel, force_assign=True)
    assert str(GeneralConfig.NOT_FOUND_TEXT) not in extractor.prompt_template
    field_info = extractor._grammar_model.model_fields["age"]
    field_type = field_info.annotation
    assert str not in getattr(field_type, "__args__", (field_type,))



def test_extractor_allow_multiple_prompt_line():
    df = pd.DataFrame({GeneralConfig.COL_INVESTIGATION: ["text"]})
    llm = DummyLLM(values={"age": 10, "ethnicity": "E"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
    )
    extractor.extract_features(feature_model=DemoModel, allow_multiple=True)
    assert "multiple categories" in extractor.prompt_template


def test_feature_schema_full_and_minimal():
    llm = DummyLLM()
    extractor_full = Extractor(llm=llm)
    extractor_min = Extractor(llm=llm)
    extractor_full.extract_features(feature_model=DemoModel, schema_detail="full")
    extractor_min.extract_features(feature_model=DemoModel, schema_detail="minimal")

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
            GeneralConfig.COL_INVESTIGATION: ["text"],
            GeneralConfig.COL_CIRCUMSTANCES: ["other"],
            "age": [10],
            "ethnicity": ["Z"],
        }
    )
    llm = DummyLLM(values={"age": 99, "ethnicity": "Y"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result = extractor.extract_features(feature_model=DemoModel)
    # No cache present, so features are re-extracted and overwritten
    assert llm.called == 1
    assert result["age"].iloc[0] == 99


def test_extract_skip_if_present_partial_row():
    """Row is processed again when cache is empty despite partial data."""
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["text"],
            GeneralConfig.COL_CIRCUMSTANCES: ["other"],
            "age": [GeneralConfig.NOT_FOUND_TEXT],
            "ethnicity": ["Cached"],
        }
    )
    llm = DummyLLM(values={"age": 88, "ethnicity": "New"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result = extractor.extract_features(feature_model=DemoModel)
    # Cache is empty so the row is processed again
    assert llm.called == 1
    assert result["ethnicity"].iloc[0] == "New"
    assert result["age"].iloc[0] == 88


def test_extract_skip_if_present_false():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["text"],
            GeneralConfig.COL_CIRCUMSTANCES: ["other"],
            "age": [1],
            "ethnicity": ["A"],
        }
    )
    llm = DummyLLM(values={"age": 55, "ethnicity": "B"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result = extractor.extract_features(feature_model=DemoModel, skip_if_present=False)
    assert llm.called == 1
    assert result["age"].iloc[0] == 55


def test_extractor_caching():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["text"],
            GeneralConfig.COL_CIRCUMSTANCES: ["other"],
        }
    )
    llm = DummyLLM(values={"age": 21, "ethnicity": "C"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    result1 = extractor.extract_features(feature_model=DemoModel, skip_if_present=False)
    assert llm.called == 1

    df2 = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["text"],
            GeneralConfig.COL_CIRCUMSTANCES: ["other"],
            "age": [GeneralConfig.NOT_FOUND_TEXT],
            "ethnicity": [GeneralConfig.NOT_FOUND_TEXT],
        }
    )
    result2 = extractor.extract_features(df2, skip_if_present=False)
    assert llm.called == 1  # cached result used
    assert result2["age"].iloc[0] == 21
    assert result2["ethnicity"].iloc[0] == "C"


def test_reset_allows_rerun():
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["text"],
            GeneralConfig.COL_CIRCUMSTANCES: ["other"],
        }
    )
    llm = DummyLLM(values={"age": 10, "ethnicity": "A"})
    extractor = Extractor(
        llm=llm,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    extractor.extract_features(feature_model=DemoModel, skip_if_present=False)
    assert llm.called == 1

    # Change values and rerun after reset
    llm.values = {"age": 20, "ethnicity": "B"}
    extractor.reset().extract_features(feature_model=DemoModel)
    assert llm.called == 2


def test_export_import_cache(tmp_path):
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["text"],
            GeneralConfig.COL_CIRCUMSTANCES: ["other"],
        }
    )
    llm1 = DummyLLM(values={"age": 50, "ethnicity": "D"})
    extractor1 = Extractor(
        llm=llm1,
        reports=df,
        include_investigation=True,
        include_circumstances=True,
    )
    extractor1.extract_features(feature_model=DemoModel, skip_if_present=False)
    assert llm1.called == 1

    cache_file = tmp_path / "cache" / "cache.pkl"
    exported = extractor1.export_cache(cache_file)
    assert cache_file.exists()
    assert cache_file.parent.is_dir()
    assert exported

    llm2 = DummyLLM(values={"age": 99, "ethnicity": "X"})
    extractor2 = Extractor(
        llm=llm2,
        include_investigation=True,
        include_circumstances=True,
    )
    extractor2.import_cache(cache_file)

    df2 = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["text"],
            GeneralConfig.COL_CIRCUMSTANCES: ["other"],
            "age": [GeneralConfig.NOT_FOUND_TEXT],
            "ethnicity": [GeneralConfig.NOT_FOUND_TEXT],
        }
    )
    result = extractor2.extract_features(df2, feature_model=DemoModel, skip_if_present=False)
    assert llm2.called == 0  # result from cache
    assert result["age"].iloc[0] == 50
    assert result["ethnicity"].iloc[0] == "D"


def test_prompt_additional_instructions():
    llm = DummyLLM()
    extractor = Extractor(llm=llm)
    extractor.extract_features(feature_model=DemoModel, extra_instructions="Extra guidance")
    assert "Extra guidance" in extractor.prompt_template


def test_estimate_tokens_default_column():
    df = pd.DataFrame({"summary": ["hello world"]})
    llm = DummyLLM()
    extractor = Extractor(llm=llm, reports=df)
    extractor.summarised_reports = df
    tokens = extractor.estimate_tokens(return_series=True)
    assert int(tokens.iloc[0]) == llm.estimate_tokens(["hello world"])[0]


def test_token_cache_export_import(tmp_path):
    df = pd.DataFrame({"summary": ["a short summary"]})
    llm = DummyLLM()
    ext1 = Extractor(llm=llm, reports=df)
    ext1.summarised_reports = df
    ext1.estimate_tokens()
    cache_file = ext1.export_cache(tmp_path / "c.pkl")

    ext2 = Extractor(llm=llm)
    ext2.import_cache(cache_file)
    assert ext2.token_cache == ext1.token_cache


def test_discover_themes_basic(monkeypatch):
    df = pd.DataFrame(
        {
            GeneralConfig.COL_INVESTIGATION: ["one", "two"],
            GeneralConfig.COL_CIRCUMSTANCES: ["circ1", "circ2"],
            GeneralConfig.COL_CONCERNS: ["conc1", "conc2"],
        }
    )
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        outputs = []
        for prompt in prompts:
            if "Report excerpt" in prompt:
                outputs.append("summary text")
            else:
                outputs.append(json.dumps({"safety": "Cases about safety"}))
        return outputs

    monkeypatch.setattr(llm, "generate", fake_generate)

    theme_model = ext.discover_themes()

    assert issubclass(theme_model, BaseModel)
    assert "safety" in theme_model.model_fields
    assert ext.feature_model is theme_model


def test_discover_themes_handles_code_fence(monkeypatch):
    df = pd.DataFrame({GeneralConfig.COL_CONCERNS: ["one"]})
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        outputs = []
        for prompt in prompts:
            if "Report excerpt" in prompt:
                outputs.append("summary text")
            else:
                outputs.append("```json\n{\n  \"fence\": \"ok\"\n}\n```")
        return outputs

    monkeypatch.setattr(llm, "generate", fake_generate)

    theme_model = ext.discover_themes()

    assert "fence" in theme_model.model_fields
    assert ext.identified_themes == "```json\n{\n  \"fence\": \"ok\"\n}\n```"


def test_discover_themes_bool_fields(monkeypatch):
    df = pd.DataFrame({GeneralConfig.COL_CONCERNS: ["one"]})
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        outputs = []
        for prompt in prompts:
            if "Report excerpt" in prompt:
                outputs.append("summary text")
            else:
                outputs.append(json.dumps({"example": "desc"}))
        return outputs

    monkeypatch.setattr(llm, "generate", fake_generate)

    theme_model = ext.discover_themes()

    assert theme_model.model_fields["example"].annotation is bool


def test_discover_themes_uses_token_cache(monkeypatch):
    df = pd.DataFrame({GeneralConfig.COL_CONCERNS: ["one", "two"]})
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        outputs = []
        for prompt in prompts:
            if "Report excerpt" in prompt:
                outputs.append("summary text")
            else:
                outputs.append(json.dumps({}))
        return outputs

    monkeypatch.setattr(llm, "generate", fake_generate)

    ext.discover_themes()

    llm.token_called = 0
    ext.token_cache[ext.summary_col] = [1, 2]

    ext.discover_themes()

    assert llm.token_called == 0


def test_discover_themes_prompt_limits(monkeypatch):
    df = pd.DataFrame({GeneralConfig.COL_CONCERNS: ["one"]})
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    captured = {}

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        prompt = prompts[0]
        if "Report excerpt" in prompt:
            return ["summary text"]
        captured["prompt"] = prompt
        return [json.dumps({})]

    monkeypatch.setattr(llm, "generate", fake_generate)
    ext.discover_themes(max_themes=5, min_themes=2)

    prompt = captured["prompt"].lower()
    assert "no more than **5" in prompt
    assert "at least **2" in prompt


def test_discover_themes_seed_topics_string(monkeypatch):
    df = pd.DataFrame({GeneralConfig.COL_CONCERNS: ["one"]})
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    captured = {}

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        prompt = prompts[0]
        if "Report excerpt" in prompt:
            return ["summary text"]
        captured["prompt"] = prompt
        return [json.dumps({})]

    monkeypatch.setattr(llm, "generate", fake_generate)
    ext.discover_themes(seed_topics="health")

    prompt = captured["prompt"].lower()
    assert "seed topics" in prompt
    assert "health" in prompt


def test_discover_themes_seed_topics_model(monkeypatch):
    class SeedModel(BaseModel):
        topic_a: str
        topic_b: str

    df = pd.DataFrame({GeneralConfig.COL_CONCERNS: ["one"]})
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    captured = {}

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        prompt = prompts[0]
        if "Report excerpt" in prompt:
            return ["summary text"]
        captured["prompt"] = prompt
        return [json.dumps({})]

    monkeypatch.setattr(llm, "generate", fake_generate)
    seeds = SeedModel(topic_a="desc", topic_b="desc")
    ext.discover_themes(seed_topics=seeds)

    prompt = captured["prompt"]
    assert "topic_a" in prompt
    assert "seed topics" in prompt.lower()


def test_discover_themes_adds_extra_instructions_to_summaries(monkeypatch):
    df = pd.DataFrame({GeneralConfig.COL_CONCERNS: ["one"]})
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    captured = {}

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        prompt = prompts[0]
        if "Report excerpt" in prompt:
            captured["summary_prompt"] = prompt
            return ["summary text"]
        return [json.dumps({})]

    monkeypatch.setattr(llm, "generate", fake_generate)
    instructions = "Themes related to setting (e.g. in hospital, at home, etc.)"
    ext.discover_themes(extra_instructions=instructions)

    summary_prompt = captured["summary_prompt"]
    assert "Downstream, your summary will be used by an analyst" in summary_prompt
    assert instructions in summary_prompt


def test_discover_themes_respects_trim_intensity(monkeypatch):
    df = pd.DataFrame({GeneralConfig.COL_CONCERNS: ["one"]})
    llm = DummyLLM()
    ext = Extractor(llm=llm, reports=df)

    captured = {}

    def fake_generate(prompts, response_format=None, **kwargs):
        llm.called += len(prompts)
        prompt = prompts[0]
        if "Report excerpt" in prompt:
            captured["summary_prompt"] = prompt
            return ["summary text"]
        return [json.dumps({})]

    monkeypatch.setattr(llm, "generate", fake_generate)
    ext.discover_themes(trim_intensity="very high")

    summary_prompt = captured["summary_prompt"].lower()
    assert "one or two sentence summary" in summary_prompt

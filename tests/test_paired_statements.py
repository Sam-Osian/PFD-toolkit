import pandas as pd

from pfd_toolkit.config import GeneralConfig
from pfd_toolkit.paired_statements import (
    ActionPhrase,
    ConcernItem,
    ConcernParser,
    ConcernResponseItem,
    ConcernResponseSet,
    ConcernSet,
    ResponseAuthor,
)


class DummyLLM:
    """Minimal stub matching the LLM.generate interface used in tests."""

    def __init__(
        self,
        responses: list[ConcernSet] | None = None,
        response_actions: list[ConcernResponseSet] | None = None,
    ):
        self.responses = responses or []
        self.response_actions = response_actions or []

    def generate(self, prompts, response_format, **_kwargs):
        if response_format is ConcernResponseSet:
            if self.response_actions:
                return self.response_actions
            return [response_format() for _ in prompts]
        if self.responses:
            return self.responses
        return [response_format() for _ in prompts]


def test_concern_item_normalises_whitespace():
    item = ConcernItem(concern="  multiple   spaces\n and lines ")
    assert item.concern == "multiple spaces and lines"


def test_prompt_includes_context_and_source_text():
    generator = ConcernParser(llm=DummyLLM(), reports=[])
    sample_text = "I am concerned about how emergency calls are triaged."
    prompt = generator.build_prompt(sample_text)

    assert "Prevention of Future Deaths" in prompt
    assert "Matters of Concern" in prompt
    assert sample_text in prompt
    assert "Guidance for identifying concerns" in prompt


def test_extract_for_reports_matches_url_and_parent_url():
    llm = DummyLLM(
        responses=[
            ConcernSet(concerns=[ConcernItem(concern="one")]),
            ConcernSet(concerns=[ConcernItem(concern="two")]),
        ]
    )
    reports = [
        {
            GeneralConfig.COL_URL: "https://example.test/report-1",
            GeneralConfig.COL_CONCERNS: "concern text",
            GeneralConfig.COL_RECEIVER: "A;B",
            GeneralConfig.COL_ID: "2024-0001",
        },
        {
            GeneralConfig.COL_URL: "https://example.test/report-2",
            GeneralConfig.COL_CONCERNS: "another",
            GeneralConfig.COL_RECEIVER: "",
            GeneralConfig.COL_ID: "2024-0002",
        },
    ]
    responses = [
        {"parent_url": "https://example.test/report-1", "response": "resp A"},
        {"parent_url": "https://example.test/report-1", "response": "resp B"},
        {"parent_url": "https://example.test/report-3", "response": "irrelevant"},
    ]

    generator = ConcernParser(llm=llm, reports=reports, responses=responses)

    paired = generator.parse_concerns(output="object")

    assert paired.reports[0].url == reports[0][GeneralConfig.COL_URL]
    assert [item.concern for item in paired.reports[0].concerns] == ["one"]
    assert paired.reports[0].responses == ["resp A", "resp B"]
    assert paired.reports[0].recipients == ["A", "B"]
    assert paired.reports[0].report_id == "2024-0001"

    assert paired.reports[1].url == reports[1][GeneralConfig.COL_URL]
    assert paired.reports[1].responses == []
    assert paired.reports[1].recipients == []


def test_as_df_flattens_concerns():
    llm = DummyLLM(responses=[ConcernSet(concerns=[ConcernItem(concern="one")])])
    reports_df = pd.DataFrame(
        [
            {
                GeneralConfig.COL_URL: "https://example.test/report-1",
                GeneralConfig.COL_CONCERNS: "concern text",
                GeneralConfig.COL_RECEIVER: "Receiver A; Receiver B",
                GeneralConfig.COL_ID: "2024-0001",
            }
        ]
    )

    generator = ConcernParser(llm=llm, reports=reports_df)

    results = generator.parse_concerns(output="object")
    df = results.as_df()

    assert len(df) == 1
    assert df.iloc[0][GeneralConfig.COL_URL] == "https://example.test/report-1"
    assert df.iloc[0][GeneralConfig.COL_ID] == "2024-0001"
    assert df.iloc[0]["concern"] == "one"
    assert df.iloc[0][GeneralConfig.COL_RECEIVER] == [
        "Receiver A",
        "Receiver B",
    ]

    json_ready = results.as_list()[0]
    assert json_ready[GeneralConfig.COL_URL] == "https://example.test/report-1"
    assert json_ready[GeneralConfig.COL_ID] == "2024-0001"
    assert json_ready[GeneralConfig.COL_RECEIVER] == [
        "Receiver A",
        "Receiver B",
    ]
    assert json_ready["concerns"] == ["one"]

    json_str = results.as_json()
    assert "\n" in json_str


def test_count_methods_require_parsed_concerns_for_concern_counts():
    llm = DummyLLM(
        responses=[ConcernSet(concerns=[ConcernItem(concern="concern one")])]
    )
    reports = [
        {
            GeneralConfig.COL_URL: "https://example.test/report-1",
            GeneralConfig.COL_CONCERNS: "concern text",
            GeneralConfig.COL_RECEIVER: "Recipient A",
            GeneralConfig.COL_ID: "2024-0001",
        }
    ]
    responses = [
        {"parent_url": "https://example.test/report-1", "response": "resp A"},
    ]

    parser = ConcernParser(llm=llm, reports=reports, responses=responses)

    assert parser.count("reports") == {"reports": 1}
    response_counts = parser.count("responses")
    assert response_counts["responses"] == 1
    assert response_counts["mean_responses_per_report"] == 1.0

    parser.parse_concerns()
    assert parser.count("concerns") == {"concerns": 1}
    combined = parser.count("all")
    assert combined["reports"] == 1
    assert combined["concerns"] == 1


def test_parse_responses_aligns_to_concerns_and_counts_actions():
    concerns_output = ConcernSet(
        concerns=[ConcernItem(concern="concern one"), ConcernItem(concern="concern two")]
    )
    response_output = ConcernResponseSet(
        concerns=[
            ConcernResponseItem(
                concern="concern one",
                responses=[
                    ResponseAuthor(
                        author="Org A",
                        action_phrases=[ActionPhrase(action_phrase="promise a change")],
                    )
                ],
            )
        ]
    )

    llm = DummyLLM(responses=[concerns_output], response_actions=[response_output])
    reports = [
        {
            GeneralConfig.COL_URL: "https://example.test/report-1",
            GeneralConfig.COL_CONCERNS: "concern text",
            GeneralConfig.COL_RECEIVER: "Recipient A; Recipient B",
            GeneralConfig.COL_ID: "2024-0001",
        }
    ]
    responses = [
        {"parent_url": "https://example.test/report-1", "response": "Letter from Org A"}
    ]

    parser = ConcernParser(llm=llm, reports=reports, responses=responses)
    parser.parse_concerns()
    paired = parser.parse_responses(output="object")

    assert len(paired.reports) == 1
    concern_items = paired.reports[0].concern_responses.concerns
    assert concern_items[0].responses[0].author == "Org A"
    assert concern_items[0].responses[0].action_phrases[0].action_phrase == "promise a change"
    assert concern_items[1].responses[0].author == "[no response]"
    assert concern_items[1].responses[0].action_phrases[0].action_phrase == "[no response]"

    df = paired.as_df()
    assert {"response_author", "action_phrase"}.issubset(df.columns)
    assert parser.count("action_phrases") == {"action_phrases": 2}

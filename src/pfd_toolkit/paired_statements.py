"""LLM-powered concern pairing for Prevention of Future Deaths reports.

This module focuses on converting the free-text "Matters of Concern" section
into a structured list of concise, discrete concerns suitable for downstream
pairing and analysis, and on linking those concerns to actions described in
response documents.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .config import GeneralConfig
from .llm import LLM


class ConcernItem(BaseModel):
    """A single distilled concern from a coroner's report."""

    concern: str = Field(
        ..., description="One short concern capturing a single specific risk or failing."
    )

    @field_validator("concern")
    @classmethod
    def _normalise_text(cls, value: str) -> str:
        cleaned = " ".join((value or "").split()).strip()
        if not cleaned:
            raise ValueError("concern text cannot be empty")
        return cleaned


class ActionPhrase(BaseModel):
    """Short description of an action or commitment from a response."""

    action_phrase: str = Field(
        ..., description="One concise action or commitment drawn from a response."
    )

    @field_validator("action_phrase")
    @classmethod
    def _normalise_text(cls, value: str) -> str:
        cleaned = " ".join((value or "").split()).strip()
        if not cleaned:
            raise ValueError("action_phrase text cannot be empty")
        return cleaned


class ResponseAuthor(BaseModel):
    """Author of a response and their associated actions."""

    author: str = Field(..., description="Name of the individual or organisation.")
    action_phrases: List[ActionPhrase] = Field(
        default_factory=list,
        description="List of discrete actions or commitments attributed to this author.",
    )

    @field_validator("author")
    @classmethod
    def _normalise_text(cls, value: str) -> str:
        cleaned = " ".join((value or "").split()).strip()
        if not cleaned:
            raise ValueError("author name cannot be empty")
        return cleaned


class ConcernResponseItem(BaseModel):
    """A concern paired with response authors and their action phrases."""

    concern: str = Field(..., description="Concern text copied from the report.")
    responses: List[ResponseAuthor] = Field(
        default_factory=list,
        description="Authors and their actions linked to this concern.",
    )

    @field_validator("concern")
    @classmethod
    def _normalise_text(cls, value: str) -> str:
        cleaned = " ".join((value or "").split()).strip()
        if not cleaned:
            raise ValueError("concern text cannot be empty")
        return cleaned


class ConcernSet(BaseModel):
    """Ordered collection of concerns extracted from a PFD report."""

    concerns: List[ConcernItem] = Field(
        default_factory=list,
        description=(
            "A list of discrete concerns, kept in the same order as the "
            "original Matters of Concern section."
        ),
    )


class ConcernResponseSet(BaseModel):
    """Responses mapped to concerns for a single PFD report."""

    concerns: List[ConcernResponseItem] = Field(
        default_factory=list,
        description="List of concerns with any linked responses and action phrases.",
    )


class ReportConcerns(BaseModel):
    """Concerns for a single report plus propagated metadata."""

    url: str = Field(..., description="Canonical URL for the PFD report.")
    report_id: str | None = Field(
        default=None, description="Unique identifier scraped from the report URL."
    )
    recipients: List[str] = Field(
        default_factory=list, description="Recipient list split from the reports CSV."
    )
    concern_set: ConcernSet = Field(
        ..., description="Structured concerns parsed from the Matters of Concern section."
    )
    responses: List[str] = Field(
        default_factory=list,
        description="All response texts linked to this report's URL (one-to-many).",
    )
    concern_responses: ConcernResponseSet | None = Field(
        default=None,
        description="Structured responses paired with each concern when available.",
    )

    @property
    def concerns(self) -> List[ConcernItem]:
        """Expose the inner concerns list for ergonomic access."""

        return self.concern_set.concerns


class ConcernResults(BaseModel):
    """Container for multiple reports worth of concerns."""

    reports: List[ReportConcerns] = Field(default_factory=list)

    def as_list(self) -> List[dict[str, Any]]:
        """Return a JSON-ready list of report-level concerns."""

        serialised: List[dict[str, Any]] = []
        for report in self.reports:
            serialised.append(
                {
                    GeneralConfig.COL_URL: report.url,
                    GeneralConfig.COL_ID: report.report_id,
                    GeneralConfig.COL_RECEIVER: report.recipients,
                    "responses": report.responses,
                    "concerns": [item.concern for item in report.concerns],
                }
            )
        return serialised

    def as_json(self) -> str:
        """Return a JSON string serialising the concerns list."""

        return json.dumps(self.as_list(), ensure_ascii=False, indent=2)

    def as_df(self) -> pd.DataFrame:
        """Return a flat DataFrame, one row per concern."""

        rows: List[dict[str, Any]] = []
        for report in self.reports:
            for item in report.concerns:
                rows.append(
                    {
                        GeneralConfig.COL_URL: report.url,
                        GeneralConfig.COL_ID: report.report_id,
                        GeneralConfig.COL_RECEIVER: report.recipients,
                        "concern": item.concern,
                    }
                )
        return pd.DataFrame(rows)


class ReportConcernResponses(BaseModel):
    """Concern-response pairings for a single report."""

    url: str = Field(..., description="Canonical URL for the PFD report.")
    report_id: str | None = Field(
        default=None, description="Unique identifier scraped from the report URL."
    )
    recipients: List[str] = Field(
        default_factory=list, description="Recipient list split from the reports CSV."
    )
    responses: List[str] = Field(
        default_factory=list,
        description="All response texts linked to this report's URL (one-to-many).",
    )
    concern_responses: ConcernResponseSet = Field(
        default_factory=ConcernResponseSet,
        description="Structured concerns with linked response authors and actions.",
    )


class ResponseResults(BaseModel):
    """Container for multiple reports worth of concern-response pairings."""

    reports: List[ReportConcernResponses] = Field(default_factory=list)

    def as_list(self) -> List[dict[str, Any]]:
        """Return a JSON-ready list of concern-response mappings."""

        serialised: List[dict[str, Any]] = []
        for report in self.reports:
            serialised.append(
                {
                    GeneralConfig.COL_URL: report.url,
                    GeneralConfig.COL_ID: report.report_id,
                    GeneralConfig.COL_RECEIVER: report.recipients,
                    "responses": report.responses,
                    "concern_responses": [
                        {
                            "concern": item.concern,
                            "responses": [
                                {
                                    "author": response.author,
                                    "action_phrases": [
                                        action.action_phrase
                                        for action in response.action_phrases
                                    ],
                                }
                                for response in item.responses
                            ],
                        }
                        for item in report.concern_responses.concerns
                    ],
                }
            )
        return serialised

    def as_json(self) -> str:
        """Return a JSON string serialising concern-response mappings."""

        return json.dumps(self.as_list(), ensure_ascii=False, indent=2)

    def as_df(self) -> pd.DataFrame:
        """Return a flat DataFrame, one row per action phrase."""

        rows: List[dict[str, Any]] = []
        for report in self.reports:
            for item in report.concern_responses.concerns:
                if not item.responses:
                    rows.append(
                        {
                            GeneralConfig.COL_URL: report.url,
                            GeneralConfig.COL_ID: report.report_id,
                            GeneralConfig.COL_RECEIVER: report.recipients,
                            "concern": item.concern,
                            "response_author": "[no response]",
                            "action_phrase": "[no response]",
                        }
                    )
                    continue
                for response in item.responses:
                    if not response.action_phrases:
                        rows.append(
                            {
                                GeneralConfig.COL_URL: report.url,
                                GeneralConfig.COL_ID: report.report_id,
                                GeneralConfig.COL_RECEIVER: report.recipients,
                                "concern": item.concern,
                                "response_author": response.author,
                                "action_phrase": "[no response]",
                            }
                        )
                        continue
                    for action in response.action_phrases:
                        rows.append(
                            {
                                GeneralConfig.COL_URL: report.url,
                                GeneralConfig.COL_ID: report.report_id,
                                GeneralConfig.COL_RECEIVER: report.recipients,
                                "concern": item.concern,
                                "response_author": response.author,
                                "action_phrase": action.action_phrase,
                            }
                        )
        return pd.DataFrame(rows)


class ConcernParser:
    """LLM-based manipulation of coroners' concerns.

    Parameters
    ----------
    llm : LLM
        LLM client used to call the OpenAI API. Parallelisation and retry
        settings are controlled directly by this object and should be provided
        by the caller.
    reports : Iterable[Mapping[str, Any]] | Any
        Table-like collection of PFD reports containing at least the URL and
        Matters of Concern text.
    responses : Iterable[Mapping[str, Any]] | Any | None
        Optional table-like collection of responses keyed by parent URL.
    """

    REPORT_URL_COLUMN = GeneralConfig.COL_URL
    REPORT_ID_COLUMN = GeneralConfig.COL_ID
    REPORT_RECEIVER_COLUMN = GeneralConfig.COL_RECEIVER
    REPORT_CONCERNS_COLUMN = GeneralConfig.COL_CONCERNS
    RESPONSES_PARENT_URL_COLUMN = "parent_url"
    RESPONSES_TEXT_COLUMN = "response"

    def __init__(
        self,
        llm: LLM,
        reports: Iterable[Mapping[str, Any]] | Any,
        responses: Iterable[Mapping[str, Any]] | Any | None = None,
    ):
        self.llm = llm
        self.reports = reports
        self.responses = responses
        self._results: ConcernResults | None = None
        self._response_results: ResponseResults | None = None

    def build_prompt(self, concerns_section: str) -> str:
        """Build a detailed prompt to split a concerns section into items."""

        return (
            "You are assisting with Prevention of Future Deaths (PFD) reports in "
            "England and Wales. Each report contains a 'Matters of Concern' "
            "section where the coroner sets out specific risks that could lead "
            "to future deaths.\n\n"
            "Your task is to read the concerns section below and rewrite it as "
            "an ordered list of concise, discrete concerns. Produce one concern "
            "per materially different issue and keep the ordering from the "
            "source text.\n\n"
            "Guidance for identifying concerns:\n"
            "- Focus on explicit risks, failings, or missing safeguards that the "
            "coroner highlights.\n"
            "- Use 1–2 sentences per concern and keep the phrasing neutral.\n"
            "- Preserve the responsible party or context when provided (e.g. a "
            "police force, hospital, housing provider).\n"
            "- Ignore narrative background that does not state a distinct risk.\n"
            "- Combine duplicate wording that refers to the same underlying risk, "
            "but do not merge unrelated issues.\n"
            "- If no clear concerns are present, return an empty list.\n\n"
            "Only populate the `concerns` field of the response schema with the "
            "final list. Do not invent information beyond the supplied text.\n\n"
            "Concerns section to analyse:\n"
            "The Coroner's Concerns text is below:\n\n"
            + concerns_section.strip()
        )

    def extract_concerns(self, concerns_section: str) -> ConcernSet:
        """Return structured concerns for a single PFD report."""

        prompt = self.build_prompt(concerns_section)
        result = self.llm.generate(
            prompts=[prompt],
            response_format=ConcernSet,
            tqdm_extra_kwargs={"desc": "Pulling out discrete concerns"},
        )[0]
        if isinstance(result, ConcernSet):
            return result
        raise ValueError(f"Unexpected LLM response: {result}")

    def extract_many(self, concerns_sections: Sequence[str]) -> List[ConcernSet]:
        """Batch-process multiple concerns sections."""

        prompts = [self.build_prompt(section) for section in concerns_sections]
        outputs = self.llm.generate(
            prompts=prompts,
            response_format=ConcernSet,
            tqdm_extra_kwargs={"desc": "Pulling out discrete concerns"},
        )
        parsed: List[ConcernSet] = []
        for idx, result in enumerate(outputs):
            if isinstance(result, ConcernSet):
                parsed.append(result)
            else:
                raise ValueError(f"Unexpected LLM response at index {idx}: {result}")
        return parsed

    def parse_concerns(
        self, output: str = "json"
    ) -> ConcernResults | list[dict[str, Any]] | pd.DataFrame:
        """Parse discrete concerns from Matters of Concern text in PFD reports.

        Parameters
        ----------
        output : {"json", "json_str", "dataframe", "object"}
            Controls the return type. ``"json"`` (default) returns a list of
            JSON-serialisable dictionaries. ``"dataframe"`` returns a pandas
            DataFrame with one row per concern. ``"object"`` returns the
            :class:`ConcernResults` instance for further manipulation.
        """

        report_records = self._records_from_table(self.reports)
        self._response_results = None
        self._validate_report_columns(report_records)
        response_lookup = self._response_lookup(self.responses)

        urls: List[str] = []
        ids: List[str | None] = []
        recipients_list: List[List[str]] = []
        concern_sections: List[str] = []
        for record in report_records:
            url = str(record.get(self.REPORT_URL_COLUMN, ""))
            report_id = self._normalise_optional(record.get(self.REPORT_ID_COLUMN))
            recipients = self._split_recipients(record.get(self.REPORT_RECEIVER_COLUMN))
            section = str(record.get(self.REPORT_CONCERNS_COLUMN, "") or "").strip()
            urls.append(url)
            ids.append(report_id)
            recipients_list.append(recipients)
            concern_sections.append(section)

        concern_sets = self.extract_many(concern_sections)
        paired: List[ReportConcerns] = []
        for url, report_id, recipients, concern_set in zip(
            urls, ids, recipients_list, concern_sets
        ):
            paired.append(
                ReportConcerns(
                    url=url,
                    report_id=report_id,
                    recipients=recipients,
                    concern_set=concern_set,
                    responses=response_lookup.get(url, []),
                )
            )

        self._results = ConcernResults(reports=paired)

        if output == "json":
            return self._results.as_list()
        if output == "json_str":
            return self._results.as_json()
        if output == "dataframe":
            return self._results.as_df()
        if output == "object":
            return self._results

        raise ValueError(
            "output must be one of {'json', 'json_str', 'dataframe', 'object'}"
        )

    def build_response_prompt(
        self,
        concerns: Sequence[ConcernItem],
        responses: Sequence[str],
        recipients: Sequence[str],
    ) -> str:
        """Build a prompt to map concerns to action phrases in responses."""

        concern_lines = "\n".join(f"- {item.concern}" for item in concerns)
        response_block = "\n\n---\n\n".join(r.strip() for r in responses if r.strip())
        response_block = response_block or "[no response text supplied]"
        recipient_line = "; ".join(recipients) if recipients else "[not specified]"

        return (
            "You are assisting with Prevention of Future Deaths (PFD) reports in "
            "England and Wales. Each concern below has already been distilled "
            "from a coroner's report. You are given response documents that may "
            "address these concerns.\n\n"
            "Your task is to link each concern to short action phrases describing "
            "what the response author says they will do. Keep the concern text "
            "exactly as provided and only include action phrases that directly "
            "address a listed concern. The adequacy of the actions is out of "
            "scope—simply capture and group them.\n\n"
            "Guidance for extracting actions:\n"
            "- Treat each response author (person or organisation) as a parent and "
            "list their discrete action phrases beneath them.\n"
            "- Prefer 5–25 word action phrases that summarise commitments or steps.\n"
            "- Ignore narrative background or text unrelated to the concerns.\n"
            "- If a concern has no matching response, use author '[no response]' "
            "with a single action phrase '[no response]'.\n"
            "- Do not invent actions or add concerns beyond those supplied.\n\n"
            "Report recipients (for context):\n"
            f"{recipient_line}\n\n"
            "Concerns to map:\n"
            f"{concern_lines}\n\n"
            "Response text:\n"
            f"{response_block}"
        )

    def parse_responses(
        self, output: str = "json"
    ) -> ResponseResults | list[dict[str, Any]] | pd.DataFrame:
        """Link parsed concerns to discrete response actions.

        Parameters
        ----------
        output : {"json", "json_str", "dataframe", "object"}
            Controls the return type. ``"json"`` (default) returns a list of
            JSON-serialisable dictionaries. ``"dataframe"`` returns a pandas
            DataFrame with one row per action phrase. ``"object"`` returns the
            :class:`ResponseResults` instance for further manipulation.
        """

        if self._results is None:
            raise ValueError("parse_concerns() must be run before parse_responses()")

        reports = self._results.reports
        if not reports:
            self._response_results = ResponseResults(reports=[])
            if output == "json":
                return []
            if output == "json_str":
                return "[]"
            if output == "dataframe":
                return pd.DataFrame()
            if output == "object":
                return self._response_results
            raise ValueError(
                "output must be one of {'json', 'json_str', 'dataframe', 'object'}"
            )

        prompts = [
            self.build_response_prompt(r.concerns, r.responses, r.recipients)
            for r in reports
        ]
        outputs = self.llm.generate(
            prompts=prompts,
            response_format=ConcernResponseSet,
            tqdm_extra_kwargs={"desc": "Finding actions for each concern"},
        )

        paired_reports: List[ReportConcernResponses] = []
        for idx, (report, output_set) in enumerate(zip(reports, outputs)):
            if not isinstance(output_set, ConcernResponseSet):
                raise ValueError(
                    f"Unexpected LLM response at index {idx}: {output_set}"
                )
            aligned = self._align_responses_to_concerns(report.concerns, output_set)
            paired_reports.append(
                ReportConcernResponses(
                    url=report.url,
                    report_id=report.report_id,
                    recipients=report.recipients,
                    responses=report.responses,
                    concern_responses=aligned,
                )
            )

        self._response_results = ResponseResults(reports=paired_reports)

        if output == "json":
            return self._response_results.as_list()
        if output == "json_str":
            return self._response_results.as_json()
        if output == "dataframe":
            return self._response_results.as_df()
        if output == "object":
            return self._response_results

        raise ValueError(
            "output must be one of {'json', 'json_str', 'dataframe', 'object'}"
        )

    def _align_responses_to_concerns(
        self, concerns: Sequence[ConcernItem], parsed: ConcernResponseSet
    ) -> ConcernResponseSet:
        """Ensure every concern has at least one response placeholder."""

        lookup: dict[str, ConcernResponseItem] = {
            item.concern: item for item in parsed.concerns
        }
        aligned: List[ConcernResponseItem] = []
        for concern in concerns:
            candidate = lookup.get(concern.concern)
            if candidate:
                aligned.append(
                    ConcernResponseItem(
                        concern=concern.concern,
                        responses=self._ensure_actions(candidate.responses),
                    )
                )
            else:
                aligned.append(
                    ConcernResponseItem(
                        concern=concern.concern,
                        responses=[self._no_response_author()],
                    )
                )
        return ConcernResponseSet(concerns=aligned)

    def _ensure_actions(
        self, responses: Sequence[ResponseAuthor] | None
    ) -> List[ResponseAuthor]:
        """Guarantee at least one action phrase per response author."""

        ensured: List[ResponseAuthor] = []
        for response in responses or []:
            actions = list(response.action_phrases) or [
                ActionPhrase(action_phrase="[no response]")
            ]
            ensured.append(
                ResponseAuthor(author=response.author, action_phrases=actions)
            )
        if not ensured:
            ensured.append(self._no_response_author())
        return ensured

    def _no_response_author(self) -> ResponseAuthor:
        return ResponseAuthor(
            author="[no response]",
            action_phrases=[ActionPhrase(action_phrase="[no response]")],
        )

    def _records_from_table(self, table: Iterable[Mapping[str, Any]] | Any) -> List[Mapping[str, Any]]:
        """Convert a DataFrame-like object or iterable of mappings into records."""

        if table is None:
            return []

        to_dict = getattr(table, "to_dict", None)
        if callable(to_dict):
            try:
                return list(to_dict(orient="records"))
            except TypeError:
                # Fall back to iterating if orient argument is unsupported
                pass

        return list(table)

    def _validate_report_columns(self, records: List[Mapping[str, Any]]) -> None:
        required = {
            self.REPORT_URL_COLUMN,
            self.REPORT_CONCERNS_COLUMN,
            self.REPORT_RECEIVER_COLUMN,
            self.REPORT_ID_COLUMN,
        }
        if not records:
            return
        missing = required - set(records[0].keys())
        if missing:
            raise ValueError(
                f"Reports are missing required columns: {', '.join(sorted(missing))}"
            )

    def _response_lookup(
        self, responses: Iterable[Mapping[str, Any]] | Any | None
    ) -> dict[str, List[str]]:
        """Build a mapping from report URL to response texts."""

        lookup: dict[str, List[str]] = {}
        for record in self._records_from_table(responses):
            parent_url = str(record.get(self.RESPONSES_PARENT_URL_COLUMN, "")).strip()
            text = str(record.get(self.RESPONSES_TEXT_COLUMN, "")).strip()
            if not parent_url or not text:
                continue
            lookup.setdefault(parent_url, []).append(text)
        return lookup

    def _normalise_optional(self, value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    def _split_recipients(self, value: Any) -> List[str]:
        raw = str(value or "")
        recipients = [segment.strip() for segment in raw.split(";") if segment.strip()]
        return recipients

    def count(self, target: str = "all") -> dict[str, Any]:
        """Count reports, responses, concerns, or all supported entities.

        Parameters
        ----------
        target : {"reports", "responses", "concerns", "action_phrases", "all"}
            Selects what to count. ``"reports"`` returns the number of PFD
            reports supplied to the parser. ``"responses"`` counts response
            documents and includes mean, median, and maximum responses per
            report. ``"concerns"`` counts discrete concerns and requires that
            :meth:`parse_concerns` has been run. ``"action_phrases"`` counts
            discrete action phrases and requires that :meth:`parse_responses`
            has been run. ``"all"`` returns a combined dictionary of all
            metrics.
        """

        target = target.lower()
        report_records = self._records_from_table(self.reports)
        response_records = self._records_from_table(self.responses)

        def report_count() -> dict[str, int]:
            return {"reports": len(report_records)}

        def response_counts() -> dict[str, Any]:
            total_responses = len(response_records)
            per_report = []
            if report_records:
                lookup = self._response_lookup(self.responses)
                per_report = [len(lookup.get(str(r.get(self.REPORT_URL_COLUMN, "")), [])) for r in report_records]
            mean = float(pd.Series(per_report).mean()) if per_report else 0.0
            median = float(pd.Series(per_report).median()) if per_report else 0.0
            maximum = int(max(per_report)) if per_report else 0
            return {
                "responses": total_responses,
                "mean_responses_per_report": mean,
                "median_responses_per_report": median,
                "max_responses_per_report": maximum,
            }

        def concern_counts() -> dict[str, int]:
            if self._results is None:
                raise ValueError("parse_concerns() must be run before counting concerns")
            total_concerns = sum(len(report.concerns) for report in self._results.reports)
            return {"concerns": total_concerns}

        def action_phrase_counts(require_parsed: bool = True) -> dict[str, int]:
            if self._response_results is None:
                if require_parsed:
                    raise ValueError(
                        "parse_responses() must be run before counting action phrases"
                    )
                return {"action_phrases": 0}
            total_actions = 0
            for report in self._response_results.reports:
                for item in report.concern_responses.concerns:
                    for response in item.responses:
                        total_actions += len(response.action_phrases)
            return {"action_phrases": total_actions}

        if target == "reports":
            return report_count()
        if target == "responses":
            return response_counts()
        if target == "concerns":
            return concern_counts()
        if target == "action_phrases":
            return action_phrase_counts()
        if target == "all":
            metrics: dict[str, Any] = {}
            metrics.update(report_count())
            metrics.update(response_counts())
            metrics.update(concern_counts())
            metrics.update(action_phrase_counts(require_parsed=False))
            return metrics

        raise ValueError(
            "target must be one of {'reports', 'responses', 'concerns', 'action_phrases', 'all'}"
        )

"""LLM-powered concern pairing for Prevention of Future Deaths reports.

This module focuses on converting the free-text "Matters of Concern" section
into a structured list of concise, discrete concerns suitable for downstream
pairing and analysis.
"""
from __future__ import annotations

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


class ConcernSet(BaseModel):
    """Ordered collection of concerns extracted from a PFD report."""

    concerns: List[ConcernItem] = Field(
        default_factory=list,
        description=(
            "A list of discrete concerns, kept in the same order as the "
            "original Matters of Concern section."
        ),
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
        )[0]
        if isinstance(result, ConcernSet):
            return result
        raise ValueError(f"Unexpected LLM response: {result}")

    def extract_many(self, concerns_sections: Sequence[str]) -> List[ConcernSet]:
        """Batch-process multiple concerns sections."""

        prompts = [self.build_prompt(section) for section in concerns_sections]
        outputs = self.llm.generate(prompts=prompts, response_format=ConcernSet)
        parsed: List[ConcernSet] = []
        for idx, result in enumerate(outputs):
            if isinstance(result, ConcernSet):
                parsed.append(result)
            else:
                raise ValueError(f"Unexpected LLM response at index {idx}: {result}")
        return parsed

    def parse_concerns(self, output: str = "json") -> ConcernResults | list[dict[str, Any]] | pd.DataFrame:
        """Parse discrete concerns from Matters of Concern text in PFD reports.

        Parameters
        ----------
        output : {"json", "dataframe", "object"}
            Controls the return type. ``"json"`` (default) returns a list of
            JSON-serialisable dictionaries. ``"dataframe"`` returns a pandas
            DataFrame with one row per concern. ``"object"`` returns the
            :class:`ConcernResults` instance for further manipulation.
        """

        report_records = self._records_from_table(self.reports)
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
        if output == "dataframe":
            return self._results.as_df()
        if output == "object":
            return self._results

        raise ValueError("output must be one of {'json', 'dataframe', 'object'}")

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
        target : {"reports", "responses", "concerns", "all"}
            Selects what to count. ``"reports"`` returns the number of PFD
            reports supplied to the parser. ``"responses"`` counts response
            documents and includes mean, median, and maximum responses per
            report. ``"concerns"`` counts discrete concerns and requires that
            :meth:`parse_concerns` has been run. ``"all"`` returns a combined
            dictionary of all metrics.
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

        if target == "reports":
            return report_count()
        if target == "responses":
            return response_counts()
        if target == "concerns":
            return concern_counts()
        if target == "all":
            metrics: dict[str, Any] = {}
            metrics.update(report_count())
            metrics.update(response_counts())
            metrics.update(concern_counts())
            return metrics

        raise ValueError("target must be one of {'reports', 'responses', 'concerns', 'all'}")

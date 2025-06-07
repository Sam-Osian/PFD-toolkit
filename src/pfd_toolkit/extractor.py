"""LLM-powered feature extraction from PFD reports."""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Optional, Type, Union, Literal

import pandas as pd
from pydantic import BaseModel, Field, create_model

from .llm import LLM
from .config import GeneralConfig

logger = logging.getLogger(__name__)


class Extractor:
    """Extract custom features from Prevention of Future Death reports using an LLM.

    Parameters
    ----------
    llm : LLM
        Instance of :class:`~pfd_toolkit.llm.LLM` used for prompting.
    feature_model : type[pydantic.BaseModel]
        Pydantic model describing the features to extract from each report.
        The model field names will be used as the JSON keys in the LLM response.
    reports : pandas.DataFrame, optional
        DataFrame of PFD reports. If provided, it will be copied and stored
        on the instance.
    include_date, include_coroner, include_area, include_receiver,
    include_investigation, include_circumstances, include_concerns : bool, optional
        Flags controlling which existing report columns are included in the text
        sent to the LLM.
    verbose : bool, optional
        Emit extra logging when ``True``.
    force_assign : bool, optional
        When ``True``, the LLM is instructed to avoid returning
        :data:`GeneralConfig.NOT_FOUND_TEXT` for any feature. Defaults to ``False``.
    allow_multiple : bool, optional
        Allow a report to be assigned to multiple categories when ``True``.
        Defaults to ``False``.
    schema_detail : Literal["full", "minimal"], optional
        Level of detail for the feature schema included in the prompt.
        ``"full"`` includes the entire Pydantic model schema while
        ``"minimal"`` uses only field names and descriptions. Defaults to ``"minimal"``.
    extra_instructions : str, optional
        Extra instructions injected into every prompt, placed before the schema
        line. Use this to provide additional context or rules for the LLM.
    """

    COL_URL = GeneralConfig.COL_URL
    COL_ID = GeneralConfig.COL_ID
    COL_DATE = GeneralConfig.COL_DATE
    COL_CORONER_NAME = GeneralConfig.COL_CORONER_NAME
    COL_AREA = GeneralConfig.COL_AREA
    COL_RECEIVER = GeneralConfig.COL_RECEIVER
    COL_INVESTIGATION = GeneralConfig.COL_INVESTIGATION
    COL_CIRCUMSTANCES = GeneralConfig.COL_CIRCUMSTANCES
    COL_CONCERNS = GeneralConfig.COL_CONCERNS

    def __init__(
        self,
        *,
        llm: LLM,
        feature_model: Type[BaseModel],
        reports: Optional[pd.DataFrame] = None,
        include_date: bool = False,
        include_coroner: bool = False,
        include_area: bool = False,
        include_receiver: bool = False,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True,
        verbose: bool = False,
        force_assign: bool = False,
        allow_multiple: bool = False,
        schema_detail: Literal["full", "minimal"] = "minimal",
        extra_instructions: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.feature_model = feature_model
        self.include_date = include_date
        self.include_coroner = include_coroner
        self.include_area = include_area
        self.include_receiver = include_receiver
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns
        self.verbose = verbose
        self.force_assign = force_assign
        self.allow_multiple = allow_multiple
        self.schema_detail = schema_detail
        self.extra_instructions = extra_instructions

        self.reports: pd.DataFrame = (
            reports.copy() if reports is not None else pd.DataFrame()
        )


        self.feature_names = self._collect_field_names()

        self._feature_schema = self._build_feature_schema(schema_detail)
        self.prompt_template = self._build_prompt_template()
        self._grammar_model = self._build_grammar_model()

        # cache mapping prompt -> feature dict
        self._cache: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    def _build_prompt_template(self) -> str:
        not_found_line_prompt = (
            f"If a feature cannot be located, respond with '{GeneralConfig.NOT_FOUND_TEXT}'.\n"
            if not self.force_assign
            else ""
        )
        category_line = (
            "A report may belong to multiple categories; separate them with semicolons (;).\n"
            if self.allow_multiple
            else "Assign only one category to each report."
        )

        extra_instr = (self.extra_instructions.strip() + "\n") if self.extra_instructions else ""

        template = f"""
You are an expert at extracting structured information from UK Prevention of Future Death reports.

Extract the following features from the report excerpt provided.

{not_found_line_prompt}
{category_line}
{extra_instr}

Return your answer strictly as a JSON object matching this schema:\n
{{schema}}

Here is the report excerpt:

{{report_excerpt}}
"""
        return template.strip()

    # ------------------------------------------------------------------
    def _extend_feature_model(self) -> Type[BaseModel]:
        """Return a feature model extended with optional fields."""
        base_fields = self.feature_model.model_fields

        fields = {}
        for name, field in base_fields.items():
            field_type = field.annotation
            alias = getattr(field, "alias", name)
            description = getattr(field, "description", None)
            fields[name] = (
                field_type,
                Field(..., alias=alias, description=description),
            )

        return create_model(f"{self.feature_model.__name__}Extended", **fields)

    # ------------------------------------------------------------------
    def _collect_field_names(self) -> List[str]:
        """Return a list of feature names from the model."""
        return list(self.feature_model.model_fields.keys())

    # ------------------------------------------------------------------

    def _build_feature_schema(self, detail: str) -> str:
        """Return a JSON schema string for the feature model."""
        properties = (
            self._extend_feature_model().model_json_schema().get("properties", {})
        )
        if detail == "minimal":
            properties = {
                name: {"type": info.get("type"), "description": info.get("description")}
                for name, info in properties.items()
            }

        return json.dumps(properties, indent=2)
    
    # ------------------------------------------------------------------
    def _build_grammar_model(self) -> Type[BaseModel]:
        """Create an internal Pydantic model allowing missing features.

        This helper builds a new model identical to ``feature_model`` but with
        each field accepting either the original type or ``str``.  This ensures
        that the LLM can return :data:`GeneralConfig.NOT_FOUND_TEXT` for any
        field regardless of its declared type.
        """

        base_fields = self.feature_model.model_fields

        fields = {}
        for name, field in base_fields.items():
            field_type = field.annotation
            alias = getattr(field, "alias", name)
            if self.force_assign:
                union_type = field_type
            else:
                union_type = Union[field_type, str]
            fields[name] = (union_type, Field(..., alias=alias))

        return create_model("ExtractorLLMModel", **fields)

    # ------------------------------------------------------------------
    def _generate_prompt(self, row: pd.Series) -> str:
        """Construct a single prompt for the given DataFrame row."""
        parts: List[str] = []
        if self.include_date and pd.notna(row.get(self.COL_DATE)):
            parts.append(f"{self.COL_DATE}: {row[self.COL_DATE]}")
        if self.include_coroner and pd.notna(row.get(self.COL_CORONER_NAME)):
            parts.append(f"{self.COL_CORONER_NAME}: {row[self.COL_CORONER_NAME]}")
        if self.include_area and pd.notna(row.get(self.COL_AREA)):
            parts.append(f"{self.COL_AREA}: {row[self.COL_AREA]}")
        if self.include_receiver and pd.notna(row.get(self.COL_RECEIVER)):
            parts.append(f"{self.COL_RECEIVER}: {row[self.COL_RECEIVER]}")
        if self.include_investigation and pd.notna(row.get(self.COL_INVESTIGATION)):
            parts.append(f"{self.COL_INVESTIGATION}: {row[self.COL_INVESTIGATION]}")
        if self.include_circumstances and pd.notna(row.get(self.COL_CIRCUMSTANCES)):
            parts.append(f"{self.COL_CIRCUMSTANCES}: {row[self.COL_CIRCUMSTANCES]}")
        if self.include_concerns and pd.notna(row.get(self.COL_CONCERNS)):
            parts.append(f"{self.COL_CONCERNS}: {row[self.COL_CONCERNS]}")
        report_text = "\n\n".join(str(p) for p in parts).strip()
        report_text = " ".join(report_text.split())
        prompt = self.prompt_template.format(
            report_excerpt=report_text, schema=self._feature_schema
        )

        return prompt

    # ------------------------------------------------------------------
    def extract_features(
        self, reports: Optional[pd.DataFrame] = None, *, skip_if_present: bool = True
    ) -> pd.DataFrame:
        """Run feature extraction for the given reports.

        Parameters
        ----------
        reports : pandas.DataFrame, optional
            DataFrame of reports to process. Defaults to the instance's stored
            reports if omitted.
        skip_if_present : bool, optional
            When ``True`` (default), skip rows when any feature column already
            holds a non-missing value that is not equal to
            :data:`GeneralConfig.NOT_FOUND_TEXT`. This assumes the row has been
            processed previously and prevents another LLM call.
        """
        df = reports.copy() if reports is not None else self.reports.copy()
        if df.empty:
            return df

        # ensure result columns exist
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = GeneralConfig.NOT_FOUND_TEXT

        prompts: List[str] = []
        indices: List[int] = []
        keys: List[str] = []

        for idx, row in df.iterrows():
            if skip_if_present:
                present = any(
                    pd.notna(row.get(f))
                    and row.get(f) != GeneralConfig.NOT_FOUND_TEXT
                    for f in self.feature_names
                )
                if present:
                    # If any feature column already contains data we assume
                    #   the row was cached from a previous run and skip calling
                    #   the LLM again.
                    continue

            prompt = self._generate_prompt(row)
            key = prompt
            if key in self._cache:
                cached = self._cache[key]
                for feat in self.feature_names:
                    df.at[idx, feat] = cached.get(feat, GeneralConfig.NOT_FOUND_TEXT)
                continue

            prompts.append(prompt)
            indices.append(idx)
            keys.append(key)

        if self.verbose:
            logger.info(
                "Sending %s prompts to LLM for feature extraction", len(prompts)
            )

        llm_results: List[BaseModel | Dict[str, object] | str] = []
        if prompts:
            llm_results = self.llm.generate_batch(
                prompts=prompts,
                response_format=self._grammar_model,
                tqdm_extra_kwargs={
                    "desc": "Extracting features",
                    "position": 0,
                    "leave": True,
                },
            )

        for i, res in enumerate(llm_results):
            idx = indices[i]
            key = keys[i]
            values: Dict[str, object] = {}
            if isinstance(res, BaseModel):
                values = res.model_dump()
            elif isinstance(res, dict):
                values = res
            else:
                logger.error("LLM returned unexpected result type: %s", type(res))
                values = {f: GeneralConfig.NOT_FOUND_TEXT for f in self.feature_names}

            for feat in self.feature_names:
                val = values.get(feat, GeneralConfig.NOT_FOUND_TEXT)
                df.at[idx, feat] = val

            self._cache[key] = values

        if reports is None:
            self.reports = df.copy()
        return df

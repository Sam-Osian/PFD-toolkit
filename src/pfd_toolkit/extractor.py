"""LLM-powered feature extraction from PFD reports."""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Optional, Type, Union

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
    """

    COL_URL = GeneralConfig.COL_URL
    COL_ID = GeneralConfig.COL_ID
    COL_DATE = GeneralConfig.COL_DATE
    COL_CORONER_NAME = GeneralConfig.COL_CORONER_NAME
    COL_AREA = GeneralConfig.COL_AREA
    COL_RECEIVER = GeneralConfig.COL_URL
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

        self.reports: pd.DataFrame = (
            reports.copy() if reports is not None else pd.DataFrame()
        )


        self.feature_names = self._collect_field_names()

        self.prompt_template = self._build_prompt_template()
        self._llm_model = self._build_llm_model()

    # ------------------------------------------------------------------
    def _build_prompt_template(self) -> str:
        features_list = ", ".join(self.feature_names)
        not_found_line_prompt = (
            f"If a feature cannot be located, respond with '{GeneralConfig.NOT_FOUND_TEXT}'."
            if not self.force_assign
            else ""
        )
        category_line = (
            "A report may belong to multiple categories; separate them with semicolons (;)."
            if self.allow_multiple
            else "Assign only one category to each report."
        )

        template = f"""
You are an expert at extracting structured information from UK Prevention of Future Death reports.

Extract the following features from the report excerpt provided.

{not_found_line_prompt}
{category_line}

Return your answer strictly as a JSON object with the following keys:
{features_list}
\n*\n\nHere is the report excerpt:

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
    def _build_llm_model(self) -> Type[BaseModel]:
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
        prompt = self.prompt_template.format(report_excerpt=report_text)
        json_template = json.dumps({name: "" for name in self.feature_names}, indent=2)
        return f"{prompt}\n\n{json_template}"

    # ------------------------------------------------------------------
    def extract_features(self, reports: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run feature extraction for the given reports."""
        df = reports.copy() if reports is not None else self.reports.copy()
        if df.empty:
            return df

        # ensure result columns exist
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = GeneralConfig.NOT_FOUND_TEXT

        prompts: List[str] = []
        indices: List[int] = []

        for idx, row in df.iterrows():
            prompts.append(self._generate_prompt(row))
            indices.append(idx)

        if self.verbose:
            logger.info(
                "Sending %s prompts to LLM for feature extraction", len(prompts)
            )

        llm_results = self.llm.generate_batch(
            prompts=prompts,
            response_format=self._llm_model,
            tqdm_extra_kwargs={
                "desc": "Extracting features",
                "position": 0,
                "leave": True,
            },
        )

        for i, res in enumerate(llm_results):
            idx = indices[i]
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

        if reports is None:
            self.reports = df.copy()
        return df

"""LLM-powered feature extraction from PFD reports."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel

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
    feature_instructions : dict[str, str]
        Mapping of feature names (matching the model fields) to the extraction
        instructions presented to the LLM.
    reports : pandas.DataFrame, optional
        DataFrame of PFD reports. If provided, it will be copied and stored
        on the instance.
    result_columns : dict[str, str] or None, optional
        Mapping of feature names to DataFrame column names. If omitted, the
        feature names themselves are used as column names.
    include_date, include_coroner, include_area, include_receiver,
    include_investigation, include_circumstances, include_concerns : bool, optional
        Flags controlling which existing report columns are included in the text
        sent to the LLM.
    verbose : bool, optional
        Emit extra logging when ``True``.
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
        feature_instructions: Dict[str, str],
        reports: Optional[pd.DataFrame] = None,
        result_columns: Optional[Dict[str, str]] = None,
        include_date: bool = False,
        include_coroner: bool = False,
        include_area: bool = False,
        include_receiver: bool = False,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.feature_model = feature_model
        self.feature_instructions = feature_instructions
        self.result_columns = result_columns or {k: k for k in feature_instructions}
        self.include_date = include_date
        self.include_coroner = include_coroner
        self.include_area = include_area
        self.include_receiver = include_receiver
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns
        self.verbose = verbose

        self.reports: pd.DataFrame = reports.copy() if reports is not None else pd.DataFrame()

        self.prompt_template = self._build_prompt_template()

    # ------------------------------------------------------------------
    def _build_prompt_template(self) -> str:
        instructions_lines = [
            f"- {name}: {desc}" for name, desc in self.feature_instructions.items()
        ]
        features_list = ", ".join(self.feature_instructions.keys())
        template = f"""
You are an expert at extracting structured information from UK Prevention of Future Death reports.

Extract the following features from the report excerpt provided.

If a feature cannot be located, respond with '{GeneralConfig.NOT_FOUND_TEXT}'.

Return your answer strictly as a JSON object with the following keys: 
{features_list}

Feature guidance:
{chr(10).join(instructions_lines)}

Here is the report excerpt:

{{report_excerpt}}
"""
        return template.strip()

    # ------------------------------------------------------------------
    def extract_features(self, reports: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run feature extraction for the given reports."""
        df = reports.copy() if reports is not None else self.reports.copy()
        if df.empty:
            return df

        # ensure result columns exist
        for feat, col in self.result_columns.items():
            if col not in df.columns:
                df[col] = GeneralConfig.NOT_FOUND_TEXT

        prompts: List[str] = []
        indices: List[int] = []

        for idx, row in df.iterrows():
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
            prompts.append(self.prompt_template.format(report_excerpt=report_text))
            indices.append(idx)

        if self.verbose:
            logger.info("Sending %s prompts to LLM for feature extraction", len(prompts))

        llm_results = self.llm.generate_batch(
            prompts=prompts,
            response_format=self.feature_model,
            tqdm_extra_kwargs={"desc": "LLM: Extracting features", "position": 0, "leave": True},
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
                values = {f: GeneralConfig.NOT_FOUND_TEXT for f in self.feature_instructions}

            for feat in self.feature_instructions:
                col = self.result_columns[feat]
                val = values.get(feat, GeneralConfig.NOT_FOUND_TEXT)
                df.at[idx, col] = val

        if reports is None:
            self.reports = df.copy()
        return df

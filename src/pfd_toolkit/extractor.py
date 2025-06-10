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
    feature_model : type[pydantic.BaseModel]
        Pydantic model describing the features to extract from each report.
        The model field names will be used as the JSON keys in the LLM response
    llm : LLM
        Instance of :class:`~pfd_toolkit.llm.LLM` used for prompting..
    reports : pandas.DataFrame, optional
        DataFrame of PFD reports. If provided, it will be copied and stored
        on the instance.
    include_date, include_coroner, include_area, include_receiver, include_investigation, include_circumstances, include_concerns : bool, optional
        Flags controlling which existing report columns are included in the text
        sent to the LLM.
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
    verbose : bool, optional
        Emit extra logging when ``True``.
    """

    # Load column names from config.py
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
        feature_model: Optional[Type[BaseModel]] = None,
        llm: LLM,
        reports: Optional[pd.DataFrame] = None,
        include_date: bool = False,
        include_coroner: bool = False,
        include_area: bool = False,
        include_receiver: bool = False,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True,
        force_assign: bool = False,
        allow_multiple: bool = False,
        schema_detail: Literal["full", "minimal"] = "minimal",
        extra_instructions: Optional[str] = None,
        verbose: bool = False,
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
        self.force_assign = force_assign
        self.allow_multiple = allow_multiple
        self.schema_detail = schema_detail
        self.extra_instructions = extra_instructions
        self.verbose = verbose

        # Default summary column name used by Cleaner.summarise
        self.summary_col = "summary"

        self.reports: pd.DataFrame = (
            reports.copy() if reports is not None else pd.DataFrame()
        )

        # Prepare feature metadata and prompt template
        if feature_model:
            self.feature_names = self._collect_field_names()
            self._feature_schema = self._build_feature_schema(schema_detail)
            self.prompt_template = self._build_prompt_template()
            self._grammar_model = self._build_grammar_model()
        else:
            self.feature_names = []
            self._feature_schema = ""
            self.prompt_template = ""
            self._grammar_model = None

        if verbose: # ...debug logging of initialisation internals
            logger.debug("Feature names: %r", self.feature_names)
            logger.debug("Feature schema: %s", self._feature_schema)
            logger.debug("Prompt template: %s", self.prompt_template)
            logger.debug("Grammar (Pydantic) model: %r", self._grammar_model)

        
        # Cache mapping prompt -> feature dict
        self.cache: Dict[str, Dict[str, object]] = {}
        # Token estimates for columns
        self.token_cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------
    def _build_prompt_template(self) -> str:
        """Optional instructions depending on ``self.force_assign`` and
        ``self.allow_multiple`` parameters."""
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
        # Include any extra user instructions if provided
        extra_instr = (self.extra_instructions.strip() + "\n") if self.extra_instructions else ""
        
        # Compose the full template
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
        """Return a feature model mirroring ``feature_model`` with all fields
        required."""
        base_fields = self.feature_model.model_fields

        fields = {}
        for name, field in base_fields.items():
            field_type = field.annotation
            alias = field.alias
            description = field.description
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
        if detail == "minimal": # ...only include type and description for brevity
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
            alias = field.alias
            if self.force_assign:
                union_type = field_type
            else: # ...allow str fallback if force_assign is False
                union_type = Union[field_type, str]
            fields[name] = (union_type, Field(..., alias=alias))

        return create_model("ExtractorLLMModel", **fields)

    # ------------------------------------------------------------------
    def _generate_prompt(self, row: pd.Series) -> str:
        """Construct a single prompt for the given DataFrame row."""
        parts: List[str] = []
        # Add each enabled section if value present
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
            processed previously and is logged in an instance of `Extractor.cache`
        """
        df = reports.copy() if reports is not None else self.reports.copy()
        if df.empty:
            return df

        # Ensure result columns exist with default NOT_FOUND_TEXT
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = GeneralConfig.NOT_FOUND_TEXT

        prompts: List[str] = []
        indices: List[int] = []
        keys: List[str] = []

        # For each row needing extraction, build prompt
        for idx, row in df.iterrows():
            if skip_if_present:
                present = any(
                    pd.notna(row.get(f))
                    and row.get(f) != GeneralConfig.NOT_FOUND_TEXT
                    for f in self.feature_names
                )
                if present:
                        # If any feature column already contains data, assume
                        # the row was cached from a previous run, and skip
                    continue

            prompt = self._generate_prompt(row)
            key = prompt
            if key in self.cache: # ...use cached values if available
                cached = self.cache[key]
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
        # Call LLM ...
        llm_results: List[BaseModel | Dict[str, object] | str] = []
        if prompts:
            llm_results = self.llm.generate(
                prompts=prompts,
                response_format=self._grammar_model,
                tqdm_extra_kwargs={
                    "desc": "Extracting features",
                    "position": 0,
                    "leave": True,
                },
            )
        # Parse and store results
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

            self.cache[key] = values # ...cache response for reuse

        if reports is None: # ...update stored reports in instance
            self.reports = df.copy()
        return df

    # ------------------------------------------------------------------

    def summarise(
        self,
        result_col_name: str = "summary",
        trim_intensity: str = "medium",
    ) -> pd.DataFrame:
        """Summarise selected report fields into one column using the LLM.

        Parameters
        ----------
        result_col_name : str, optional
            Name of the summary column. Defaults to ``"summary"``.
        trim_intensity : {"low", "medium", "high", "very high"}, optional
            Controls how concise the summary should be. Defaults to ``"medium"``.

        Returns
        ------- 
        pandas.DataFrame
            A new DataFrame identical to the one provided at initialisation with
            an extra summary column.
        """
        if trim_intensity not in {"low", "medium", "high"}:
            raise ValueError("trim_intensity must be 'low', 'medium', 'high', or 'very high'")

        self.summary_col = result_col_name
        summary_df = self.reports.copy()

        instructions = {
            "low": "write a fairly detailed paragraph summarising the report.",
            "medium": "write a concise summary of the key points of the report.",
            "high": "write a very short summary of the report.",
            "very high": "write a one or two sentence summary of the report."
        }
        base_prompt = (
            "You are an assistant summarising UK Prevention of Future Death reports."
            "You will be given an exerpt from one report.\n\n"
            "Your task is to "
            + instructions[trim_intensity]
            + "\n\nDo not provide any commentary or headings; simply summarise the report."
            + "Always use British English. Do not re-write acronyms to full form."
            + "\n\nReport exerpt:\n\n"
        )

        fields = [
            (self.include_coroner, self.COL_CORONER_NAME, "Coroner name"),
            (self.include_area, self.COL_AREA, "Area"),
            (self.include_receiver, self.COL_RECEIVER, "Receiver"),
            (self.include_investigation, self.COL_INVESTIGATION, "Investigation and Inquest"),
            (self.include_circumstances, self.COL_CIRCUMSTANCES, "Circumstances of Death"),
            (self.include_concerns, self.COL_CONCERNS, "Matters of Concern"),
        ]

        prompts = []
        idx_order = []
        for idx, row in summary_df.iterrows():
            parts = []
            for flag, col, label in fields:
                if flag and col in summary_df.columns:
                    val = row.get(col)
                    if pd.notna(val):
                        parts.append(f"{label}: {str(val)}")
            if not parts:
                prompts.append(base_prompt + "\nN/A")
            else:
                text = " ".join(str(p) for p in parts)
                prompts.append(base_prompt + "\n" + text)
            idx_order.append(idx)

        if prompts:
            results = self.llm.generate(
                prompts=prompts,
                tqdm_extra_kwargs={"desc": "Summarising reports", "leave": False},
            )
        else:
            results = []

        summary_series = pd.Series(index=idx_order, dtype=object)
        for i, res in enumerate(results):
            summary_series.loc[idx_order[i]] = res

        summary_df[result_col_name] = summary_series
        self.summarised_reports = summary_df
        return summary_df


    # ------------------------------------------------------------------
    def estimate_tokens(
        self, 
        col_name: Optional[str] = None,
        return_series: Optional[bool] = False
    ) -> Union[int, pd.Series]:
        """Estimate token counts for a columnsummarise( using :mod:`tiktoken`.

        Parameters
        ----------
        summary_col : str, optional
            Name of the column containing report summaries. Defaults to
            :pyattr:`summary_col`.
        return_series : bool, optional
            Returns a pandas.Series of per-row token counts for that field
            if ``True``, or an integer if ``False``. Defaults to ``false``.

        Returns
        -------
        Union[int, pandas.Series]
            If `return_series` is `False`, returns an `int` representing the total sum
            of all token counts across all rows for the provided field.
            If `return_series` is `True`, returns a `pandas.Series` of token counts
            aligned to :pyattr:`self.reports` for the provided field.
        
        """
        
        # Check if summarise() has been run; throw error if not
        if not hasattr(self, 'summarised_reports'):
            raise AttributeError(
                "The 'summarised_reports' attribute does not exist. "
                "Please run the `summarise()` method before estimating tokens."
            )
        
        col = col_name or self.summary_col
        
        if col not in self.summarised_reports.columns:
            raise ValueError(
                f"Column '{col}' not found in reports. "
                f"Did you run `summarise()` with a different `result_col_name`?"
            )

        texts = self.summarised_reports[col].fillna("").astype(str).tolist()
        counts = self.llm.estimate_tokens(texts)
        series = pd.Series(counts, index=self.reports.index, name=f"{col}_tokens")

        self.token_cache[col] = counts
        
        if return_series:
            return series
        else:
            total_sum = series.sum()
            return total_sum.item()


    # ------------------------------------------------------------------
    def export_cache(self, path: str = "extractor_cache.pkl") -> str:
        """Save the current cache to ``path``.

        Parameters
        ----------
        path : str, optional
            Full path to the cache file including the filename. If ``path`` is a
            directory, ``extractor_cache.pkl`` will be created inside it.

        Returns
        -------
        str
            The path to the written cache file.
        """

        from pathlib import Path
        import pickle

        file_path = Path(path)
        # Handle directory paths by appending default filename
        if file_path.is_dir() or file_path.suffix == "":
            file_path.mkdir(parents=True, exist_ok=True)
            file_path = file_path / "extractor_cache.pkl"
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump({"features": self.cache, "tokens": self.token_cache}, f)
        return str(file_path)

    # ------------------------------------------------------------------
    def import_cache(self, path: str = "extractor_cache.pkl") -> None:
        """Load cache from ``path``.

        Parameters
        ----------
        path : str, optional
            Full path to the cache file including the filename. If ``path`` is a
            directory, ``extractor_cache.pkl`` will be loaded from inside it.
        """

        from pathlib import Path
        import pickle

        file_path = Path(path)
        if file_path.is_dir() or file_path.suffix == "":
            file_path = file_path / "extractor_cache.pkl"

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "features" in data:
            self.cache = data.get("features", {})
            self.token_cache = data.get("tokens", {})
        else:
            # backwards compatibility with older cache files
            self.cache = data
            self.token_cache = {}


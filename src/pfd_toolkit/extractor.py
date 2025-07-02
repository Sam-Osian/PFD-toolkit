"""LLM-powered feature extraction from PFD reports."""

from __future__ import annotations

import logging
import json
import re
import warnings
from typing import Dict, List, Optional, Type, Union, Literal

import pandas as pd
from pydantic import BaseModel, Field, create_model, ConfigDict

from .llm import LLM
from .config import GeneralConfig

logger = logging.getLogger(__name__)


class Extractor:
    """Extract custom features from Prevention of Future Death reports using an
    LLM.

    Parameters
    ----------
    llm : LLM
        Instance of the ``LLM`` helper used for prompting.
    reports : pandas.DataFrame, optional
        DataFrame of PFD reports. When provided it is copied and stored on the
        instance. Defaults to ``None``.
    include_date : bool, optional
        Include the ``date`` column in prompts. Defaults to ``False``.
    include_coroner : bool, optional
        Include the ``coroner`` column in prompts. Defaults to ``False``.
    include_area : bool, optional
        Include the ``area`` column in prompts. Defaults to ``False``.
    include_receiver : bool, optional
        Include the ``receiver`` column in prompts. Defaults to ``False``.
    include_investigation : bool, optional
        Include the ``investigation`` column in prompts. Defaults to
        ``True``.
    include_circumstances : bool, optional
        Include the ``circumstances`` column in prompts. Defaults to
        ``True``.
    include_concerns : bool, optional
        Include the ``concerns`` column in prompts. Defaults to ``True``.
    verbose : bool, optional
        Emit extra logging when ``True``. Defaults to ``False``.
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
        llm: LLM,
        reports: Optional[pd.DataFrame] = None,
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
        self.feature_model: Optional[Type[BaseModel]] = None
        self._base_feature_model: Optional[Type[BaseModel]] = None
        self.include_date = include_date
        self.include_coroner = include_coroner
        self.include_area = include_area
        self.include_receiver = include_receiver
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns
        self.force_assign = False
        self.allow_multiple = False
        self.schema_detail: Literal["full", "minimal"] = "minimal"
        self.extra_instructions: Optional[str] = None
        self.verbose = verbose
        self.produce_spans = False

        # Default summary column name used by Cleaner.summarise
        self.summary_col = "summary"

        self.reports: pd.DataFrame = (
            reports.copy() if reports is not None else pd.DataFrame()
        )

        # No feature model defined until ``extract_features`` or
        # ``discover_themes`` is called
        self.feature_names: List[str] = []
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
        # Store raw LLM output from discover_themes for debugging
        self.identified_themes = None

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
        span_line = (
            "Text spans should be as concise as possible, but always **verbatim** from the source."
            "Wrap each text span in quotation marks. "
            "If multiple spans are found, separate them with semicolons (;).\n"
            if self.produce_spans
            else ""
        )
        # Include any extra user instructions if provided
        extra_instr = (self.extra_instructions.strip() + "\n") if self.extra_instructions else ""

        # Compose the full template
        template = f"""
You are an expert at extracting structured information from UK Prevention of Future Death reports.

Extract the following features from the report excerpt provided.

{not_found_line_prompt}
{category_line}
{span_line}
{extra_instr}

Return your answer strictly as a JSON object matching this schema:\n
{{schema}}

Here is the report excerpt:

{{report_excerpt}}
"""
        return template.strip()

    # ------------------------------------------------------------------
    def _add_span_fields(self, model: Type[BaseModel]) -> Type[BaseModel]:
        """Return a new model with ``spans_`` prefixed fields added."""

        base_fields = model.model_fields
        fields = {}
        for name, field in base_fields.items():
            if name.startswith("spans_"):
                fields[name] = (
                    field.annotation,
                    Field(..., alias=field.alias, description=field.description),
                )
                continue
            span_name = f"spans_{name}"
            fields[span_name] = (
                str,
                Field(
                    ...,
                    alias=span_name,
                    description=f"Text spans supporting {name}",
                ),
            )
            fields[name] = (
                field.annotation,
                Field(..., alias=field.alias, description=field.description),
            )

        return create_model(
            f"{model.__name__}WithSpans",
            **fields,
            __config__=ConfigDict(extra="forbid"),
        )

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

        return create_model(
            f"{self.feature_model.__name__}Extended",
            **fields,
            __config__=ConfigDict(extra="forbid"),
        )

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
            else:  # allow str or None when force_assign is False
                union_type = Union[field_type, str, None]
            fields[name] = (union_type, Field(..., alias=alias))

        return create_model(
            "ExtractorLLMModel",
            **fields,
            __config__=ConfigDict(extra="forbid"),
        )


    # ------------------------------------------------------------------
    def _generate_prompt(self, row: pd.Series) -> str:
        """Construct a single prompt for the given DataFrame row."""
        parts: List[str] = []
        fields = [
            (self.include_date, self.COL_DATE),
            (self.include_coroner, self.COL_CORONER_NAME),
            (self.include_area, self.COL_AREA),
            (self.include_receiver, self.COL_RECEIVER),
            (self.include_investigation, self.COL_INVESTIGATION),
            (self.include_circumstances, self.COL_CIRCUMSTANCES),
            (self.include_concerns, self.COL_CONCERNS),
        ]

        for flag, col in fields:
            if not flag:
                continue
            val = row.get(col)
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        report_text = "\n\n".join(str(p) for p in parts).strip()
        report_text = " ".join(report_text.split())
        prompt = self.prompt_template.format(
            report_excerpt=report_text, schema=self._feature_schema
        )

        return prompt

    # ------------------------------------------------------------------
    def extract_features(
        self,
        reports: Optional[pd.DataFrame] = None,
        *,
        feature_model: Optional[Type[BaseModel]] = None,
        produce_spans: bool = False,
        drop_spans: bool = False,
        force_assign: bool = False,
        allow_multiple: bool = False,
        schema_detail: Literal["full", "minimal"] = "minimal",
        extra_instructions: Optional[str] = None,
        skip_if_present: bool = True,
    ) -> pd.DataFrame:
        """Run feature extraction for the given reports.

        Parameters
        ----------
        reports : pandas.DataFrame, optional
            DataFrame of reports to process. Defaults to the instance's stored
            reports if omitted.
        feature_model : type[pydantic.BaseModel], optional
            Pydantic model describing the features to extract. Must be provided
            on first call or after calling ``discover_themes``.
        produce_spans : bool, optional
            When ``True``, create ``spans_`` versions of each feature to capture
            the supporting text snippets. Defaults to ``False``.
        drop_spans : bool, optional
            When ``True`` and ``produce_spans`` is also ``True``, remove any
            ``spans_`` columns created during this call from the returned
            DataFrame. Spans columns from other sources (e.g. Screener) are
            preserved. If ``produce_spans`` is ``False`` a warning is emitted and
            no columns are dropped. Defaults to ``False``.
        force_assign : bool, optional
            When ``True``, the LLM is instructed to avoid returning
            :data:`GeneralConfig.NOT_FOUND_TEXT` for any feature.
        allow_multiple : bool, optional
            Allow a report to be assigned to multiple categories when ``True``.
        schema_detail : {"full", "minimal"}, optional
            Level of detail for the feature schema included in the prompt.
        extra_instructions : str, optional
            Additional instructions injected into each prompt before the schema.
        skip_if_present : bool, optional
            When ``True`` (default), skip rows when any feature column already
            holds a non-missing value that is not equal to
            :data:`GeneralConfig.NOT_FOUND_TEXT`. This assumes the row has been
            processed previously and is logged in an instance of `Extractor.cache`
        """
        # Update feature extraction configuration
        if feature_model is not None:
            self._base_feature_model = feature_model
        if self._base_feature_model is None:
            raise ValueError("feature_model must be provided")

        self.produce_spans = produce_spans
        if self.produce_spans:
            self.feature_model = self._add_span_fields(self._base_feature_model)
        else:
            self.feature_model = self._base_feature_model
        self.force_assign = force_assign
        self.allow_multiple = allow_multiple
        self.schema_detail = schema_detail
        self.extra_instructions = extra_instructions

        self.feature_names = self._collect_field_names()
        self._feature_schema = self._build_feature_schema(self.schema_detail)
        self.prompt_template = self._build_prompt_template()
        self._grammar_model = self._build_grammar_model()
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
            prompt = self._generate_prompt(row)
            key = prompt

            if skip_if_present:
                has_data = any(pd.notna(row.get(f)) for f in self.feature_names)
                if has_data and key in self.cache:
                    # Row previously processed; reuse cached values
                    cached = self.cache[key]
                    for feat in self.feature_names:
                        df.at[idx, feat] = cached.get(feat, GeneralConfig.NOT_FOUND_TEXT)
                    continue

            if key in self.cache:  # ...use cached values if available
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
                if feat.startswith("spans_"):
                    if not isinstance(val, str) or not val.strip():
                        val = GeneralConfig.NOT_FOUND_TEXT
                df.at[idx, feat] = val

            self.cache[key] = values # ...cache response for reuse

        result_df = df
        if drop_spans:
            if not self.produce_spans:
                logger.warning(
                    "drop_spans=True has no effect because produce_spans=False"
                )
                warnings.warn(
                    "drop_spans=True has no effect because produce_spans=False",
                    UserWarning,
                )
            else:
                span_cols = [
                    feat
                    for feat in self.feature_names
                    if feat.startswith("spans_") and feat in result_df.columns
                ]
                if span_cols:
                    result_df = result_df.drop(columns=span_cols)

        if reports is None: # ...update stored reports in instance
            self.reports = result_df.copy()
        return result_df

    # ------------------------------------------------------------------

    def summarise(
        self,
        result_col_name: str = "summary",
        trim_intensity: str = "medium",
        extra_instructions: Optional[str] = None,
    ) -> pd.DataFrame:
        """Summarise selected report fields into one column using the LLM.

        Parameters
        ----------
        result_col_name : str, optional
            Name of the summary column. Defaults to ``"summary"``.
        trim_intensity : {"low", "medium", "high", "very high"}, optional
            Controls how concise the summary should be. Defaults to ``"medium"``.
        extra_instructions : str, optional
            Additional instructions to append to the prompt before the report
            excerpt.

        Returns
        ------- 
        pandas.DataFrame
            A new DataFrame identical to the one provided at initialisation with
            an extra summary column.
        """
        if trim_intensity not in {"low", "medium", "high", "very high"}:
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
            "You will be given an excerpt from one report.\n\n"
            "Your task is to "
            + instructions[trim_intensity]
            + "\n\nDo not provide any commentary or headings; simply summarise the report."
            + "Always use British English. Do not re-write acronyms to full form."
        )
        if extra_instructions:
            base_prompt += "\n\n" + extra_instructions.strip()
        base_prompt += "\n\nReport excerpt:\n\n"

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
        """Estimate token counts for all rows of a given column using
        the ``tiktoken`` library.

        Parameters
        ----------
        col_name : str, optional
            Name of the column containing report summaries. Defaults to
            ``summary_col``, which is generated after running ``summarise``.
        return_series : bool, optional
            Returns a pandas.Series of per-row token counts for that field
            if ``True``, or an integer if ``False``. Defaults to ``False``.

        Returns
        -------
        Union[int, pandas.Series]
            If `return_series` is `False`, returns an `int` representing the total sum
            of all token counts across all rows for the provided field.
            If ``return_series`` is ``True``, returns a ``pandas.Series`` of
            token counts aligned to ``self.reports`` for the provided field.
        
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
    def discover_themes(
        self,
        *,
        warn_exceed: int = 100000,
        error_exceed: int = 500000,
        max_themes: Optional[int] = None,
        min_themes: Optional[int] = None,
        extra_instructions: Optional[str] = None,
        seed_topics: Optional[Union[str, List[str], BaseModel]] = None,
    ) -> Type[BaseModel]:
        """Use an LLM to automatically discover report themes.

        The method expects ``summarise`` to have been run so that a summary
        column exists. All summaries are concatenated into one prompt sent to
        the LLM. The LLM should return a JSON object mapping theme names to
        descriptions. A new ``pydantic`` model is built from this mapping and
        stored as ``feature_model``.

        Parameters
        ----------
        warn_exceed : int, optional
            Emit a warning if the estimated token count exceeds this value.
            Defaults to ``100000``.
        error_exceed : int, optional
            Raise a ``ValueError`` if the estimated token count exceeds this
            value. Defaults to ``500000``.
        max_themes : int or None, optional
            Instruct the LLM to identify no more than this number of themes when
            provided.
        min_themes : int or None, optional
            Instruct the LLM to identify at least this number of themes when
            provided.
        extra_instructions : str, optional
            Additional instructions appended to the theme discovery prompt.
        seed_topics : str | list[str] | pydantic.BaseModel, optional
            Optional seed topics to include in the prompt. These are treated as
            starting suggestions and the model should incorporate them into a
            broader list of themes.

        Returns
        -------
        type[pydantic.BaseModel]
            The generated feature model containing discovered themes.
        """

        if not hasattr(self, "summarised_reports"):
            raise AttributeError(
                "Please run `summarise()` before calling discover_themes()."
            )

        if self.summary_col not in self.summarised_reports.columns:
            raise ValueError(
                f"Column '{self.summary_col}' not found in summarised reports."
            )

        if self.summary_col in self.token_cache:
            total_tokens = sum(self.token_cache[self.summary_col])
        else:
            total_tokens = self.estimate_tokens(col_name=self.summary_col)
            
        if total_tokens > error_exceed:
            raise ValueError(
                f"Token estimate {total_tokens} exceeds error threshold {error_exceed}."
            )
        if total_tokens > warn_exceed:
            logger.warning(
                "Estimated token count %s exceeds warning threshold %s",
                total_tokens,
                warn_exceed,
            )

        summaries = self.summarised_reports[self.summary_col].fillna("")
        combined_text = "\n".join(str(s) for s in summaries)

        prompt = (
            "You are analysing UK Prevention of Future Death report summaries. "
            "Your task is to identify a small but cohesive set of commonly "
            "recurring themes across report summaries.\n\n "
        )
        if extra_instructions:
            prompt += extra_instructions.strip() + "\n\n"
        if seed_topics:
            if isinstance(seed_topics, BaseModel):
                seed_text = seed_topics.model_dump_json(indent=2)
            elif isinstance(seed_topics, str):
                seed_text = seed_topics
            else:
                seed_text = ", ".join(str(t) for t in seed_topics)
            prompt += (
                "The following seed topics were provided by the user. "
                "They are only suggestions; please incorporate them into a "
                "broader list of themes as appropriate:\n"
                f"{seed_text}\n\n"
            )
        if max_themes is not None:
            prompt += f"Identify no more than **{max_themes} themes.** "
        if min_themes is not None:
            prompt += f"Identify at least **{min_themes} themes.** "
        prompt += (
            "\n\nRespond ONLY with a JSON object mapping short theme names suitable "
            "as Python identifiers to a brief description of that theme. Identifiers "
            "must be no more than 2 words and must be in snake case.\n\n"
            "Denote each theme as being of type bool.\n\n"
            "Do not provide nested themes; there should be only one 'tier'.\n\n"
            "Here are the report summaries:\n\n" + combined_text
        )

        result = self.llm.generate([prompt])[0]
        self.identified_themes = result

        if isinstance(result, dict):
            theme_dict = result
        else:
            raw = str(result)
            try:
                theme_dict = json.loads(raw)
                if not isinstance(theme_dict, dict):
                    raise ValueError("LLM output is not a JSON object")
            except Exception as exc:
                fenced = re.match(r"^```(?:json)?\n(?P<json>.*)\n```$", raw.strip(), re.DOTALL)
                if fenced:
                    try:
                        theme_dict = json.loads(fenced.group("json"))
                        if not isinstance(theme_dict, dict):
                            raise ValueError("LLM output is not a JSON object")
                    except Exception as exc2:
                        logger.warning(
                            "Failed to parse theme JSON: %s. Raw output: %r",
                            exc2,
                            result,
                        )
                        return self.feature_model
                else:
                    logger.warning(
                        "Failed to parse theme JSON: %s. Raw output: %r",
                        exc,
                        result,
                    )
                    return self.feature_model

        fields = {
            name: (bool, Field(description=str(desc)))
            for name, desc in theme_dict.items()
        }
        ThemeModel = create_model(
            "DiscoveredThemes", **fields, __config__=ConfigDict(extra="forbid")
        )

        self.feature_model = ThemeModel
        self._base_feature_model = ThemeModel
        self.feature_names = self._collect_field_names()
        self._feature_schema = self._build_feature_schema(self.schema_detail)
        self.prompt_template = self._build_prompt_template()
        self._grammar_model = self._build_grammar_model()
        return ThemeModel


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
            pickle.dump({"features": self.cache, "tokens": self.token_cache, 
                         "summary_col": self.summary_col}, f)
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
            self.summary_col = data.get("summary_col", {})
        else:
            # backwards compatibility with older cache files
            self.cache = data
            self.token_cache = {}


    # ------------------------------------------------------------------
    def reset(self) -> "Extractor":
        """Reset internal caches and intermediate state.

        This clears any cached feature extraction results and token
        estimations so that ``extract_features`` can be run again on
        the same reports. The instance itself is returned to allow method
        chaining, e.g. ``extractor.reset().extract_features()``.
        """

        self.cache.clear()
        self.token_cache.clear()
        self.identified_themes = None
        return self


    # ------------------------------------------------------------------
    def tabulate(
        self,
        columns: Optional[Union[str, List[str]]] = None,
        labels: Optional[Union[str, List[str]]] = None,
        *,
        count_col: str = "Count",
        pct_col: str = "Percentage",
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return a simple frequency table for extracted feature columns.

        Parameters
        ----------
        columns : str or list[str], optional
            Column name or list of column names to summarise. Defaults to all
            feature columns added by :meth:`extract_features` (excluding any
            ``spans_`` columns).
        labels : str or list[str], optional
            Human friendly label or list of labels corresponding to ``columns``.
            If omitted, column names are used.
        count_col, pct_col : str, optional
            Column names for the count and percentage values in the output
            DataFrame. Defaults to ``"Count"`` and ``"Percentage"``.
        df : pandas.DataFrame, optional
            DataFrame containing the columns to tabulate. Defaults to the
            reports stored on the instance.

        Returns
        -------
        pandas.DataFrame
            A DataFrame summarising the frequencies of the specified columns.

        Raises
        ------
        RuntimeError
            If :meth:`extract_features` has not been run yet.
        """

        if not self.feature_names:
            raise RuntimeError("extract_features() must be run before tabulate()")

        if df is None:
            df = self.reports

        if columns is None:
            columns = [c for c in self.feature_names if not c.startswith("spans_") and c in df.columns]
        elif isinstance(columns, str):
            columns = [columns]

        if labels is None:
            labels = columns
        elif isinstance(labels, str):
            labels = [labels]

        if len(columns) != len(labels):
            raise ValueError("labels must match the number of columns")

        total = len(df)
        rows: List[Dict[str, object]] = []

        for col, label in zip(columns, labels):
            if col not in df.columns:
                raise KeyError(f"Column {col!r} not found in DataFrame")

            series = df[col]
            non_na = series.dropna()

            unique_vals = set(non_na.unique())
            bool_like = unique_vals.issubset({True, False, 1, 0}) or series.dtype == bool

            if bool_like:
                count = int((series == True).sum())
                percentage = (count / total * 100) if total else 0.0
                rows.append({"Category": label, count_col: count, pct_col: percentage})
                continue

            value_counts = non_na.value_counts()
            for val, count in value_counts.items():
                row_label = f"{label}: {val}"
                percentage = (count / total * 100) if total else 0.0
                rows.append({"Category": row_label, count_col: int(count), pct_col: percentage})

        return pd.DataFrame(rows)


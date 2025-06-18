from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from ..text_utils import normalise_date

logger = logging.getLogger(__name__)


def run_llm_fallback(
    df: pd.DataFrame,
    *,
    llm,
    pdf_extractor,
    llm_field_config: List[Tuple[bool, str, str, str]],
    llm_to_df_mapping: Dict[str, str],
    col_url: str,
    not_found_text: object,
    llm_key_date: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Apply LLM fallback extraction to a DataFrame of reports."""

    # Skip early if there are no rows
    if df.empty:
        logger.info("Report DataFrame is empty. Skipping LLM fallback.")
        return df

    def _process_row(idx: int, row_data: pd.Series) -> Tuple[int, Dict[str, str]]:
        # Build dict of fields still missing
        missing_fields: Dict[str, str] = {}
        for include_flag, df_col_name, llm_key, llm_prompt in llm_field_config:
            if include_flag and pd.isna(row_data.get(df_col_name)):
                missing_fields[llm_key] = llm_prompt
        if not missing_fields:
            return idx, {}

        pdf_bytes = None
        report_url = row_data.get(col_url)
        if report_url:
            pdf_bytes = pdf_extractor.fetch_pdf_bytes(report_url)
        if not pdf_bytes and verbose:
            logger.warning(
                "Could not obtain PDF bytes for URL %s (row %s). LLM fallback for this row might be impaired.",
                report_url,
                idx,
            )

        updates = llm._call_llm_fallback(
            pdf_bytes=pdf_bytes,
            missing_fields=missing_fields,
            report_url=str(report_url) if report_url else "N/A",
            verbose=verbose,
            tqdm_extra_kwargs={"disable": True},
        )
        return idx, updates if updates else {}

    results_map: Dict[int, Dict[str, str]] = {}
    # Use threads when the LLM client supports it
    use_parallel = llm and hasattr(llm, "max_workers") and llm.max_workers > 1
    if use_parallel:
        # Process rows concurrently
        with ThreadPoolExecutor(max_workers=llm.max_workers) as executor:
            future_to_idx = {
                executor.submit(_process_row, idx, row): idx
                for idx, row in df.iterrows()
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="LLM scraping",
                position=0,
                leave=True,
            ):
                idx = future_to_idx[future]
                try:
                    _, updates = future.result()
                    results_map[idx] = updates
                except Exception as e:
                    logger.error("LLM fallback failed for row index %s: %s", idx, e)
    else:
        # Fallback to sequential processing
        for idx, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="LLM fallback (sequential processing)",
            position=0,
            leave=True,
        ):
            try:
                _, updates = _process_row(idx, row)
                results_map[idx] = updates
            except Exception as e:
                logger.error("LLM fallback failed for row index %s: %s", idx, e)

    # Write LLM results back into the DataFrame
    for idx, updates_dict in results_map.items():
        if not updates_dict:
            continue
        for llm_key, value_from_llm in updates_dict.items():
            df_col_name = llm_to_df_mapping.get(llm_key)
            if df_col_name:
                if llm_key == llm_key_date:
                    if not pd.isna(value_from_llm):
                        df.at[idx, df_col_name] = normalise_date(value_from_llm, verbose=verbose)
                else:
                    df.at[idx, df_col_name] = value_from_llm

    return df

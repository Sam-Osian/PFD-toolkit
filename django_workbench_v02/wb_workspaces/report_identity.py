from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

REPORT_IDENTITY_COLUMN = "__report_identity"

_PRIORITY_COLUMNS = (
    "report_url",
    "url",
    "link",
    "uri",
    "report_id",
    "id",
)

_FALLBACK_COLUMNS = (
    "date",
    "coroner",
    "area",
    "receiver",
    "investigation",
    "concerns",
)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"nan", "nat", "none", "null"}:
        return ""
    return text


def report_identity_from_mapping(payload: dict[str, Any]) -> str:
    for key in _PRIORITY_COLUMNS:
        value = _clean(payload.get(key))
        if value:
            return value

    compact = {key: _clean(payload.get(key)) for key in _FALLBACK_COLUMNS}
    digest = hashlib.sha256(
        json.dumps(compact, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def with_report_identities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        result = df.copy()
        if REPORT_IDENTITY_COLUMN not in result.columns:
            result[REPORT_IDENTITY_COLUMN] = pd.Series(dtype="object")
        return result

    result = df.copy()
    identities: list[str] = []
    for _, row in result.iterrows():
        payload = {str(column): row[column] for column in result.columns}
        identities.append(report_identity_from_mapping(payload))

    result[REPORT_IDENTITY_COLUMN] = identities
    return result

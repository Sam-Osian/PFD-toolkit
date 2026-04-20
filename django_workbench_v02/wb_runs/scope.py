from __future__ import annotations

from copy import deepcopy

from wb_workspaces.models import WorkspaceReportExclusion


FILTER_KEYS = ("coroner", "area", "receiver")


def _clean_text(value) -> str:
    return str(value or "").strip()


def _clean_text_list(values) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for item in values:
        text = _clean_text(item)
        if text:
            result.append(text)
    return result


def _normalise_selected_filters(raw) -> dict[str, list[str]]:
    source = raw if isinstance(raw, dict) else {}
    return {
        key: _clean_text_list(source.get(key))
        for key in FILTER_KEYS
    }


def _scope_text(scope_json: dict, key: str) -> str:
    return _clean_text(scope_json.get(key))


def resolve_run_scope_config(*, investigation, input_config_json: dict | None) -> dict:
    """
    Build a deterministic run scope for all run types.

    Precedence contract:
    - Explicit run config wins for scope keys if provided.
    - Investigation scope_json backfills missing scope keys.
    - Workspace exclusions are always enforced from current workbook state.
    """
    resolved = deepcopy(input_config_json) if isinstance(input_config_json, dict) else {}
    scope_json = (
        investigation.scope_json
        if isinstance(investigation.scope_json, dict)
        else {}
    )

    if not _clean_text(resolved.get("collection_slug")):
        collection_slug = _scope_text(scope_json, "collection_slug")
        if collection_slug:
            resolved["collection_slug"] = collection_slug

    if not _clean_text(resolved.get("collection_query")):
        collection_query = _scope_text(scope_json, "collection_query")
        if collection_query:
            resolved["collection_query"] = collection_query

    if "selected_filters" in resolved:
        resolved["selected_filters"] = _normalise_selected_filters(resolved.get("selected_filters"))
    elif isinstance(scope_json.get("selected_filters"), dict):
        resolved["selected_filters"] = _normalise_selected_filters(scope_json.get("selected_filters"))

    if "report_identity_allowlist" in resolved:
        resolved["report_identity_allowlist"] = _clean_text_list(resolved.get("report_identity_allowlist"))
    elif isinstance(scope_json.get("report_identity_allowlist"), list):
        resolved["report_identity_allowlist"] = _clean_text_list(scope_json.get("report_identity_allowlist"))

    excluded_identities = [
        text
        for text in (
            _clean_text(item)
            for item in WorkspaceReportExclusion.objects.filter(workspace=investigation.workspace)
            .values_list("report_identity", flat=True)
        )
        if text
    ]
    if excluded_identities:
        resolved["excluded_report_identities"] = excluded_identities
        resolved["excluded_report_count"] = len(excluded_identities)
    else:
        resolved.pop("excluded_report_identities", None)
        resolved.pop("excluded_report_count", None)

    return resolved

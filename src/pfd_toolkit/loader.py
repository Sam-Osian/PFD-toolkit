from __future__ import annotations

from typing import Final

from pathlib import Path
from threading import Lock
import requests

import pandas as pd
from dateutil import parser as _date_parser
from pfd_toolkit.collections import COLLECTION_COLUMNS as _COLLECTION_COLUMNS

# Path within the package used for tests
_DATA_FILE: Final[str] = "all_reports.csv"

# Location of the latest dataset release
_DATA_URL: Final[str] = (
    "https://github.com/Sam-Osian/PFD-toolkit/releases/download/"
    "dataset-latest/all_reports.csv"
)

# Cache path for the downloaded dataset
_CACHE_DIR: Final[Path] = Path.home() / ".cache" / "pfd_toolkit"
_CACHE_FILE: Final[Path] = _CACHE_DIR / _DATA_FILE
_DATAFRAME_CACHE_LOCK: Final[Lock] = Lock()
_DATAFRAME_CACHE_SIGNATURE: tuple[str, int, int] | None = None
_DATAFRAME_CACHE_FRAME: pd.DataFrame | None = None
_THEME_PREFIX: Final[str] = "theme_"
_THEME_COLLECTION_NAME_OVERRIDES: Final[dict[str, str]] = {
    "suicide_risk": "suicide",
    "care_home_safety": "care_home",
}
_COLLECTION_ALIASES: Final[dict[str, str]] = {
    "welsh": "wales",
    "health_reg": "health_regulators",
    "suicide_risk": "suicide",
    "care_home_safety": "care_home",
}
_EXCLUDED_THEME_COLLECTIONS: Final[set[str]] = {
    "nutrition",
}


def _normalise_collection_name(raw_value: str) -> str:
    value = str(raw_value or "").strip()
    return _COLLECTION_ALIASES.get(value, value)

def _ensure_cached_dataset(
    cache_file: Path,
    *,
    dataset_url: str,
    force_download: bool = False,
) -> Path:
    if force_download and cache_file.exists():
        cache_file.unlink()

    if not cache_file.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(dataset_url, timeout=30)
            resp.raise_for_status()
            cache_file.write_bytes(resp.content)
        except Exception as exc:  # pragma: no cover - network failure
            raise FileNotFoundError(
                "Failed to download dataset from GitHub release"
            ) from exc
    return cache_file


def _ensure_dataset(force_download: bool = False) -> Path:
    """Download the base dataset if not already cached and return its path."""
    return _ensure_cached_dataset(
        _CACHE_FILE,
        dataset_url=_DATA_URL,
        force_download=force_download,
    )


def _reset_dataframe_cache() -> None:
    global _DATAFRAME_CACHE_SIGNATURE, _DATAFRAME_CACHE_FRAME
    with _DATAFRAME_CACHE_LOCK:
        _DATAFRAME_CACHE_SIGNATURE = None
        _DATAFRAME_CACHE_FRAME = None


def _load_base_reports(force_download: bool = False) -> pd.DataFrame:
    global _DATAFRAME_CACHE_SIGNATURE, _DATAFRAME_CACHE_FRAME

    csv_path = _ensure_dataset(force_download=force_download)
    stat = csv_path.stat()
    signature = (str(csv_path.resolve()), stat.st_mtime_ns, stat.st_size)

    with _DATAFRAME_CACHE_LOCK:
        if _DATAFRAME_CACHE_SIGNATURE != signature or _DATAFRAME_CACHE_FRAME is None:
            reports = pd.read_csv(csv_path)
            reports = reports.loc[:, ~reports.columns.str.startswith("Unnamed")]
            reports["date"] = pd.to_datetime(
                reports["date"], format="%Y-%m-%d", errors="coerce"
            )
            reports = (
                reports.dropna(subset=["date"])
                .sort_values("date", ascending=False)
                .reset_index(drop=True)
            )
            _DATAFRAME_CACHE_SIGNATURE = signature
            _DATAFRAME_CACHE_FRAME = reports

        return _DATAFRAME_CACHE_FRAME


def _get_thematic_collection_columns(reports: pd.DataFrame) -> dict[str, str]:
    """Return available thematic collections keyed by collection name."""
    theme_columns: dict[str, str] = {}
    collection_columns = set(_COLLECTION_COLUMNS.values())
    for column in reports.columns:
        if column.startswith(_THEME_PREFIX) and column not in collection_columns:
            theme_name = column[len(_THEME_PREFIX):]
            if theme_name:
                public_name = _THEME_COLLECTION_NAME_OVERRIDES.get(theme_name, theme_name)
                if public_name in _EXCLUDED_THEME_COLLECTIONS:
                    continue
                theme_columns[public_name] = column
    return theme_columns


def load_reports(
    start_date: str = "2000-01-01",
    end_date: str = "2050-01-01",
    n_reports: int | None = None,
    refresh: bool = True,
    collection: str | list[str] | None = None,
) -> pd.DataFrame:
    """Load Prevention of Future Death reports as a DataFrame.

    Parameters
    ----------
    start_date : str, optional
        Inclusive lower bound for the report date in ``YYYY-MM-DD`` format.
        Default: ``"2000-01-01"``.
    end_date : str, optional
        Inclusive upper bound for the report date in ``YYYY-MM-DD`` format.
        Dates after today are capped to the current date at runtime.
        Default: ``"2050-01-01"``.
    n_reports : int or None, optional
        Keep only the most recent ``n_reports`` rows after filtering by date.
        Default: ``None`` (all rows).
    refresh : bool, optional
        If ``True``, force a fresh download of the dataset. If ``False``, reuse
        the cached copy.
        Default: ``True``.
    collection : str | list[str] | None, optional
        One collection name or a list of collection names.
        Collection matching uses OR semantics across provided names.
        When one collection is provided, the helper boolean column is dropped from
        output. When multiple collections are provided, helper columns are kept.
        Supported collection names:

        - wales
        - nhs
        - gov_department
        - prisons
        - health_regulators
        - health_reg
        - local_gov
        - access_to_care
        - ambulance_response
        - care_home
        - discharge_planning
        - environmental_safety
        - equipment_safety
        - falls_prevention
        - family_involvement
        - hospital_care
        - infection_control
        - interagency_communication
        - medication_safety
        - mental_health_care
        - observation_failures
        - online_hazards
        - physical_health_in_mental_health
        - record_keeping
        - road_safety
        - safeguarding
        - staff_shortages
        - staff_training
        - substance_misuse
        - suicide
        - vulnerable_groups
        - emergency_departments
        - ambulance_services
        - primary_care
        - out_of_hours_care
        - acute_hospital_wards
        - intensive_care
        - surgical_care
        - maternity_neonatal_perinatal_care
        - mental_health_services
        - substance_use_services
        - care_homes
        - domiciliary_care
        - hospices_palliative_care
        - prisons_criminal_justice_supervision
        - police_custody
        - immigration_detention
        - secure_health_settings
        - housing_homelessness
        - universities
        - workplaces
        - roads_highways
        - rail_settings
        - domestic_settings
        - suicide_self_harm
        - drug_related_deaths
        - alcohol_related_deaths
        - polypharmacy
        - diagnostic_delay
        - sepsis_infection
        - cancer_care
        - cardiovascular_conditions
        - respiratory_conditions
        - neurological_conditions
        - diabetes_metabolic_conditions
        - falls_frailty
        - choking_aspiration
        - learning_disability
        - autism
        - cognitive_impairment
        - substance_dependence
        - domestic_abuse
        - self_neglect
        - violence_homicide_related_systems_failures
        - environmental_hazards
        - epilepsy_seizure_management
        - allergy_anaphylaxis
        - ligature_anchor_point_risks
        - failure_recognise_escalate_deterioration
        - communication_failures
        - handover_failures
        - record_sharing_failures
        - referral_failures
        - follow_up_failures
        - transitions_discharge_failures
        - observation_monitoring_failures
        - test_result_management_failures
        - consent_decision_making_failures
        - capacity_best_interests_failures
        - staffing_shortages_workload_pressure
        - training_competence_gaps
        - policy_procedure_failures
        - equipment_failures
        - it_digital_system_failures
        - alarm_alert_failures
        - environmental_design_failures
        - transport_access_barriers
        - missed_appointments_non_attendance
        - restraint_restrictive_intervention
        - delayed_admission
        - bed_shortages
        - language_interpreter_barriers
        - remote_digital_care
        - safeguarding_failures
        - inter_agency_working
        - continuity_of_care
        - family_carer_concerns_not_acted_on
        - reasonable_adjustments_not_made
        - investigation_incident_review_failures
        - failure_learn_previous_deaths_incidents
        - thresholds_eligibility_barriers
        - waiting_times_delays
        - children_young_people
        - older_people
        - people_detention_state_control
        - people_experiencing_multiple_disadvantage
        - people_living_alone_socially_isolated

    Returns
    -------
    pandas.DataFrame
        Reports filtered by date, sorted newest first and optionally
        limited to ``n_reports`` rows.

    Raises
    ------
    ValueError
        If *start_date* is after *end_date*.
    FileNotFoundError
        If the dataset cannot be downloaded.
    ValueError
        If a requested collection is not present in the packaged dataset.

    Examples
    --------
        from pfd_toolkit import load_reports
        df = load_reports(start_date="2020-01-01", end_date="2022-12-31", n_reports=100)
        df.head()
    """
    
    # Date param reading
    date_from = _date_parser.parse(start_date)
    requested_date_to = _date_parser.parse(end_date)
    today = pd.Timestamp.today().normalize()
    date_to = min(requested_date_to, today)
    if date_from > date_to:
        raise ValueError("start_date must be earlier than or equal to end_date")

    reports = _load_base_reports(force_download=refresh)
    reports = reports.loc[
        (reports["date"] >= date_from) & (reports["date"] <= date_to)
    ].copy()
    reports.reset_index(drop=True, inplace=True)

    requested_collections_raw: list[str] = []
    if collection is not None:
        requested_collections_raw.extend(
            [collection] if isinstance(collection, str) else list(collection)
        )

    if requested_collections_raw:
        requested_collections = []
        for item in requested_collections_raw:
            cleaned = _normalise_collection_name(str(item or "").strip())
            if cleaned:
                requested_collections.append(cleaned)
        requested_collections = list(dict.fromkeys(requested_collections))
        if not requested_collections:
            raise ValueError(
                "collection must contain at least one non-empty collection name."
            )

        thematic_collections = _get_thematic_collection_columns(reports)
        available_collections = {**_COLLECTION_COLUMNS, **thematic_collections}
        missing_collections = [
            item for item in requested_collections if item not in available_collections
        ]
        if missing_collections:
            available = ", ".join(sorted(available_collections)) or "none"
            raise ValueError(
                f"Unknown collection(s): {missing_collections}. Available collections are: {available}."
            )

        collection_columns = [available_collections[item] for item in requested_collections]
        missing_columns = [
            column for column in collection_columns if column not in reports.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Requested collection column(s) are not present in the packaged dataset: {missing_columns}."
            )

        collection_mask = (
            reports[collection_columns].fillna(False).astype(bool).any(axis=1)
        )
        reports = reports.loc[collection_mask].reset_index(drop=True)
        if len(requested_collections) == 1:
            reports = reports.drop(columns=collection_columns, errors="ignore")

    # Limit to n_reports
    if n_reports is not None:
        # .head(n) will return all rows if n > len(reports)
        reports = reports.head(n_reports).reset_index(drop=True)

    return reports

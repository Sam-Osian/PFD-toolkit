from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "issue_tracker_mvp"
sys.path.insert(0, str(SCRIPT_DIR))

import build_issue_index as pipeline  # noqa: E402
import adjudicate_recurring_issue_groups as adjudication  # noqa: E402
import assign_issue_prototypes as prototype_assignment  # noqa: E402
import build_issue_registry as issue_registry  # noqa: E402
import build_recurring_issue_audit as quality_audit  # noqa: E402
import normalize_issue_occurrences as normalization  # noqa: E402
import repair_reviewed_issue_groups as repair  # noqa: E402
import prepare_relational_linkage_pilot as relational_pilot  # noqa: E402
import run_relational_linkage_experiment as relational_linkage  # noqa: E402
import run_full_issue_tracker as full_workflow  # noqa: E402
import score_recurring_issue_audit as audit_scoring  # noqa: E402
import tune_issue_index as tuning  # noqa: E402


def test_targeted_risk_gate_flags_directional_and_action_conflicts():
    groups = pd.DataFrame(
        [
            {
                "subissue_id": "sub_test",
                "recurrence_status": "recurring",
                "report_count": 4,
                "issue_count": 4,
                "median_centroid_similarity": 0.97,
            }
        ]
    )
    indexed = pd.DataFrame(
        [
            {
                "subissue_id": "sub_test",
                "canonical_issue": "The service failed to escalate a referral",
                "issue_object": "referral escalation",
                "failure_state": "omitted",
                "process_stage": "referral",
                "communication_direction": "professional_to_professional",
                "responsible_actor_role": "provider organisation",
            },
            {
                "subissue_id": "sub_test",
                "canonical_issue": "The receiving team failed to process the referral",
                "issue_object": "referral processing",
                "failure_state": "delayed",
                "process_stage": "follow_up",
                "communication_direction": "between_organisations",
                "responsible_actor_role": "team",
            },
        ]
    )

    result = adjudication.build_risk_gate(
        groups, indexed, risk_threshold=3, low_cohesion=0.955
    ).iloc[0]

    assert bool(result["flagged_for_adjudication"])
    assert bool(result["direction_conflict"])
    assert bool(result["lifecycle_conflict"])
    assert bool(result["semantic_action_conflict"])


def test_recurrence_boundary_alone_does_not_trigger_adjudication():
    groups = pd.DataFrame(
        [
            {
                "subissue_id": "sub_tight",
                "recurrence_status": "recurring",
                "report_count": 3,
                "issue_count": 3,
                "median_centroid_similarity": 0.98,
            }
        ]
    )
    indexed = pd.DataFrame(
        [
            {
                "subissue_id": "sub_tight",
                "canonical_issue": "The employer omitted an automated external defibrillator",
                "issue_object": "automated external defibrillator",
                "failure_state": "omitted",
                "process_stage": "equipment_availability",
                "communication_direction": "not_applicable",
                "responsible_actor_role": "employer",
            }
            for _ in range(3)
        ]
    )

    result = adjudication.build_risk_gate(
        groups, indexed, risk_threshold=3, low_cohesion=0.955
    ).iloc[0]

    assert result["risk_reasons"] == "recurrence_boundary"
    assert result["adjudication_priority"] == "monitor"
    assert not bool(result["flagged_for_adjudication"])


def test_recurrence_strength_preserves_discovery_status_but_bands_evidence():
    assert pipeline.recurrence_strength(1) == "isolated"
    assert pipeline.recurrence_strength(2) == "emerging"
    assert pipeline.recurrence_strength(3) == "recurring_candidate"
    assert pipeline.recurrence_strength(5) == "established_recurring"
    assert pipeline.recurrence_strength(10) == "high_frequency"


def test_failed_normalization_uses_nonempty_auditable_original_fallback():
    frame = pd.DataFrame(
        [
            {
                "issue_id": "iss_1",
                "canonical_issue": "The provider organisation omitted follow-up",
                "evidence_quote": "There was no follow-up.",
                "responsible_actor_type": "provider_organisation",
                "responsible_actor_text": "The Trust",
            }
        ]
    )
    output = normalization.build_output(frame, {})
    output = normalization.apply_original_fallbacks(output, {"iss_1"})

    assert output.loc[0, "normalization_status"] == "fallback_original"
    assert output.loc[0, "responsible_actor_role"] == "provider organisation"
    assert output.loc[0, "issue_object"] == "issue described in evidence"
    assert output.loc[0, "canonical_issue"] == output.loc[
        0, "canonical_issue_original"
    ]
    assert "normalization_failed_original_preserved" in output.loc[
        0, "normalization_warnings"
    ]


def test_issue_registry_keeps_identity_when_a_cluster_expands():
    first_groups = pd.DataFrame(
        [
            {
                "subissue_id": "sub_old",
                "label": "Follow-up was omitted",
                "description": "",
                "recurrence_status": "recurring",
                "recurrence_strength": "recurring_candidate",
                "report_count": 3,
                "issue_count": 3,
            }
        ]
    )
    first_assignments = pd.DataFrame(
        {
            "subissue_id": ["sub_old"] * 3,
            "issue_id": ["iss_1", "iss_2", "iss_3"],
        }
    )
    first_types, first_members, _, _ = issue_registry.build_registry(
        first_groups,
        first_assignments,
        pd.DataFrame(),
        pd.DataFrame(columns=["issue_type_id", "issue_id"]),
        match_containment=0.5,
        minimum_overlap=2,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    issue_type_id = first_types.loc[0, "issue_type_id"]
    rerun_types, _, rerun_lineage, _ = issue_registry.build_registry(
        first_groups,
        first_assignments,
        first_types,
        first_members,
        match_containment=0.5,
        minimum_overlap=2,
        generated_at="2026-01-02T00:00:00+00:00",
    )
    assert rerun_types.loc[0, "issue_type_id"] == issue_type_id
    assert rerun_lineage.empty
    first_types.loc[0, "curation_status"] = "human_validated"
    first_types.loc[0, "publication_status"] = "published"
    second_groups = first_groups.copy()
    second_groups.loc[0, "subissue_id"] = "sub_new_membership_hash"
    second_groups.loc[0, ["report_count", "issue_count"]] = [4, 4]
    second_assignments = pd.DataFrame(
        {
            "subissue_id": ["sub_new_membership_hash"] * 4,
            "issue_id": ["iss_1", "iss_2", "iss_3", "iss_4"],
        }
    )

    second_types, _, lineage, registered = issue_registry.build_registry(
        second_groups,
        second_assignments,
        first_types,
        first_members,
        match_containment=0.5,
        minimum_overlap=2,
        generated_at="2026-02-01T00:00:00+00:00",
    )
    active = second_types[second_types["registry_status"].eq("active")].iloc[0]

    assert active["issue_type_id"] == issue_type_id
    assert active["current_cluster_snapshot_id"] == "sub_new_membership_hash"
    assert active["curation_status"] == "needs_review"
    assert active["publication_status"] == "review_required"
    assert lineage.iloc[0]["relationship"] == "expanded"
    assert set(registered["issue_type_id"]) == {issue_type_id}


def test_adjudication_split_preserves_omitted_members_as_singletons():
    members = pd.DataFrame(
        {
            "issue_id": ["iss_1", "iss_2", "iss_3"],
            "report_key": ["r1", "r2", "r3"],
        }
    )
    payload = {
        "decision": "split",
        "rationale": "Different directions",
        "exclude_issue_ids": [],
        "partitions": [
            {"label_hint": "sent", "issue_ids": ["iss_1", "iss_2"]},
            {"label_hint": "received", "issue_ids": ["iss_3"]},
        ],
    }

    decision = adjudication.validate_decision(payload, "sub_test", members)

    assert decision["decision"] == "split"
    assert len(decision["partitions"]) == 2
    payload["partitions"][0]["issue_ids"] = ["iss_1"]
    payload["partitions"][1]["issue_ids"] = ["iss_2"]
    corrected = adjudication.validate_decision(payload, "sub_test", members)
    assert corrected["partitions"][-1] == {
        "label_hint": "Outlier retained separately",
        "issue_ids": ["iss_3"],
    }
    payload["partitions"][1]["issue_ids"] = ["iss_unknown"]
    with pytest.raises(ValueError, match="duplicate or unknown"):
        adjudication.validate_decision(payload, "sub_test", members)


def test_adjudication_corrects_one_character_opaque_id_typo():
    valid = {"iss_094bb97c590acde4", "iss_1ea00d7477f086fb"}

    assert (
        adjudication.correct_opaque_issue_id("iss_094bb97590acde4", valid)
        == "iss_094bb97c590acde4"
    )
    assert adjudication.correct_opaque_issue_id("iss_unrelated", valid) == "iss_unrelated"


def test_rejected_group_becomes_singletons_without_losing_occurrences():
    issue_ids = ["iss_1", "iss_2", "iss_3"]
    decision = {
        "decision": "reject",
        "exclude_issue_ids": [],
        "partitions": [],
    }

    units = adjudication.repair_units_for_decision(issue_ids, decision)

    assert units == [
        ("rejected_singleton", ["iss_1"], ""),
        ("rejected_singleton", ["iss_2"], ""),
        ("rejected_singleton", ["iss_3"], ""),
    ]


def test_stale_adjudication_decision_cannot_change_a_group_no_longer_flagged():
    occurrences = pd.DataFrame(
        [
            {
                "issue_id": f"iss_{number}",
                "report_key": f"r{number}",
                "canonical_issue": "The provider omitted follow-up",
                "failure_state": "omitted",
                "process_stage": "follow_up",
                "responsible_actor_role": "provider organisation",
                "issue_object": "patient follow-up",
                "issue_themes": "clinical_assessment_care",
            }
            for number in range(1, 4)
        ]
    )
    groups = pd.DataFrame(
        [
            {
                "subissue_id": "sub_same",
                "label": "Follow-up omitted",
                "description": "",
                "recurrence_status": "recurring",
            }
        ]
    )
    assignments = pd.DataFrame(
        {
            "issue_id": ["iss_1", "iss_2", "iss_3"],
            "subissue_id": ["sub_same"] * 3,
        }
    )
    risk = pd.DataFrame(
        [
            {
                "subissue_id": "sub_same",
                "flagged_for_adjudication": False,
                "risk_score": 1,
                "risk_reasons": "recurrence_boundary",
            }
        ]
    )
    stale = {
        "sub_same": {
            "decision": "reject",
            "rationale": "Produced under an older gate",
            "exclude_issue_ids": [],
            "partitions": [],
        }
    }

    repaired, _, provenance, _ = adjudication.apply_adjudication(
        occurrences,
        np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float32),
        groups,
        assignments,
        risk,
        stale,
        min_recurring_reports=3,
    )

    assert repaired["adjudication_action"].tolist() == ["not_flagged"]
    assert repaired["recurrence_status"].tolist() == ["recurring"]
    assert provenance["decision"].tolist() == ["not_flagged"]


def test_adjudication_outputs_can_feed_the_existing_audit_builder():
    occurrences = pd.DataFrame(
        {
            "issue_id": ["iss_1", "iss_2"],
            "report_key": ["r1", "r2"],
            "canonical_issue": ["Issue one", "Issue two"],
        }
    )
    groups = pd.DataFrame(
        {
            "final_group_id": ["adj_1"],
            "recurrence_status": ["emerging"],
            "report_count": [2],
            "issue_count": [2],
        }
    )
    assignments = pd.DataFrame(
        {
            "issue_id": ["iss_1", "iss_2"],
            "final_group_id": ["adj_1", "adj_1"],
            "assignment_similarity": [0.97, 0.96],
            "recurrence_status": ["emerging", "emerging"],
        }
    )

    subissues, indexed = adjudication.compatible_audit_frames(
        occurrences, groups, assignments
    )

    assert subissues.loc[0, "subissue_id"] == "adj_1"
    assert indexed["subissue_id"].tolist() == ["adj_1", "adj_1"]
    assert indexed["recurrence_status"].tolist() == ["emerging", "emerging"]


def _raw_v3_issue(**overrides):
    issue = {
        "canonical_issue": "Abnormal test results were not reviewed",
        "evidence_quote": "no reliable system for reviewing abnormal test results",
        "failure_state": "omitted",
        "process_stage": "assessment",
        "service_sectors": ["healthcare"],
        "issue_themes": ["clinical_assessment_care"],
        "service_contexts": ["acute_hospital"],
        "populations_at_risk": ["patient"],
        "communication_direction": "not_applicable",
        "responsible_actor_type": "provider_organisation",
        "responsible_actor_text": "hospital trust",
        "concern_status": "current_system_gap",
    }
    issue.update(overrides)
    return issue


def test_source_span_validation_distinguishes_section_and_missing_text():
    report = pd.Series(
        {
            "concerns": "The follow-up appointment was not arranged after discharge.",
            "circumstances": "The patient left hospital on Monday.",
        }
    )

    assert pipeline.source_span_status(
        "follow-up appointment was not arranged", "concerns", report
    ) == (True, "exact_normalised")
    assert pipeline.source_span_status(
        "The patient left hospital", "concerns", report
    ) == (True, "found_other_section")
    assert pipeline.source_span_status(
        "A fact not in the report", "concerns", report
    ) == (
        False,
        "not_found",
    )


def test_investigation_only_report_is_in_scope_and_evidence_is_valid(tmp_path: Path):
    input_path = tmp_path / "reports.csv"
    pd.DataFrame(
        [
            {
                "id": "report-1",
                "url": "https://example.test/report-1",
                "date": "2026-01-01",
                "coroner": "A Coroner",
                "area": "Area",
                "receiver": "Receiver",
                "investigation": "The employer did not provide first aid equipment.",
                "circumstances": "",
                "concerns": "",
            }
        ]
    ).to_csv(input_path, index=False)

    reports = pipeline.load_reports(input_path, subset_size=0, seed=42)
    report = reports.iloc[0]

    assert len(reports) == 1
    assert "Investigation:\nThe employer did not provide" in pipeline.build_source_text(
        report, 12000
    )
    assert pipeline.locate_evidence_section(
        "The employer did not provide first aid equipment.", report
    ) == (True, "investigation", "exact_normalised")


def test_full_workflow_input_coverage_makes_exclusions_explicit(tmp_path: Path):
    input_path = tmp_path / "reports.csv"
    pd.DataFrame(
        [
            {
                "id": "usable",
                "url": "https://example.test/usable",
                "concerns": "",
                "circumstances": "",
                "investigation": "Usable investigation evidence.",
            },
            {
                "id": "empty",
                "url": "https://example.test/empty",
                "concerns": "",
                "circumstances": "",
                "investigation": "",
            },
        ]
    ).to_csv(input_path, index=False)

    metrics = full_workflow.build_input_coverage(input_path, tmp_path)
    excluded = pd.read_csv(tmp_path / "00_excluded_reports.csv")

    assert metrics["usable_reports"] == 1
    assert metrics["excluded_missing_source_text"] == 1
    assert excluded[["id", "exclusion_reason"]].to_dict("records") == [
        {"id": "empty", "exclusion_reason": "missing_source_text"}
    ]


def test_invalid_elliptical_source_span_can_recover_exact_sentence():
    report = pd.Series(
        {
            "concerns": "The sling had caused a long deep grade 2 pressure sore.",
            "circumstances": "",
        }
    )
    issue = {
        "source_section": "concerns",
        "source_span": "sling... had caused a long deep grade 2 pressure sore",
        "issue_statement_original": "The sling caused a pressure sore.",
        "canonical_issue": "Incorrect sling application caused a pressure sore",
    }

    recovered = pipeline.recover_source_span(issue, report)

    assert recovered == "The sling had caused a long deep grade 2 pressure sore."


def test_minor_transcription_error_can_recover_but_loose_paraphrase_cannot():
    report = pd.Series(
        {
            "concerns": (
                "An on-site defibrillator was not used by staff even though it was "
                "available nearby. Records were incomplete."
            ),
            "circumstances": "",
        }
    )
    near_exact = {
        "source_section": "concerns",
        "source_span": "An on-site defibrintillator was not used by staff",
        "issue_statement_original": "An on-site defibrillator was not used by staff.",
        "canonical_issue": "On-site defibrillator was not used",
    }
    paraphrase = {
        "source_section": "concerns",
        "source_span": "Emergency equipment was unavailable",
        "issue_statement_original": "Emergency equipment was unavailable.",
        "canonical_issue": "Emergency equipment was unavailable",
    }

    assert pipeline.recover_source_span(near_exact, report) == (
        "An on-site defibrillator was not used by staff even though it was available nearby."
    )
    assert pipeline.recover_source_span(paraphrase, report) == ""


def test_changed_lead_in_can_recover_a_long_contiguous_exact_clause():
    report = pd.Series(
        {
            "concerns": (
                "Had there not been a failure by the college and Probation to set up "
                "an effective referral system, the risk would have been reassessed."
            ),
            "circumstances": "",
        }
    )
    issue = {
        "evidence_quote": (
            "There was a failure by the college and Probation to set up an effective "
            "referral system"
        ),
        "canonical_issue": (
            "An effective referral system between probation and colleges was absent"
        ),
    }

    assert pipeline.recover_source_span(issue, report) == report["concerns"]


def test_single_inserted_token_can_recover_an_otherwise_exact_long_quote():
    report = pd.Series(
        {
            "concerns": (
                "He was seen by the consultant in elderly medicine at 1645 and was "
                "given analgesia at 1700 hrs, almost 7 hours after arrival."
            ),
            "circumstances": "",
        }
    )
    issue = {
        "evidence_quote": (
            "He was seen by the consultant in elderly medicine at 1645 hrs and was "
            "given analgesia at 1700 hrs, almost 7 hours after arrival."
        ),
        "canonical_issue": "Analgesia administration was delayed",
    }

    assert pipeline.recover_source_span(issue, report) == report["concerns"]


def test_evidence_section_is_derived_from_the_quote():
    report = pd.Series(
        {
            "concerns": "At night, her buzzer was taken away and her door was shut.",
            "circumstances": "",
        }
    )

    assert pipeline.locate_evidence_section(report["concerns"], report) == (
        True,
        "concerns",
        "exact_normalised",
    )


def test_validate_issue_coerces_unknown_enums_without_inventing_values():
    schema = pipeline.load_schema()
    report = pd.Series(
        {
            "concerns": "There was no reliable system for reviewing abnormal test results.",
            "circumstances": "",
        }
    )
    raw = _raw_v3_issue(failure_state="made_up_value")

    issue, warnings = pipeline.validate_issue(
        raw,
        report=report,
        enums=pipeline.schema_enums(schema),
        canonical_max_words=18,
    )

    assert issue is not None
    assert issue["failure_state"] == "other_review"
    assert issue["evidence_valid"] is True
    assert "invalid_failure_state:made_up_value" in warnings


def test_validate_issue_maps_explicitly_unclear_responsibility_to_ambiguous():
    schema = pipeline.load_schema()
    report = pd.Series(
        {
            "concerns": "Responsibility and timing for reviewing the plan were unclear.",
            "circumstances": "",
        }
    )
    raw = _raw_v3_issue(
        canonical_issue="Responsibility for plan review was unclear",
        evidence_quote=report["concerns"],
        failure_state="other_review",
        process_stage="planning",
        issue_themes=["governance_learning"],
        responsible_actor_text="",
    )

    issue, warnings = pipeline.validate_issue(
        raw,
        report=report,
        enums=pipeline.schema_enums(schema),
        canonical_max_words=18,
    )

    assert issue is not None
    assert issue["failure_state"] == "ambiguous"
    assert "failure_state_inferred_ambiguous" in warnings


def test_v3_schema_separates_sector_theme_context_and_population():
    schema = pipeline.load_schema()
    properties = schema["properties"]["issues"]
    enums = pipeline.schema_enums(schema)

    assert schema["$id"].endswith("issue-occurrence-v3.json")
    assert properties["maxItems"] == 24
    assert {
        "policy_development",
        "design_engineering",
        "product_labelling_warning",
        "commissioning_funding",
        "workforce_management",
        "procurement_supply",
        "prevention_risk_reduction",
    } <= enums["process_stage"]
    assert {"transport", "defence_military", "animal_care_control"} <= enums[
        "service_sectors"
    ]
    assert {"communication_handover", "risk_assessment_management"} <= enums[
        "issue_themes"
    ]
    assert "higher_education" in enums["service_contexts"]
    assert "outdoor_water" in enums["service_contexts"]
    assert {"trainee", "service_member", "participant"} <= enums["populations_at_risk"]
    assert "patient_family_to_professional" in enums["communication_direction"]


def test_extraction_quality_metrics_exposes_caps_and_weak_taxonomy_values():
    reports = pd.DataFrame({"report_key": ["r1", "r2"]})
    occurrences = pd.DataFrame(
        [
            {
                "report_key": "r1",
                "canonical_issue": "Need for a safer process",
                "evidence_valid": True,
                **{field: "other_review" for field in pipeline.SCALAR_ENUM_FIELDS},
                **{field: "other_review" for field in pipeline.ARRAY_ENUM_FIELDS},
            },
            {
                "report_key": "r1",
                "canonical_issue": "Results were not communicated",
                "evidence_valid": False,
                **{field: "not_stated" for field in pipeline.SCALAR_ENUM_FIELDS},
                **{field: "not_stated" for field in pipeline.ARRAY_ENUM_FIELDS},
            },
        ]
    )

    metrics = pipeline.extraction_quality_metrics(reports, occurrences, max_issues=2)

    assert metrics["reports_with_no_issues"] == 1
    assert metrics["reports_at_issue_cap"] == 1
    assert metrics["canonical_not_failure_framed"] == 1
    assert metrics["weak_controlled_values"]["process_stage"] == {
        "count": 1,
        "percent": 50.0,
    }


def test_one_report_call_returns_enriched_occurrence(monkeypatch):
    schema = pipeline.load_schema()
    report = pd.Series(
        {
            "report_key": "rpt_1",
            "id": "report-1",
            "concerns": "There was no reliable system for reviewing abnormal test results.",
            "circumstances": "",
        }
    )
    payload = {"issues": [_raw_v3_issue(responsible_actor_text="")]}
    calls = []

    def fake_ollama_json(**kwargs):
        calls.append(kwargs)
        return payload, json.dumps(payload)

    monkeypatch.setattr(pipeline, "ollama_json", fake_ollama_json)
    args = Namespace(
        max_source_chars=12000,
        max_issues=8,
        continuation_issues=0,
        continue_at_cap=False,
        canonical_max_words=18,
        retries=2,
        extract_num_predict=6400,
        ollama_num_ctx=16384,
        ollama_host="http://localhost:11434",
        model="test-model",
        request_timeout=10,
    )

    record = pipeline.extract_report(
        report,
        args=args,
        schema=schema,
        enums=pipeline.schema_enums(schema),
    )

    assert len(calls) == 1
    assert calls[0]["num_predict"] == 6400
    assert calls[0]["num_ctx"] == 16384
    assert record["status"] == "success"
    assert record["issues"][0]["failure_state"] == "omitted"
    assert record["issues"][0]["evidence_valid"] is True
    assert record["issues"][0]["_validation_warnings"] == []


def test_nonempty_remedy_only_concern_retries_an_empty_response(monkeypatch):
    schema = pipeline.load_schema()
    report = pd.Series(
        {
            "report_key": "rpt_1",
            "id": "report-1",
            "concerns": "Black-box telematics should be compulsory for young drivers.",
            "circumstances": "",
        }
    )
    valid = _raw_v3_issue(
        canonical_issue="Black-box telematics are not mandatory for young drivers",
        evidence_quote=report["concerns"],
        failure_state="omitted",
        process_stage="regulation_oversight",
        service_sectors=["transport", "government_regulation"],
        issue_themes=["policy_regulation"],
        service_contexts=["road"],
        populations_at_risk=["young_person", "road_user"],
        concern_status="recommendation_only",
    )
    payloads = [({"issues": []}, "{}"), ({"issues": [valid]}, json.dumps(valid))]

    def fake_ollama_json(**_kwargs):
        return payloads.pop(0)

    monkeypatch.setattr(pipeline, "ollama_json", fake_ollama_json)
    args = Namespace(
        max_source_chars=12000,
        max_issues=24,
        continuation_issues=0,
        continue_at_cap=False,
        canonical_max_words=22,
        retries=1,
        extract_num_predict=6400,
        ollama_num_ctx=16384,
        ollama_host="http://localhost:11434",
        model="test-model",
        request_timeout=10,
    )

    record = pipeline.extract_report(
        report,
        args=args,
        schema=schema,
        enums=pipeline.schema_enums(schema),
    )

    assert not payloads
    assert len(record["issues"]) == 1
    assert record["issues"][0]["concern_status"] == "recommendation_only"


def test_cap_response_makes_one_continuation_call(monkeypatch):
    schema = pipeline.load_schema()
    report = pd.Series(
        {
            "report_key": "rpt_1",
            "id": "report-1",
            "concerns": (
                "The first safeguard failed. The second safeguard failed. "
                "The third safeguard failed."
            ),
            "circumstances": "",
        }
    )
    first = [
        _raw_v3_issue(
            canonical_issue="First safeguard was omitted",
            evidence_quote="The first safeguard failed.",
        ),
        _raw_v3_issue(
            canonical_issue="Second safeguard was omitted",
            evidence_quote="The second safeguard failed.",
        ),
    ]
    extra = _raw_v3_issue(
        canonical_issue="Third safeguard was omitted",
        evidence_quote="The third safeguard failed.",
    )
    payloads = [
        ({"issues": first}, json.dumps(first)),
        ({"issues": [extra]}, json.dumps(extra)),
    ]

    def fake_ollama_json(**_kwargs):
        return payloads.pop(0)

    monkeypatch.setattr(pipeline, "ollama_json", fake_ollama_json)
    args = Namespace(
        max_source_chars=12000,
        max_issues=2,
        continuation_issues=1,
        continue_at_cap=True,
        canonical_max_words=22,
        retries=0,
        extract_num_predict=6400,
        ollama_num_ctx=16384,
        ollama_host="http://localhost:11434",
        model="test-model",
        request_timeout=10,
    )

    record = pipeline.extract_report(
        report,
        args=args,
        schema=schema,
        enums=pipeline.schema_enums(schema),
    )

    assert not payloads
    assert len(record["issues"]) == 3


def test_occurrence_warnings_are_not_copied_from_other_issues():
    reports = pd.DataFrame(
        [
            {
                "report_key": "r1",
                "id": "1",
                "url": "https://example.test/1",
                "date": "2026-01-01",
                "coroner": "A Coroner",
                "area": "Area",
                "receiver": "Receiver",
            }
        ]
    )
    records = {
        "r1": {
            "status": "success",
            "warnings": ["source_span_not_found"],
            "issues": [
                {
                    "canonical_issue": "First failure",
                    "_validation_warnings": [],
                },
                {
                    "canonical_issue": "Second failure",
                    "_validation_warnings": ["source_span_not_found"],
                },
            ],
        }
    }

    _, occurrences, _ = pipeline.records_to_frames(reports, records)

    assert occurrences["extraction_warnings"].tolist() == ["", "source_span_not_found"]
    assert "_validation_warnings" not in occurrences.columns


def test_occurrence_frame_repairs_recoverable_checkpoint_evidence():
    source = (
        "Had there not been a failure by the college and Probation to set up an "
        "effective referral system, the risk would have been reassessed."
    )
    reports = pd.DataFrame(
        [
            {
                "report_key": "r1",
                "id": "1",
                "url": "https://example.test/1",
                "date": "2026-01-01",
                "coroner": "A Coroner",
                "area": "Area",
                "receiver": "Receiver",
                "concerns": source,
                "circumstances": "",
            }
        ]
    )
    records = {
        "r1": {
            "status": "success",
            "issues": [
                {
                    "canonical_issue": "An effective referral system was absent",
                    "evidence_quote": (
                        "There was a failure by the college and Probation to set up "
                        "an effective referral system"
                    ),
                    "evidence_valid": False,
                    "evidence_section": "not_stated",
                    "evidence_status": "not_found",
                    "_validation_warnings": ["evidence_quote_not_found"],
                }
            ],
        }
    }

    _, occurrences, _ = pipeline.records_to_frames(reports, records)

    assert occurrences.loc[0, "evidence_quote"] == source
    assert bool(occurrences.loc[0, "evidence_valid"])
    assert occurrences.loc[0, "evidence_section"] == "concerns"
    assert occurrences.loc[0, "extraction_warnings"] == "evidence_quote_recovered"


def _index_args() -> Namespace:
    return Namespace(
        top_k=4,
        edge_similarity=0.90,
        split_similarity=0.90,
        min_centroid_similarity=0.85,
        min_recurring_reports=3,
        label_subissues=False,
        embedding_model="unused-in-test",
        allow_model_download=False,
        ollama_host="http://localhost:11434",
        model="unused-in-test",
        request_timeout=10,
    )


def _occurrence(
    issue_id: str,
    report: str,
    canonical_issue: str,
    *,
    failure_mode: str,
) -> dict[str, str]:
    return {
        "issue_id": issue_id,
        "report_key": report,
        "canonical_issue": canonical_issue,
        "failure_state": "omitted" if failure_mode == "not_reviewed" else "inadequate",
        "process_stage": "assessment"
        if failure_mode == "not_reviewed"
        else "workforce_management",
        "issue_themes": (
            "clinical_assessment_care"
            if failure_mode == "not_reviewed"
            else "staffing_capacity"
        ),
        "communication_direction": "not_applicable",
    }


def test_index_uses_distinct_reports_for_recurrence_and_unique_presence(tmp_path: Path):
    occurrences = pd.DataFrame(
        [
            _occurrence(
                "i1",
                "r1",
                "Abnormal test results were not reviewed.",
                failure_mode="not_reviewed",
            ),
            _occurrence(
                "i2",
                "r2",
                "Abnormal investigation results were not reviewed.",
                failure_mode="not_reviewed",
            ),
            _occurrence(
                "i3",
                "r3",
                "Abnormal test findings were not reviewed.",
                failure_mode="not_reviewed",
            ),
            _occurrence(
                "i4",
                "r4",
                "Staffing levels were insufficient.",
                failure_mode="insufficient",
            ),
            _occurrence(
                "i5",
                "r4",
                "The service had insufficient staff.",
                failure_mode="insufficient",
            ),
            _occurrence(
                "i6",
                "r5",
                "Staffing capacity was insufficient.",
                failure_mode="insufficient",
            ),
        ]
    )
    # Two semantic components: three reports in the first, only two distinct
    # reports in the second despite three issue occurrences.
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.035, 0.0],
            [0.998, -0.04, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.999, 0.035],
            [0.0, 0.998, -0.04],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    pd.DataFrame(
        [
            {
                "report_key": report,
                "report_id": report,
                "report_url": f"https://example.test/{report}",
                "report_date": "2025-01-01",
            }
            for report in ["r1", "r2", "r3", "r4", "r5"]
        ]
    ).to_csv(tmp_path / "01_reports.csv", index=False)
    np.save(tmp_path / "02_issue_embeddings.npy", embeddings)
    embedding_texts = occurrences.apply(pipeline.embedding_text, axis=1)
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{issue_id}\t{text}"
            for issue_id, text in zip(
                occurrences["issue_id"], embedding_texts, strict=False
            )
        ).encode("utf-8")
    ).hexdigest()
    (tmp_path / "02_embeddings_meta.json").write_text(
        json.dumps(
            {
                "model": "unused-in-test",
                "row_count": len(occurrences),
                "input_sha256": fingerprint,
            }
        ),
        encoding="utf-8",
    )

    metrics = pipeline.build_index(
        occurrences,
        run_dir=tmp_path,
        args=_index_args(),
    )

    subissues = pd.read_csv(tmp_path / "03_subissues.csv")
    presence = pd.read_csv(tmp_path / "03_issue_report_presence.csv")
    assert metrics["recurring_subissues"] == 1
    assert metrics["emerging_subissues"] == 1
    assert sorted(subissues["report_count"].tolist()) == [2, 3]
    assert not presence.duplicated(["subissue_id", "report_key"]).any()
    emerging_id = subissues.loc[
        subissues["recurrence_status"] == "emerging", "subissue_id"
    ].iloc[0]
    assert len(presence[presence["subissue_id"] == emerging_id]) == 2


def test_quality_audit_group_selection_is_stratified_and_bounded():
    groups = pd.DataFrame(
        [
            {
                "subissue_id": f"g{number}",
                "report_count": reports,
                "issue_count": reports,
                "facet_risk_score": risk,
                "median_centroid_similarity": cohesion,
            }
            for number, reports, risk, cohesion in [
                (1, 10, 1, 0.96),
                (2, 3, 0, 0.97),
                (3, 3, 1, 0.96),
                (4, 5, 6, 0.94),
                (5, 5, 0, 0.98),
            ]
        ]
    )

    queue = quality_audit.select_group_review(
        groups,
        review_size=4,
        boundary_size=1,
        risk_size=1,
        large_report_count=8,
        large_size=1,
        seed=7,
    )

    assert len(queue) == 4
    assert queue["subissue_id"].is_unique
    assert set(queue["review_reason"]) == {
        "large_group",
        "recurrence_boundary",
        "facet_conflict_risk",
        "random_fill",
    }


def test_quality_audit_missed_links_exclude_same_report_and_recurring_rows():
    indexed = pd.DataFrame(
        [
            {
                "issue_id": "i1",
                "report_key": "r1",
                "recurrence_status": "isolated",
                "canonical_issue": "Service failed to contact patient",
                "responsible_actor_role": "provider organisation",
                "issue_object": "patient contact",
                "failure_state": "omitted",
                "process_stage": "follow_up",
                "communication_direction": "professional_to_patient",
            },
            {
                "issue_id": "i2",
                "report_key": "r2",
                "recurrence_status": "emerging",
                "canonical_issue": "Provider omitted follow-up contact",
                "responsible_actor_role": "provider organisation",
                "issue_object": "follow-up contact",
                "failure_state": "omitted",
                "process_stage": "follow_up",
                "communication_direction": "professional_to_patient",
            },
            {
                "issue_id": "i3",
                "report_key": "r1",
                "recurrence_status": "isolated",
                "canonical_issue": "Same-report duplicate",
                "responsible_actor_role": "provider organisation",
                "issue_object": "patient contact",
                "failure_state": "omitted",
                "process_stage": "follow_up",
                "communication_direction": "professional_to_patient",
            },
            {
                "issue_id": "i4",
                "report_key": "r4",
                "recurrence_status": "recurring",
                "canonical_issue": "Recurring candidate",
                "responsible_actor_role": "provider organisation",
                "issue_object": "patient contact",
                "failure_state": "omitted",
                "process_stage": "follow_up",
                "communication_direction": "professional_to_patient",
            },
        ]
    )
    indexed["subissue_id"] = ["", "emerging_group", "", "recurring_group"]
    embeddings = np.asarray(
        [[1.0, 0.0], [0.99, 0.1], [1.0, 0.01], [1.0, 0.02]], dtype=np.float32
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    queue = quality_audit.build_missed_link_candidates(
        indexed,
        embeddings,
        minimum_similarity=0.78,
        review_size=5,
    )

    assert any(
        {row.left_issue_id, row.right_issue_id} == {"i1", "i2"}
        for row in queue.itertuples(index=False)
    )
    assert not any(
        row.left_report_key == row.right_report_key
        for row in queue.itertuples(index=False)
    )
    assert (
        not queue[["left_recurrence_status", "right_recurrence_status"]]
        .eq("recurring")
        .any()
        .any()
    )


def test_quality_audit_scoring_requires_complete_decision_coverage(tmp_path):
    pd.DataFrame(
        [
            {
                "subissue_id": group_id,
                "review_reason": reason,
                "issue_count": issue_count,
                "audit_decision": "",
                "incorrect_member_issue_ids": "",
                "proposed_split": "",
                "audit_notes": "",
            }
            for group_id, reason, issue_count in [
                ("g1", "large_group", 4),
                ("g2", "recurrence_boundary", 3),
            ]
        ]
    ).to_csv(tmp_path / "group_review_queue.csv", index=False)
    (tmp_path / "group_decisions.json").write_text(
        json.dumps([{"subissue_id": "g1", "audit_decision": "accept"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="decision coverage mismatch"):
        audit_scoring.score_group_review(tmp_path)


def test_tuning_evaluation_reports_coverage_and_builds_review_queue():
    occurrences = pd.DataFrame(
        [
            _occurrence(
                "i1",
                "r1",
                "Abnormal test results were not reviewed.",
                failure_mode="not_reviewed",
            ),
            _occurrence(
                "i2",
                "r2",
                "Abnormal investigation results were not reviewed.",
                failure_mode="not_reviewed",
            ),
            _occurrence(
                "i3",
                "r3",
                "Abnormal test findings were not reviewed.",
                failure_mode="not_reviewed",
            ),
            _occurrence(
                "i4",
                "r4",
                "Staffing levels were insufficient.",
                failure_mode="insufficient",
            ),
            _occurrence(
                "i5",
                "r5",
                "Staffing capacity was insufficient.",
                failure_mode="insufficient",
            ),
        ]
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.035, 0.0],
            [0.998, -0.04, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.999, 0.035],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    config = tuning.TuningConfig("test", 4, 0.90, 0.90, 0.85)

    summary, groups, assignments, centroids = tuning.evaluate_config(
        occurrences,
        embeddings,
        config,
        min_recurring_reports=3,
    )

    recurring = groups[groups["recurrence_status"] == "recurring"]
    queue = tuning.build_review_queue(recurring, review_size=60)
    assert summary["recurring_subissues"] == 1
    assert summary["emerging_subissues"] == 1
    assert summary["assigned_issue_occurrences"] == 5
    assert summary["assigned_percent"] == 100.0
    assert len(assignments) == 5
    assert len(centroids) == 2
    assert len(queue) == 1
    assert {"coherent", "over_merged", "near_duplicate_of", "review_notes"} <= set(
        queue
    )


def test_tuning_config_parser_rejects_invalid_values():
    assert tuning.parse_config("candidate:40:0.84:0.87:0.83") == tuning.TuningConfig(
        "candidate", 40, 0.84, 0.87, 0.83
    )

    with pytest.raises(argparse.ArgumentTypeError, match="between 0 and 1"):
        tuning.parse_config("candidate:40:1.1:0.87:0.83")


def test_review_repair_merges_duplicates_and_splits_flagged_groups():
    occurrences = pd.DataFrame(
        [
            _occurrence(
                "i1", "r1", "Delayed ambulance response.", failure_mode="delayed"
            ),
            _occurrence(
                "i2", "r2", "Slow ambulance attendance.", failure_mode="delayed"
            ),
            _occurrence(
                "i3",
                "r3",
                "Ambulance response exceeded target.",
                failure_mode="delayed",
            ),
            _occurrence(
                "i4", "r4", "Ambulance arrival was delayed.", failure_mode="delayed"
            ),
            _occurrence(
                "i5", "r5", "Medication was not supplied.", failure_mode="missing"
            ),
            _occurrence(
                "i6",
                "r6",
                "Pain relief was inappropriate.",
                failure_mode="unsafe_design",
            ),
            _occurrence(
                "i7",
                "r7",
                "A discharge assessment was omitted.",
                failure_mode="missing",
            ),
        ]
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.999, 0.03, 0.0],
            [0.998, -0.04, 0.0],
            [0.997, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    groups = pd.DataFrame(
        [
            {"candidate_group_id": "g1", "recurrence_status": "recurring"},
            {"candidate_group_id": "g2", "recurrence_status": "recurring"},
            {"candidate_group_id": "g3", "recurrence_status": "recurring"},
        ]
    )
    assignments = pd.DataFrame(
        [
            {"candidate_group_id": "g1", "issue_id": "i1"},
            {"candidate_group_id": "g1", "issue_id": "i2"},
            {"candidate_group_id": "g2", "issue_id": "i3"},
            {"candidate_group_id": "g2", "issue_id": "i4"},
            {"candidate_group_id": "g3", "issue_id": "i5"},
            {"candidate_group_id": "g3", "issue_id": "i6"},
            {"candidate_group_id": "g3", "issue_id": "i7"},
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "candidate_group_id": "g1",
                "coherent": "yes",
                "over_merged": "no",
                "near_duplicate_of": "g2",
            },
            {
                "candidate_group_id": "g2",
                "coherent": "yes",
                "over_merged": "no",
                "near_duplicate_of": "g1",
            },
            {
                "candidate_group_id": "g3",
                "coherent": "no",
                "over_merged": "yes",
                "near_duplicate_of": "",
            },
        ]
    )

    repaired, repaired_assignments, changed, metrics = repair.repair_groups(
        occurrences,
        embeddings,
        groups,
        assignments,
        reviews,
        split_similarity=0.95,
        min_recurring_reports=3,
    )

    assert metrics["confirmed_merge_pairs"] == 1
    assert metrics["split_output_groups"] == 3
    assert metrics["repaired_recurring_groups"] == 1
    assert sorted(repaired["repair_action"].tolist()) == [
        "merged",
        "split",
        "split",
        "split",
    ]
    assert len(repaired_assignments) == len(occurrences)
    assert len(changed) == 4


def test_prototype_assignment_requires_domain_cross_report_and_margin():
    occurrences = pd.DataFrame(
        [
            {
                "issue_id": "p1",
                "report_key": "r1",
                "canonical_issue": "Delayed handover",
                "issue_themes": "clinical_assessment_care",
            },
            {
                "issue_id": "p2",
                "report_key": "r2",
                "canonical_issue": "Slow handover",
                "issue_themes": "clinical_assessment_care",
            },
            {
                "issue_id": "u1",
                "report_key": "r3",
                "canonical_issue": "Handover was delayed",
                "issue_themes": "clinical_assessment_care",
            },
            {
                "issue_id": "u2",
                "report_key": "r1",
                "canonical_issue": "Same-report handover",
                "issue_themes": "clinical_assessment_care",
            },
            {
                "issue_id": "u3",
                "report_key": "r4",
                "canonical_issue": "Road handover wording",
                "issue_themes": "environmental_design",
            },
        ]
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.995, 0.05],
            [0.999, 0.02],
            [0.998, 0.03],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    prototype = prototype_assignment.Prototype(
        prototype_id="group",
        source="test",
        status="emerging",
        subject_domain="clinical_assessment_care",
        issue_ids=["p1", "p2"],
        report_ids={"r1", "r2"},
        centroid=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    alternative = prototype_assignment.Prototype(
        prototype_id="alternative",
        source="test",
        status="emerging",
        subject_domain="clinical_assessment_care",
        issue_ids=[],
        report_ids=set(),
        centroid=np.asarray([0.0, 1.0], dtype=np.float32),
    )

    assigned = prototype_assignment.assign_occurrences(
        occurrences,
        embeddings,
        np.asarray([2, 3, 4]),
        [prototype, alternative],
        similarity_threshold=0.84,
        margin_threshold=0.02,
        domain_match=True,
    )

    assert assigned["issue_id"].tolist() == ["u1"]


def test_normalization_preserves_requested_order_and_trims_fields():
    raw = [
        {
            "issue_id": "i2",
            "responsible_actor_role": "the local highway authority",
            "failed_action": "inspect and maintain",
            "issue_object": "road barrier inspection and maintenance arrangements",
            "counterparty_role": "road users",
            "canonical_issue": "The local highway authority did not adequately inspect and maintain road barriers before foreseeable vehicle impacts occurred",
        },
        {
            "issue_id": "i1",
            "responsible_actor_role": "community mental health service",
            "failed_action": "contact",
            "issue_object": "follow-up contact",
            "counterparty_role": "patient",
            "canonical_issue": "The community mental health service did not make follow-up contact after the patient sought help",
        },
    ]

    results = normalization.normalize_result(raw, ["i1", "i2"], max_words=12)

    assert [result["issue_id"] for result in results] == ["i1", "i2"]
    assert results[0]["issue_object"] == "follow-up contact"
    assert len(results[1]["canonical_issue"].split()) == 12


def test_normalization_rejects_missing_or_unknown_issue_ids():
    raw = [
        {
            "issue_id": "unknown",
            "responsible_actor_role": "employer",
            "failed_action": "assess",
            "issue_object": "workplace risk assessment",
            "counterparty_role": "worker",
            "canonical_issue": "The employer omitted a workplace risk assessment",
        }
    ]

    with pytest.raises(ValueError, match="unexpected issue_id"):
        normalization.normalize_result(raw, ["i1"], max_words=26)


def test_normalization_recovers_unique_near_match_issue_id():
    expected = "iss_3c64aff0b03b47c9"
    raw = [
        {
            "issue_id": "iss_3c64aff0b03b479",
            "responsible_actor_role": "provider organisation",
            "failed_action": "maintain",
            "issue_object": "clinical records",
            "counterparty_role": "not stated",
            "canonical_issue": "Provider organisation failed to maintain clinical records",
        }
    ]

    result = normalization.normalize_result(raw, [expected], max_words=26)

    assert result[0]["issue_id"] == expected
    assert result[0]["normalization_warnings"] == [
        "issue_id_recovered:iss_3c64aff0b03b479"
    ]


def test_normalization_does_not_guess_ambiguous_near_match_issue_id():
    with pytest.raises(ValueError, match="unexpected issue_id"):
        normalization.recover_issue_id("iss_abc", ["iss_ab1", "iss_ab2"])


def test_normalization_requires_stated_actor_in_canonical_issue():
    raw = [
        {
            "issue_id": "i1",
            "responsible_actor_role": "mental health service",
            "failed_action": "contact",
            "issue_object": "follow-up contact",
            "counterparty_role": "patient",
            "canonical_issue": "Follow-up contact was not attempted",
        }
    ]

    with pytest.raises(ValueError, match="does not contain responsible_actor_role"):
        normalization.normalize_result(raw, ["i1"], max_words=26)


def test_normalization_does_not_discard_extracted_actor_type():
    result = [
        {
            "issue_id": "i1",
            "responsible_actor_role": "not stated",
            "issue_object": "risk assessment",
            "canonical_issue": "A risk assessment was omitted",
        }
    ]
    source = [
        {
            "issue_id": "i1",
            "responsible_actor_type": "provider_organisation",
            "responsible_actor_text": "",
        }
    ]

    with pytest.raises(ValueError, match="discarded extracted actor information"):
        normalization.validate_source_actor(result, source)


def test_normalized_output_preserves_original_issue_and_evidence():
    frame = pd.DataFrame(
        [
            {
                "issue_id": "i1",
                "canonical_issue": "Follow-up contact was omitted",
                "evidence_quote": "the team did not contact him again",
            },
            {
                "issue_id": "i2",
                "canonical_issue": "Barrier inspection was inadequate",
                "evidence_quote": "the inspection regime was inadequate",
            },
        ]
    )
    records = {
        "batch": {
            "status": "success",
            "results": [
                {
                    "issue_id": "i1",
                    "responsible_actor_role": "mental health team",
                    "failed_action": "contact",
                    "issue_object": "follow-up contact",
                    "counterparty_role": "patient",
                    "canonical_issue": "The mental health team omitted follow-up contact",
                }
            ],
        }
    }

    output = normalization.build_output(frame, records)

    assert output.loc[0, "canonical_issue_original"] == "Follow-up contact was omitted"
    assert (
        output.loc[0, "canonical_issue"]
        == "The mental health team omitted follow-up contact"
    )
    assert output.loc[0, "evidence_quote"] == "the team did not contact him again"
    assert output.loc[1, "normalization_status"] == "not_selected"


def test_normalized_output_preserves_results_from_partial_batch():
    frame = pd.DataFrame(
        [
            {
                "issue_id": "i1",
                "canonical_issue": "Records were incomplete",
                "evidence_quote": "records were incomplete",
            },
            {
                "issue_id": "i2",
                "canonical_issue": "Training was absent",
                "evidence_quote": "there was no training",
            },
        ]
    )
    records = {
        "batch": {
            "status": "partial",
            "results": [
                {
                    "issue_id": "i1",
                    "responsible_actor_role": "provider organisation",
                    "failed_action": "maintain",
                    "issue_object": "clinical records",
                    "counterparty_role": "not stated",
                    "canonical_issue": "Provider organisation maintained incomplete clinical records",
                    "normalization_warnings": [],
                }
            ],
        }
    }

    output = normalization.build_output(frame, records)

    assert output.loc[0, "normalization_status"] == "success"
    assert output.loc[1, "normalization_status"] == "not_selected"


def test_normalization_review_queue_includes_warnings_and_failures():
    output = pd.DataFrame(
        [
            {
                "issue_id": "ok",
                "normalization_status": "success",
                "normalization_warnings": "",
            },
            {
                "issue_id": "warned",
                "normalization_status": "success",
                "normalization_warnings": "single_word_object_review",
            },
            {
                "issue_id": "failed",
                "normalization_status": "failed",
                "normalization_warnings": "",
            },
        ]
    )

    review = normalization.build_review_queue(output)

    assert review["issue_id"].tolist() == ["warned", "failed"]


def test_relational_normalization_builds_stable_linkage_statement():
    raw = [
        {
            "issue_id": "i1",
            "responsible_actor_role": "mental health service",
            "failed_action": "contact",
            "issue_object": "follow-up contact",
            "counterparty_role": "patient",
            "canonical_issue": "The mental health service failed to contact the patient",
        }
    ]

    result = normalization.normalize_result(raw, ["i1"], max_words=26)[0]

    assert result["linkage_statement"] == (
        "Actor: mental health service. Failed action: contact. "
        "Object: follow-up contact. Counterparty: patient."
    )


def test_relational_quality_gate_retains_direction_without_inventing_counterparty():
    eligible, reason = relational_linkage.linkage_quality(
        pd.Series(
            {
                "failed_action": "share",
                "issue_object": "discharge information",
                "counterparty_role": "not stated",
                "communication_direction": "between_organisations",
                "normalization_status": "success",
            }
        )
    )

    assert eligible
    assert reason == "eligible"


def test_relational_pair_scoring_penalizes_direction_and_action_conflicts():
    base = {
        "responsible_actor_role": "mental health service",
        "issue_object": "follow-up contact",
        "process_stage": "follow_up",
        "failure_state": "omitted",
    }
    service_omission = pd.Series(
        {
            **base,
            "failed_action": "contact",
            "counterparty_role": "patient",
            "communication_direction": "professional_to_patient",
        }
    )
    patient_action = pd.Series(
        {
            **base,
            "responsible_actor_role": "patient",
            "failed_action": "attend",
            "counterparty_role": "mental health service",
            "communication_direction": "patient_family_to_professional",
        }
    )

    adjustment, reasons = relational_linkage.categorical_adjustment(
        service_omission, patient_action
    )

    assert adjustment < 0
    assert "different_action_family:-0.05" in reasons
    assert "different_communication_direction:-0.055" in reasons


def test_relational_pair_scoring_rejects_generic_object_head_match():
    ct_result = pd.Series(
        {
            "responsible_actor_role": "provider organisation",
            "failed_action": "report",
            "issue_object": "CT scan results",
            "counterparty_role": "not stated",
            "communication_direction": "not_applicable",
            "process_stage": "information_record_management",
            "failure_state": "delayed",
        }
    )
    smear_result = pd.Series(
        {
            **ct_result.to_dict(),
            "issue_object": "smear test results",
            "failure_state": "incorrect",
        }
    )

    adjustment, reasons = relational_linkage.categorical_adjustment(
        ct_result, smear_result
    )

    assert adjustment < 0
    assert "object_shared_terms_generic_only:-0.08" in reasons
    assert (
        relational_linkage.hard_relation_conflict(ct_result, smear_result)
        == "object_entity_mismatch"
    )


def test_guarded_relational_scoring_vetoes_generic_action_with_disjoint_objects():
    investigation_policy = pd.Series(
        {
            "responsible_actor_role": "provider organisation",
            "failed_action": "provide",
            "issue_object": "policy for requesting investigations",
            "counterparty_role": "not stated",
            "communication_direction": "not_applicable",
            "process_stage": "policy_development",
            "failure_state": "omitted",
        }
    )
    furniture_policy = pd.Series(
        {
            **investigation_policy.to_dict(),
            "issue_object": "policy regarding furniture barricades",
        }
    )

    assert relational_linkage.guarded_object_conflict(
        investigation_policy,
        furniture_policy,
        action_object_similarity=0.84,
        minimum_specific_object_similarity=0.88,
    ) == "disjoint_specific_objects"


def test_guarded_relational_scoring_separates_material_failure_states():
    omitted = pd.Series(
        {
            "responsible_actor_role": "provider organisation",
            "failed_action": "administer",
            "issue_object": "prescribed medication",
            "counterparty_role": "patient",
            "communication_direction": "not_applicable",
            "failure_state": "omitted",
        }
    )
    incorrect = pd.Series({**omitted.to_dict(), "failure_state": "incorrect"})

    assert relational_linkage.guarded_relation_conflict(
        omitted, incorrect
    ) == "incompatible_failure_state"


def test_guarded_relational_scoring_separates_recording_from_performance():
    performed = pd.Series(
        {
            "responsible_actor_role": "practitioner",
            "failed_action": "perform",
            "issue_object": "clinical observations",
            "counterparty_role": "patient",
            "communication_direction": "not_applicable",
            "failure_state": "omitted",
        }
    )
    recorded = pd.Series({**performed.to_dict(), "failed_action": "record"})

    assert relational_linkage.guarded_relation_conflict(
        performed, recorded
    ) == "incompatible_action_family"


def test_guarded_relational_scoring_separates_calling_from_ambulance_response():
    caller = pd.Series(
        {
            "canonical_issue": "The healthcare team delayed calling an ambulance.",
            "responsible_actor_role": "healthcare team",
            "failed_action": "call",
            "issue_object": "ambulance",
            "counterparty_role": "ambulance service",
            "communication_direction": "service_to_service",
            "failure_state": "delayed",
        }
    )
    responder = pd.Series(
        {
            "canonical_issue": "The ambulance service delayed responding to the call.",
            "responsible_actor_role": "ambulance service",
            "failed_action": "respond to",
            "issue_object": "ambulance call",
            "counterparty_role": "patient",
            "communication_direction": "not_applicable",
            "failure_state": "delayed",
        }
    )

    assert relational_linkage.guarded_relation_conflict(
        caller, responder
    ) == "incompatible_ambulance_direction"


def test_guarded_relational_scoring_stops_generic_risk_assessment_bridge():
    conducted = pd.Series(
        {
            "responsible_actor_role": "provider organisation",
            "failed_action": "conduct",
            "issue_object": "risk assessments",
            "counterparty_role": "patient",
            "communication_direction": "not_applicable",
            "failure_state": "inadequate",
        }
    )
    updated = pd.Series(
        {
            **conducted.to_dict(),
            "failed_action": "update",
            "issue_object": "risk assessment",
        }
    )

    assert relational_linkage.guarded_object_conflict(
        conducted,
        updated,
        action_object_similarity=0.95,
        minimum_specific_object_similarity=0.82,
    ) == "underspecified_generic_object"


def test_guarded_relational_scoring_keeps_exact_generic_relations_linkable():
    first = pd.Series(
        {
            "responsible_actor_role": "provider organisation",
            "failed_action": "conduct",
            "issue_object": "risk assessment",
            "counterparty_role": "patient",
            "communication_direction": "not_applicable",
            "failure_state": "inadequate",
        }
    )
    paraphrase = pd.Series(
        {
            **first.to_dict(),
            "failed_action": "perform",
            "issue_object": "risk assessments",
        }
    )

    assert relational_linkage.guarded_object_conflict(
        first,
        paraphrase,
        action_object_similarity=0.95,
        minimum_specific_object_similarity=0.82,
    ) == ""


def test_guarded_relational_scoring_separates_communication_recipients():
    to_gp = pd.Series(
        {
            "responsible_actor_role": "provider organisation",
            "failed_action": "send",
            "issue_object": "discharge summary",
            "counterparty_role": "GP",
            "communication_direction": "between_organisations",
            "failure_state": "omitted",
        }
    )
    to_family = pd.Series(
        {
            **to_gp.to_dict(),
            "counterparty_role": "family",
            "communication_direction": "professional_to_family_carer",
        }
    )

    assert relational_linkage.guarded_relation_conflict(
        to_gp, to_family
    ) == "incompatible_communication_direction"


def test_view_similarities_recover_cosines_from_weighted_concatenation():
    weights = [0.10, 0.55, 0.25, 0.10]
    first = np.asarray(
        [
            value
            for weight in weights
            for value in (weight**0.5, 0.0)
        ],
        dtype=np.float32,
    )
    second = np.asarray(
        [
            value
            for weight in weights
            for value in (0.0, weight**0.5)
        ],
        dtype=np.float32,
    )

    similarities = relational_linkage.view_similarities(
        np.vstack([first, second]), 0, 1, weights
    )

    assert similarities == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-6)


def test_guarded_consolidation_can_merge_supported_groups_with_same_report():
    frame = pd.DataFrame(
        [
            {
                "report_key": report,
                "responsible_actor_role": "provider organisation",
                "counterparty_role": "not stated",
                "failed_action": "maintain",
                "issue_object": "nursing staffing levels",
                "communication_direction": "not_applicable",
                "failure_state": "inadequate",
            }
            for report in ("r1", "r2", "r1", "r3")
        ]
    )
    weights = [0.10, 0.55, 0.25, 0.10]
    vector = np.asarray(
        [
            value
            for weight in weights
            for value in (weight**0.5, 0.0)
        ],
        dtype=np.float32,
    )
    embeddings = np.vstack([vector] * 4)
    edges = [
        relational_linkage.PairScore(0, 2, 0.95, 0.0, 0.95, ()),
        relational_linkage.PairScore(1, 3, 0.95, 0.0, 0.95, ()),
    ]

    groups, provenance = relational_linkage.consolidate_guarded_groups(
        frame,
        embeddings,
        [[0, 1], [2, 3]],
        edges,
        weights=weights,
        minimum_score=0.85,
        minimum_coverage=0.5,
        minimum_relation_similarity=0.8,
        minimum_action_object_similarity=0.8,
        minimum_specific_object_similarity=0.88,
        maximum_positive_adjustment=0.03,
    )

    assert groups == [[0, 1, 2, 3]]
    assert len(provenance) == 1


def test_prototype_grouping_vetoes_directionally_reversed_bridge():
    frame = pd.DataFrame(
        [
            {
                "report_key": "r1",
                "responsible_actor_role": "service",
                "counterparty_role": "patient",
                "failed_action": "contact",
                "issue_object": "follow-up contact",
                "communication_direction": "professional_to_patient",
            },
            {
                "report_key": "r2",
                "responsible_actor_role": "service",
                "counterparty_role": "patient",
                "failed_action": "support",
                "issue_object": "service engagement",
                "communication_direction": "bidirectional",
            },
            {
                "report_key": "r3",
                "responsible_actor_role": "patient",
                "counterparty_role": "service",
                "failed_action": "attend",
                "issue_object": "follow-up appointment",
                "communication_direction": "patient_family_to_professional",
            },
        ]
    )
    edges = [
        relational_linkage.PairScore(0, 1, 0.9, 0.0, 0.9, ()),
        relational_linkage.PairScore(1, 2, 0.9, 0.0, 0.9, ()),
    ]

    groups = relational_linkage.prototype_anchored_groups(
        frame, edges, minimum_group_score=0.8
    )

    assert sorted(map(len, groups)) == [1, 2]


def test_constrained_density_grouping_supports_multiple_representatives():
    frame = pd.DataFrame(
        [
            {
                "report_key": f"r{index}",
                "responsible_actor_role": "provider organisation",
                "counterparty_role": "patient",
                "failed_action": "contact",
                "issue_object": "follow-up contact",
                "communication_direction": "professional_to_patient",
            }
            for index in range(3)
        ]
    )
    edges = [
        relational_linkage.PairScore(0, 1, 0.90, 0.0, 0.90, ()),
        relational_linkage.PairScore(1, 2, 0.86, 0.0, 0.86, ()),
    ]

    groups = relational_linkage.constrained_density_groups(
        frame,
        seed_edges=edges[:1],
        eligible_edges=edges,
        minimum_group_score=0.83,
        minimum_edge_density=0.60,
        minimum_member_coverage=0.34,
    )

    assert sorted(map(len, groups)) == [3]


def test_constrained_density_grouping_honours_cannot_link_constraints():
    frame = pd.DataFrame(
        [
            {
                "report_key": "r1",
                "responsible_actor_role": "service",
                "counterparty_role": "patient",
                "failed_action": "contact",
                "issue_object": "follow-up contact",
                "communication_direction": "professional_to_patient",
            },
            {
                "report_key": "r2",
                "responsible_actor_role": "service",
                "counterparty_role": "patient",
                "failed_action": "support",
                "issue_object": "service engagement",
                "communication_direction": "bidirectional",
            },
            {
                "report_key": "r3",
                "responsible_actor_role": "patient",
                "counterparty_role": "service",
                "failed_action": "attend",
                "issue_object": "follow-up appointment",
                "communication_direction": "patient_family_to_professional",
            },
        ]
    )
    edges = [
        relational_linkage.PairScore(0, 1, 0.90, 0.0, 0.90, ()),
        relational_linkage.PairScore(1, 2, 0.90, 0.0, 0.90, ()),
        relational_linkage.PairScore(0, 2, 0.86, 0.0, 0.86, ()),
    ]

    groups = relational_linkage.constrained_density_groups(
        frame,
        seed_edges=edges,
        eligible_edges=edges,
        minimum_group_score=0.83,
        minimum_edge_density=0.60,
        minimum_member_coverage=0.34,
    )

    assert sorted(map(len, groups)) == [1, 2]


def test_embedding_text_uses_actor_object_once_through_canonical_sentence():
    text = pipeline.embedding_text(
        pd.Series(
            {
                "canonical_issue": "The service failed to contact the patient",
                "responsible_actor_role": "mental health service",
                "issue_object": "follow-up contact",
                "failure_state": "omitted",
                "process_stage": "follow_up",
                "issue_themes": "communication_handover",
                "communication_direction": "professional_to_patient",
            }
        )
    )

    assert text.startswith("The service failed to contact the patient")
    assert "Actor:" not in text
    assert "Object:" not in text
    assert "Communication: professional to patient" in text


def test_blend_embedding_views_returns_normalized_weighted_vectors():
    original = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    normalized = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    blended = pipeline.blend_embedding_views(original, normalized, 0.5)

    expected = np.asarray([[2**-0.5, 2**-0.5], [2**-0.5, 2**-0.5]])
    assert np.allclose(blended, expected)
    assert np.allclose(np.linalg.norm(blended, axis=1), 1.0)


def test_index_auto_uses_and_caches_dual_embedding_views(tmp_path, monkeypatch):
    occurrences = pd.DataFrame(
        [
            {
                **_occurrence(
                    "i1",
                    "r1",
                    "Service failed to send results",
                    failure_mode="not_reviewed",
                ),
                "canonical_issue_original": "Results were not sent",
            },
            {
                **_occurrence(
                    "i2",
                    "r2",
                    "Team failed to review results",
                    failure_mode="not_reviewed",
                ),
                "canonical_issue_original": "Results were not reviewed",
            },
        ]
    )
    pd.DataFrame(
        [
            {
                "report_key": report,
                "report_id": report,
                "report_url": f"https://example.test/{report}",
                "report_date": "2025-01-01",
            }
            for report in ["r1", "r2"]
        ]
    ).to_csv(tmp_path / "01_reports.csv", index=False)
    encoded = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32
    )

    encode_calls = []

    def fake_encode(texts, model_name, *, allow_model_download, batch_size=48):
        encode_calls.append(texts)
        if "Results were not sent" in texts[0]:
            return encoded[:2]
        return encoded[2:]

    monkeypatch.setattr(pipeline, "encode_embeddings", fake_encode)
    args = _index_args()
    args.embedding_mode = "auto"
    args.original_view_weight = 0.5

    metrics = pipeline.build_index(occurrences, run_dir=tmp_path, args=args)

    assert metrics["embedding_mode"] == "dual"
    assert metrics["original_view_weight"] == 0.5
    assert len(encode_calls) == 2
    assert np.array_equal(
        np.load(tmp_path / "02_issue_embeddings_original.npy"), encoded[:2]
    )
    assert np.array_equal(
        np.load(tmp_path / "02_issue_embeddings_normalized.npy"), encoded[2:]
    )
    expected = pipeline.blend_embedding_views(encoded[:2], encoded[2:], 0.5)
    assert np.allclose(np.load(tmp_path / "02_issue_embeddings.npy"), expected)
    meta = json.loads(
        (tmp_path / "02_embeddings_meta.json").read_text(encoding="utf-8")
    )
    assert meta["embedding_mode"] == "dual"
    snapshot = pd.read_csv(tmp_path / "02_embedding_occurrences.csv").fillna("")
    assert snapshot["canonical_issue_original"].tolist() == [
        "Results were not sent",
        "Results were not reviewed",
    ]
    assert "embedding_text_original" in snapshot.columns

    encode_calls.clear()
    updated = occurrences.copy()
    updated.loc[0, "canonical_issue"] = "Service omitted sending results"
    pipeline.build_index(updated, run_dir=tmp_path, args=args)

    assert len(encode_calls) == 1
    assert "Service omitted sending results" in encode_calls[0][0]


def test_normalization_lock_recovers_stale_owner(tmp_path):
    lock_path = tmp_path / "normalization.lock"
    lock_path.write_text(json.dumps({"pid": 999_999_999}), encoding="utf-8")

    with normalization.normalization_lock(lock_path):
        assert lock_path.exists()
        assert json.loads(lock_path.read_text(encoding="utf-8"))["pid"] == os.getpid()

    assert not lock_path.exists()

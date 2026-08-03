#!/usr/bin/env python3
"""Archived: audit proposed secondary automatic-family memberships."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


AUDIT_VERSION = "overlapping-family-attachment-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-summary-csv", required=True, type=Path)
    parser.add_argument("--memberships-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--flagged-attachments", type=int, default=30)
    parser.add_argument("--unflagged-attachments", type=int, default=30)
    parser.add_argument("--maximum-per-family", type=int, default=3)
    parser.add_argument("--core-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def stable_audit_id(family_id: str, child_group_id: str) -> str:
    digest = hashlib.sha256(
        f"{family_id}:{child_group_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"att_{digest}"


def group_balanced_sample(
    frame: pd.DataFrame,
    size: int,
    *,
    maximum_per_family: int,
    seed: int,
) -> pd.DataFrame:
    if size <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if maximum_per_family < 1:
        raise ValueError("Maximum per family must be positive")
    shuffled = frame.sample(frac=1.0, random_state=seed)
    selected: list[int] = []
    family_counts: dict[str, int] = {}
    for index, row in shuffled.iterrows():
        family_id = str(row["family_id"])
        if family_counts.get(family_id, 0) >= maximum_per_family:
            continue
        selected.append(index)
        family_counts[family_id] = family_counts.get(family_id, 0) + 1
        if len(selected) >= size:
            break
    return frame.loc[selected].copy()


def build_audit(
    summaries: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    flagged_attachments: int,
    unflagged_attachments: int,
    maximum_per_family: int,
    core_examples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if summaries["family_id"].astype(str).duplicated().any():
        raise ValueError("Family summary IDs must be unique")
    if memberships.duplicated(["family_id", "child_group_id"]).any():
        raise ValueError("Membership family/child pairs must be unique")
    summary_lookup = summaries.set_index("family_id").to_dict("index")
    core_example_lookup: dict[str, str] = {}
    for family_id, local in memberships[
        memberships["membership_type"].eq("core")
    ].groupby("family_id"):
        examples = local.sort_values(
            "similarity_to_family_centroid", ascending=False
        ).head(max(core_examples, 0))
        core_example_lookup[str(family_id)] = " | ".join(
            examples["prototype_canonical_issue"].astype(str)
        )

    attachments = memberships[
        memberships["membership_type"].ne("core")
    ].copy()
    attachments["target_expansion_flagged"] = attachments["family_id"].map(
        lambda family_id: bool(
            summary_lookup[str(family_id)]["expansion_review_required"]
        )
    )
    flagged = group_balanced_sample(
        attachments[attachments["target_expansion_flagged"]],
        flagged_attachments,
        maximum_per_family=maximum_per_family,
        seed=seed,
    ).assign(audit_stratum="risk_flagged")
    unflagged = group_balanced_sample(
        attachments[~attachments["target_expansion_flagged"]],
        unflagged_attachments,
        maximum_per_family=maximum_per_family,
        seed=seed + 1,
    ).assign(audit_stratum="unflagged_random")
    selected = pd.concat([flagged, unflagged], ignore_index=True)

    rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        family_id = str(row["family_id"])
        primary_family_id = str(row.get("primary_family_id", ""))
        target = summary_lookup[family_id]
        primary = summary_lookup.get(primary_family_id, {})
        rows.append(
            {
                "audit_id": stable_audit_id(
                    family_id, str(row["child_group_id"])
                ),
                "audit_stratum": row["audit_stratum"],
                "target_family_id": family_id,
                "target_family_label_hint": target["family_label_hint"],
                "target_family_prototype": target["family_prototype"],
                "target_core_examples": core_example_lookup.get(family_id, ""),
                "target_expansion_risk_reasons": target[
                    "expansion_risk_reasons"
                ],
                "child_group_id": row["child_group_id"],
                "child_report_count": row["child_report_count"],
                "child_prototype": row["prototype_canonical_issue"],
                "primary_family_id": primary_family_id,
                "primary_family_label_hint": primary.get(
                    "family_label_hint", ""
                ),
                "primary_family_prototype": primary.get(
                    "family_prototype", ""
                ),
                "similarity_to_target": row[
                    "similarity_to_family_centroid"
                ],
                "similarity_to_primary": row.get("primary_similarity", ""),
                "similarity_gap_from_primary": row.get(
                    "similarity_gap_from_primary", ""
                ),
                "attachment_evidence": row["attachment_evidence"],
                "shared_themes": row["shared_themes"],
                "shared_process_stages": row["shared_process_stages"],
                "supports_secondary_family": "",
                "decision_reason": "",
                "review_notes": "",
            }
        )
    queue = pd.DataFrame(rows).sort_values(
        ["audit_stratum", "target_family_id", "audit_id"]
    ).reset_index(drop=True)
    family_diagnostics = summaries[
        summaries["expansion_review_required"].astype(bool)
    ].sort_values(
        ["expanded_report_count", "family_id"], ascending=[False, True]
    ).copy()
    family_diagnostics["family_review_decision"] = ""
    family_diagnostics["family_review_notes"] = ""
    metrics = {
        "audit_version": AUDIT_VERSION,
        "attachment_population": len(attachments),
        "flagged_attachment_population": int(
            attachments["target_expansion_flagged"].sum()
        ),
        "unflagged_attachment_population": int(
            (~attachments["target_expansion_flagged"]).sum()
        ),
        "sample_rows": len(queue),
        "sample_by_stratum": queue["audit_stratum"].value_counts().to_dict(),
        "sample_families_by_stratum": {
            str(stratum): local["target_family_id"].nunique()
            for stratum, local in queue.groupby("audit_stratum")
        },
        "risk_flagged_family_diagnostics": len(family_diagnostics),
        "seed": seed,
        "maximum_per_family": maximum_per_family,
        "interpretation": (
            "The attachment sample is family-balanced and diagnostic. Compare "
            "strata directly, but do not treat it as prevalence-weighted."
        ),
    }
    return queue, family_diagnostics, metrics


def review_markdown(queue: pd.DataFrame) -> str:
    lines = [
        "# Overlapping family attachment audit",
        "",
        "Judge whether each child sub-issue genuinely belongs to the target family as an additional, non-exclusive membership.",
        "",
        "Allowed decisions: `yes`, `no`, or `uncertain`.",
        "",
    ]
    for stratum, local in queue.groupby("audit_stratum", sort=True):
        lines.extend([f"# Stratum: {stratum}", ""])
        for row in local.to_dict("records"):
            lines.extend(
                [
                    f"## {row['audit_id']}",
                    "",
                    f"Target: **{row['target_family_label_hint']}** — {row['target_family_prototype']}",
                    "",
                    f"Target core examples: {row['target_core_examples']}",
                    "",
                    f"Candidate child: **{row['child_prototype']}**",
                    "",
                    f"Primary family: {row['primary_family_label_hint']} — {row['primary_family_prototype']}",
                    "",
                    (
                        f"Similarity: {float(row['similarity_to_target']):.3f}; "
                        f"gap from primary: {float(row['similarity_gap_from_primary']):.3f}; "
                        f"evidence: {row['attachment_evidence']}; "
                        f"shared themes: {row['shared_themes']}; "
                        f"shared stages: {row['shared_process_stages']}."
                    ),
                    "",
                    "Decision:",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summaries = pd.read_csv(args.family_summary_csv).fillna("")
    memberships = pd.read_csv(args.memberships_csv).fillna("")
    queue, diagnostics, metrics = build_audit(
        summaries,
        memberships,
        flagged_attachments=args.flagged_attachments,
        unflagged_attachments=args.unflagged_attachments,
        maximum_per_family=args.maximum_per_family,
        core_examples=args.core_examples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output_dir / "attachment_review_queue.csv", index=False)
    diagnostics.to_csv(
        args.output_dir / "risk_flagged_family_queue.csv", index=False
    )
    (args.output_dir / "attachment_review_packet.md").write_text(
        review_markdown(queue), encoding="utf-8"
    )
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

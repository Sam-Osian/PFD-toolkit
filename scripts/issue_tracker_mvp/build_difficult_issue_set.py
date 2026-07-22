#!/usr/bin/env python3
"""Build a reproducible, stratified difficult-report extraction set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NON_CLINICAL_DOMAINS = {
    "aviation_maritime_rail",
    "environmental_public_health",
    "fire_building_safety",
    "other",
    "policing_public_protection",
    "product_consumer_safety",
    "roads_transport",
    "workplace_public_safety",
}

KNOWN_MISSED_URLS = (
    "https://www.judiciary.uk/prevention-of-future-death-reports/100065-2/",
    "https://www.judiciary.uk/prevention-of-future-death-reports/jacob-brown/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-csv", type=Path, default=Path("all_reports.csv"))
    parser.add_argument(
        "--v1-occurrences",
        type=Path,
        default=Path(
            "artifacts/issue_index_v1/run_20260714_171206/01_issue_occurrences.csv"
        ),
    )
    parser.add_argument(
        "--v2-occurrences",
        type=Path,
        default=Path(
            "artifacts/issue_index_v2/run_20260716_145120/01_issue_occurrences.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/issue_index_v3_difficult"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[
            "dense_previous_cap",
            "remedy_future_risk",
            "non_clinical",
            "ordinary_control",
        ],
        help="Optionally write only selected strata after constructing the full set.",
    )
    parser.add_argument(
        "--report-ids",
        nargs="+",
        help="Optionally write only these report IDs from the constructed set.",
    )
    return parser.parse_args()


def build_set(
    reports: pd.DataFrame,
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reports = reports.drop_duplicates("url", keep="first").copy()
    reports = reports[
        reports["concerns"].fillna("").str.strip().ne("")
        | reports["circumstances"].fillna("").str.strip().ne("")
    ]
    v1_counts = v1.groupby("report_url").size()
    v2_counts = v2.groupby("report_url").size()
    invalid_counts = (
        v2.loc[~v2["source_span_valid"].astype(bool)].groupby("report_url").size()
    )
    report_stats = reports[["url", "id"]].copy()
    report_stats["v1_count"] = report_stats["url"].map(v1_counts).fillna(0).astype(int)
    report_stats["v2_count"] = report_stats["url"].map(v2_counts).fillna(0).astype(int)
    report_stats["v2_invalid_count"] = (
        report_stats["url"].map(invalid_counts).fillna(0).astype(int)
    )

    selected: list[dict[str, object]] = []
    selected_urls: set[str] = set()

    def add(frame: pd.DataFrame, category: str, reason: str, limit: int) -> None:
        for row in frame.itertuples(index=False):
            if row.url in selected_urls:
                continue
            selected.append(
                {
                    "url": row.url,
                    "id": row.id,
                    "difficult_category": category,
                    "selection_reason": reason,
                    "v1_count": row.v1_count,
                    "v2_count": row.v2_count,
                    "v2_invalid_count": row.v2_invalid_count,
                }
            )
            selected_urls.add(row.url)
            if (
                sum(item["difficult_category"] == category for item in selected)
                >= limit
            ):
                break

    dense = report_stats[report_stats["v1_count"] == 8].sort_values(
        ["v2_count", "v2_invalid_count", "url"], ascending=[False, False, True]
    )
    add(dense, "dense_previous_cap", "v1 reached eight-issue ceiling", 20)

    recommendation_urls = set(
        v2.loc[
            v2["claim_status"].isin(
                ["recommended_improvement", "potential_future_risk"]
            ),
            "report_url",
        ]
    )
    missed_with_concerns = report_stats[
        (report_stats["v2_count"] == 0)
        & report_stats["url"].isin(
            reports.loc[reports["concerns"].fillna("").str.strip().ne(""), "url"]
        )
    ]
    remedy = pd.concat(
        [
            report_stats[report_stats["url"].isin(KNOWN_MISSED_URLS)]
            .assign(
                priority=lambda frame: frame["url"].map(
                    {url: position for position, url in enumerate(KNOWN_MISSED_URLS)}
                )
            )
            .sort_values("priority")
            .drop(columns="priority"),
            missed_with_concerns,
            report_stats[report_stats["url"].isin(recommendation_urls)].sort_values(
                ["v2_count", "url"], ascending=[False, True]
            ),
        ]
    ).drop_duplicates("url")
    add(
        remedy,
        "remedy_future_risk",
        "recommendation-only, future-risk, or previously missed concern",
        10,
    )

    nonclinical = v2[v2["subject_domain"].isin(NON_CLINICAL_DOMAINS)].copy()
    nonclinical["fallback"] = (
        nonclinical["subject_domain"].eq("other").astype(int)
        + nonclinical["process_stage"].isin(["other", "unclear"]).astype(int)
        + nonclinical["setting"].isin(["other", "unclear"]).astype(int)
    )
    nonclinical_scores = (
        nonclinical.groupby("report_url")
        .agg(nonclinical_issues=("issue_id", "size"), fallback=("fallback", "sum"))
        .reset_index()
        .merge(report_stats, left_on="report_url", right_on="url", how="inner")
        .sort_values(
            ["fallback", "nonclinical_issues", "url"],
            ascending=[False, False, True],
        )
    )
    add(
        nonclinical_scores,
        "non_clinical",
        "non-clinical taxonomy stress case",
        10,
    )

    ordinary = report_stats[
        report_stats["v1_count"].between(3, 6)
        & report_stats["v2_count"].between(3, 6)
        & report_stats["v2_invalid_count"].eq(0)
        & ~report_stats["url"].isin(selected_urls)
    ].sample(frac=1.0, random_state=seed)
    add(ordinary, "ordinary_control", "ordinary control report", 10)

    manifest = pd.DataFrame(selected)
    if len(manifest) != 50 or manifest[
        "difficult_category"
    ].value_counts().to_dict() != {
        "dense_previous_cap": 20,
        "remedy_future_risk": 10,
        "non_clinical": 10,
        "ordinary_control": 10,
    }:
        raise RuntimeError(
            f"Could not construct the required 50-report strata: "
            f"{manifest['difficult_category'].value_counts().to_dict()}"
        )
    selected_reports = reports.merge(
        manifest[["url", "difficult_category"]], on="url", how="inner"
    )
    selected_reports = (
        selected_reports.set_index("url").loc[manifest["url"]].reset_index()
    )
    return selected_reports, manifest


def main() -> None:
    args = parse_args()
    reports = pd.read_csv(args.reports_csv)
    v1 = pd.read_csv(args.v1_occurrences)
    v2 = pd.read_csv(args.v2_occurrences)
    selected, manifest = build_set(reports, v1, v2, seed=args.seed)
    if args.categories:
        manifest = manifest[manifest["difficult_category"].isin(args.categories)]
        selected = selected[selected["url"].isin(manifest["url"])]
        selected = selected.set_index("url").loc[manifest["url"]].reset_index()
    if args.report_ids:
        manifest = manifest[manifest["id"].astype(str).isin(args.report_ids)]
        selected = selected[selected["url"].isin(manifest["url"])]
        selected = selected.set_index("url").loc[manifest["url"]].reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "difficult_reports.csv", index=False)
    manifest.to_csv(args.output_dir / "difficult_set_manifest.csv", index=False)
    print(manifest["difficult_category"].value_counts().sort_index().to_string())
    print(f"Reports: {args.output_dir / 'difficult_reports.csv'}")


if __name__ == "__main__":
    main()

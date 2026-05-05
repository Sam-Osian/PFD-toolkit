# Issue Tracker MVP (Gemma + BERTopic-style)

Last updated: 2026-05-05

## 1. Goal

Replace broad theme grouping with issue-level tracking:

1. Extract concise issue sentences from each report's `circumstances` + `concerns`.
2. Cluster semantically similar issue sentences across the full archive.
3. Label clusters with short, plain-English issue names.
4. Preserve unclustered issues as unique/sparse signals.
5. Surface recurrence over time and related reports in-app.

## 2. Scope

This directory defines the MVP contract for:

1. Data flow and jobs.
2. Data model additions.
3. Gemma extraction and label prompts.
4. Clustering defaults and tuning path.
5. UI/API contract.
6. Evaluation and rollout guardrails.

## 3. Non-Goals (MVP)

1. Perfect canonical taxonomy from day one.
2. Full deprecation of existing collections/themes before quality gates.
3. Fully automated merge/split governance without human review.

## 4. Suggested Phases

1. **Phase A: Offline shadow pipeline**
   1. Run extraction + clustering offline against full archive.
   2. Measure precision/coherence and iteration speed.
2. **Phase B: Read-only in app**
   1. Add "Issues (Beta)" views alongside existing collections.
   2. Keep theme views as fallback.
3. **Phase C: User-defined grouping**
   1. Add "group similar issues" interaction via embedding similarity thresholds.
4. **Phase D: Default switch**
   1. Make issue tracker primary navigation once quality criteria are met.

## 5. Document Map

1. `PIPELINE_SPEC.md`
2. `DATA_MODEL_PROPOSAL.md`
3. `PROMPTS_GEMMA4.md`
4. `CLUSTERING_TUNING.md`
5. `UI_API_CONTRACT.md`
6. `EVALUATION_ROLLOUT.md`

# Issue extraction schema v2 evaluation

Status: v2 contract implemented and smoke-validated  
Last updated: 2026-07-16

## Why v2 was required

The 1,500-report v1 sample contained 6,996 issue occurrences. Controlled-field
coverage was uneven:

| Field | `other` or `unclear` |
| --- | ---: |
| Process stage | 2,522 (36.0%) |
| Setting | 776 (11.1%) |
| Failure mode | 415 (5.9%) |
| Affected population | 246 (3.5%) |
| Subject domain | 233 (3.3%) |
| System mechanism | 218 (3.1%) |

The original process vocabulary modelled a mostly clinical pathway. It had no
adequate values for prevention, policy, service provision, workforce management,
training delivery, procurement, regulation, professional practice, or public-
safety functions. The setting and subject-domain vocabularies similarly collapsed
aviation, maritime, rail, fire, building, product, policing, and public-health
issues into broad fallback values.

The v1 extraction limit was also saturated: 226 of 1,497 reports with issues
(15.1%) returned exactly the maximum eight issues.

## Contract changes

`issue_schema_v2.json`:

1. Raises schema capacity and the default extraction limit to 24. An intermediate
   live check found that 2 of 22 cap-hit reports still saturated a limit of 12,
   and both then saturated 16 with additional grounded issues, so both lower
   operational limits were rejected.
2. Expands process, domain, setting, failure-mode, mechanism, and population values.
   The final audit added `operational_execution` for required frontline actions
   and `housing_service` for provider-wide housing failures; these replaced
   concentrated `other` values found in the clean checkpoint.
3. Adds `communication_direction` to distinguish who should communicate to whom.
4. Adds `essential_qualifier_text` for reusable distinctions such as discharge,
   falls risk, Mental Health Act, or Category 2 response.
5. Keeps source-grounded original wording separate from reusable canonical text.

Validation warnings are now occurrence-specific rather than copied from every
issue in the same report. Source recovery accepts only an exact report sentence
supported by an exact original issue statement, an ellipsis-corrupted quote, or
a contiguous, at least 90%-similar near-exact transcription; loose paraphrases
remain invalid.

The prompt now requires one failure per issue, problem rather than remedy framing,
removal of names and local details, explicit preservation of essential qualifiers,
and a final within-report consolidation pass.

The embedding text uses canonical issue, object, qualifier, failure mode, process
stage, and communication direction. The noisier inferred system mechanism is no
longer included in the semantic identity text.

## Final ten-report cap-hit comparison

The final v2 smoke test used the same seeded ten reports that each produced eight
v1 issues:

| Measure | v1 | v2 |
| --- | ---: | ---: |
| Issue occurrences | 80 | 92 |
| Mean issues per report | 8.0 | 9.2 |
| Valid source spans before final deterministic repair | 92.5% | 98.9% |
| Weak failure mode before final deterministic repair | 6.2% | 1.1% |
| Weak subject domain | 1.2% | 0.0% |
| Weak process stage | 28.8% | 0.0% |
| Weak system mechanism | 2.5% | 0.0% |
| Weak setting | 10.0% | 0.0% |
| Non-failure canonical framing | 1 | 0 |

No v2 report hit the new limit. Issue yield increased by 15%, despite two reports
returning fewer issues because the final consolidation pass removed duplication.
Forty occurrences (43.5%) used an essential qualifier and 19 communication issues
received a specific direction rather than collapsing into undifferentiated text.

The single remaining weak value was an issue explicitly framed as “unclear
responsibility”; validation now maps that safely to `ambiguous`. The single
invalid source span used a paraphrase while its `issue_statement_original` was
an exact report sentence; the validator now selects that exact sentence. Both
repairs are deterministic and covered by unit tests.

Earlier manual comparison found useful additional issues and better non-clinical
classification. It also found two duplicate pairs and local hospital details in
one report. A refined consolidation and canonicalisation prompt removed those
duplicates and details while retaining distinct operational failures.

Two especially dense reports were also tested independently at limits of 12, 16,
and 24. Both saturated 16 but naturally stopped at 17 and 18 under a limit of 24,
with additional grounded, distinct failures. This is why 24 remains the contract
ceiling even though none of the seeded ten reports reached it.

## What the full manual group review showed

The review did not support simply lowering every similarity threshold. It found
simultaneous fragmentation and over-merging:

- Repeated issue families were fragmented by incidental wording or narrow
  qualifiers. Examples included ambulance response categories, ambulance
  offloading/handover, clinical handover, falls-risk assessment, generic risk
  assessment, clinical documentation, observation documentation, escalation of
  deterioration, and failure to implement learning from investigations.
- Twelve of 99 high-recall candidates over-merged materially different failures.
  Examples included medication supply at discharge versus dose administration;
  contacting emergency services versus ambulance response delay; record creation
  versus record review; general psychiatric assessment versus Mental Health Act
  assessment; and hospital transfer, inter-hospital transfer, and ambulance
  handover.
- Concrete, consistently phrased operational failures were captured best:
  ambulance delays, missing assessments, inadequate handovers, documentation
  failures, observation failures, and unimplemented incident learning.
- Heterogeneous non-clinical issues and failures expressed through local facts or
  remedies were captured least consistently. The v1 clinical-pathway taxonomy
  amplified this problem by assigning many such occurrences to `other`.

These findings explain the v2 identity facets. `failure_mode`, `process_stage`,
`communication_direction`, and `essential_qualifier_text` preserve distinctions
that should block a merge, while canonical text is instructed to remove names,
dates, and supporting causes that should not create separate issue types.

The grouping implication is asymmetric: use richer facets and reviewed repairs
to prevent false merges, then recover missed members through a domain-gated,
margin-tested prototype assignment. A further global threshold reduction would
increase the exact over-merges seen in review.

## Prototype-assignment experiment

The domain-gated `0.84` similarity and `0.02` margin experiment accepted 74
previously ungrouped occurrences and provisionally promoted 41 emerging groups.
Manual review found:

| Result | Groups |
| --- | ---: |
| Semantically correct assignment | 25 |
| Unique new recurring type | 21 |
| Correct but duplicate of existing type | 4 |
| Incorrect promotion | 16 |

Raw promotion precision was therefore 61.0%, and the reviewed recurring total
rose from 76 to 97 rather than the provisional 117. A stricter automatic rule
of similarity `0.845` and margin `0.03` retained 16 promotions with 87.5%
assignment precision, though one was a duplicate existing type. Prototype
assignment must remain review-gated until richer v2 facets are available.
The command now defaults to the reviewed stricter thresholds (`0.845` similarity,
`0.03` margin) rather than the exploratory lenient values.

## Interpretation

The benchmark establishes that the eight-issue cap and v1 controlled vocabulary
were suppressing extraction. It does not estimate the final recurring-type count:
that requires a complete v2 extraction and embedding pass over the same 1,500
reports. The next production-scale run should use v2 for the whole sample rather
than mixing v1 and v2 occurrences, then apply reviewed group repair and the
strict prototype-assignment pass.

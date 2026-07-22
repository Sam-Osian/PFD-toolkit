# Actor/object normalization evaluation

## Decision

Keep `issue_object` as the neutral name. It covers actions, information,
services, policies, systems, equipment, environments, decisions, and duties in
both medical and non-medical PFD reports. “Safeguard” would incorrectly imply a
clinical or risk-control interpretation for every occurrence.

Do not add actor-based hard gates to grouping. Instead, surface actor and object
once in the normalized sentence, alongside the existing failure-state, process,
theme, and communication-direction embedding facets. Do not append separate
actor/object facets after embedding the actor/object-explicit sentence: the first
A/B experiment showed that this double weighting over-emphasised broad roles such
as `provider organisation`. This lets semantic grouping use agency and direction
while still allowing the embedding model to recognise legitimate cross-role
parallels.

## Difficult-set result

The normalization stage was run over the 50-report difficult set produced at
`artifacts/issue_index_v3_difficult_runs/run_20260720_125016`:

- 493 of 493 occurrences normalized; zero unresolved failures.
- 490 canonical sentences changed; the original is retained in
  `canonical_issue_original`.
- Every one of the 356 stated actor roles appears explicitly in its canonical
  sentence. The remaining 137 have both `responsible_actor_type=not_stated` and
  no supported source actor, so passive wording is retained rather than inventing
  responsibility.
- No non-`not_stated` actor type was discarded, and no source actor text was
  discarded.
- All 493 rows have an `issue_object`; there are 475 distinct values. Median
  object length is four words (maximum eleven).
- Median canonical length is thirteen words (maximum twenty-five under the
  twenty-six-word cap).
- The final deterministic review queue contains one benign single-word object,
  `documentation`. It is specific enough in its sentence and does not require a
  model rerun.

The first pilot revealed that an evidence-only interpretation discarded actor
information when the short quote was passive. The corrected rule treats an
already extracted non-`not_stated` actor type as supported schema information and
uses its generic label when no narrower role is justified. This reduced
`responsible_actor_role=not stated` from 196 to 137 without inventing a named
actor.

## Direction and non-medical coverage

Manual review found that contact and follow-up failures retained service agency:
for example, a healthcare team “failed to invite or contact” an individual,
rather than the individual being described as disengaged. No normalized sentence
in the difficult set used `disengagement` language.

Non-medical examples produced meaningful objects and actor-explicit sentences
for policing, probation and prisons, education, housing, workplaces, water
recreation, agriculture, product regulation, and dog-control legislation. This
supports retaining the broad definition of object rather than a medicalised
“safeguard” field.

## Remaining evaluation

`responsible_actor_role` intentionally remains a short normalized free-text role,
alongside the controlled `responsible_actor_type`. Its 86 distinct values retain
useful distinctions such as police, prison service, local authority, employer,
manufacturer, event organiser, and healthcare team. Before adopting a smaller
controlled role vocabulary, compare grouping precision/recall with and without
that lexical detail.

The next meaningful test is a paired grouping comparison on the difficult set:
embed the original and normalized occurrence files using identical thresholds,
then manually inspect newly joined and newly separated pairs—especially contact,
handover, multi-agency communication, and similarly worded failures with opposite
agency.

## Paired grouping comparison

The paired comparison used Qwen3-Embedding-8B for both representations and the
same four grouping configurations. Actor and object occur once in the normalized
canonical sentence; an earlier experimental variant that appended them again as
facets was rejected because generic actors such as `provider organisation` were
overweighted.

At the `recall` configuration (top-k 40, edge 0.84, split 0.87, centroid 0.83):

| Representation | Edges | Recurring | Emerging | Assigned | Ungrouped |
|---|---:|---:|---:|---:|---:|
| Original sentence | 74 | 2 | 15 | 83 | 410 |
| Actor/object-normalized sentence | 58 | 1 | 14 | 57 | 436 |
| 50/50 soft embedding blend | 170 | 2 | 21 | 137 | 356 |

The counts alone are misleading. One original recurring group was incorrectly
merged: medical-record omissions were grouped with failure to ensure significant
information was considered by medically qualified staff. Its second recurring
group—omitted medication administration—was coherent. The normalized recurring
group—key information omitted during clinical/shift handovers—was coherent, but
normalization lost the medication group.

The 50/50 soft blend recovered both coherent recurring types and excluded the
false medical-record recurrence. Manual review of its recurring and emerging
groups found 15 of 23 fully coherent (65.2%), including both recurring groups;
the other eight were over-merged and require split/rejection. The blend is
therefore the strongest
candidate-generation representation, not evidence that model similarity can
replace the review/repair stage.

The practical implication is to retain two semantic views rather than force one
sentence to serve every purpose:

1. the original concise formulation preserves broad failure equivalence;
2. the normalized formulation preserves actor, object, direction, and agency;
3. normalized vector averaging combines the views as a soft signal, without an
   actor gate or categorical veto.

Before changing the production default, run this blended comparison over the
1,500-report sample and review all resulting recurring groups. The difficult set
supports a 50/50 starting weight and the `recall` thresholds, but is intentionally
heterogeneous and too small to establish final production parameters alone.

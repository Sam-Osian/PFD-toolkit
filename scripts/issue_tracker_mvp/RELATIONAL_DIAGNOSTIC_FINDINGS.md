# Relational diagnostic findings

## Scope

These findings describe the deterministic diagnostic pass over
`guarded_v22`. The pass does not alter groups and makes no LLM calls. Its
purpose is to distinguish likely false splits from possible false joins and to
localize each confirmed error to a pipeline stage.

The generated outputs contain 1,619 possible duplicate-group leads and 1,298
possible contaminated-group leads. These counts are deliberately broad and
must not be read as error counts. They are ranked queues for spot checking.

## Initial duplicate-queue spot check

The leading results show that cross-group evidence can indicate either a false
split or an overly broad accepted edge. This is why the queue must not merge
groups automatically.

| Groups | Assessment | Pipeline interpretation |
|---|---|---|
| `rel_00040` / `rel_00170` | Likely duplicate: omitted or inadequate suicide-risk assessment | The all-pairs veto overreacts to failure-state variation. |
| `rel_00300` / `rel_00339` | Unsafe to merge: generic risk-assessment members connect private-maternity and bed-rail concerns | Extraction/normalization has lost the assessed hazard for some members; generic objects permit false-positive edges. |
| `rel_00026` / `rel_00035` | Likely duplicate at a broad record-keeping granularity | Object mismatch between nursing and medical records blocks otherwise extensive support. |
| `rel_00012` / `rel_00267` | Not the same precise issue: slippery/subsided road surface versus unsafe geometry and lighting | Pair scoring accepts generic road-safety similarity without a common corrective object. |
| `rel_00014` / `rel_00442` | Likely duplicate at a mental-health-assessment granularity | One group describes a questioning method rather than a different obligation. |
| `rel_00049` / `rel_00640` | Likely duplicate: failure to recognise clinical deterioration or severity | Object mismatch treats related condition wording as contradictory. |
| `rel_00012` / `rel_01406` | Not the same precise issue: road markings/surface versus defective road studs | Pair scoring is too tolerant of shared road-maintenance context. |
| `rel_00006` / `rel_00380` | Likely duplicate: inadequate first-aid/CPR training | Specific training-object variants trigger an all-pairs veto. |
| `rel_00313` / `rel_00352` | Likely duplicate: delayed ambulance-to-hospital handover | Counterparty normalization inconsistencies block consolidation. |
| `rel_00043` / `rel_01316` | Likely duplicate: inappropriate or unreduced speed limits | Failure-state wording blocks consolidation of the same corrective obligation. |

This small review therefore contains both sides of the problem: seven likely
false splits, two clear examples of false-positive cross-group edges, and one
generic risk-assessment case that is unsafe to merge without examining member
specificity.

## Contamination checks

The contamination queue successfully retains known problem cases:

- `rel_00336`, the rejected mixture of clinical follow-up and patient letters,
  is flagged because its prototype is an accepted-edge bridge and another
  member has low prototype fit;
- `rel_00011`, the capacity group containing staffing and physical-design
  concerns, is flagged through multiple compound members with distinct object
  vocabularies;
- `rel_00209`, the naloxone group containing generic and depot-medication
  concerns, is flagged as a prototype bridge with a low-fit member.

It also flags `rel_00108`. That group is an important negative control: the
pipeline currently groups the members together, and the intended interpretation
accepts review of records, history, clinical information, and the patient's
condition as one broad review concern. Its diagnostic flag is not a confirmed
false positive and must not drive a split.

Comparison with the existing 60-group manual assessment further confirms that
the queue is a prioritizer rather than a classifier. Known rejected groups
appear at ranks 24, 210, 289, 510, and 614, but some accepted broad groups also
rank highly because compound wording and prototype-anchored graph structure
are not errors by themselves.

## Guarded v23 consolidation experiment

The decisions were encoded in `relational_regression_cases_v1.json` and are
evaluated by stable issue membership rather than unstable group IDs. The first
experiment relaxed conflicts inside the original consolidation pass. Although
it recovered the target duplicates, it changed merge order and retained only 9
of 11 members from the user-confirmed broad `rel_00108` group. That approach
was rejected.

The adopted v3 method retains the original strict consolidation as phase one.
A second additive phase may merge whole existing recurring groups but cannot
split or repartition them. It requires:

- ordinary accepted-edge coverage of at least 0.65 on the smaller group;
- conflicts on no more than 0.15 of all cross-member pairs;
- non-context object-token Jaccard of at least 0.50 on supporting edges; and
- qualifying distinctive-object support for at least 0.45 of both groups.

Words such as `road`, `safety`, `clinical`, and `medical` count as context, not
as a corrective object. The final `guarded_v23_candidate` run made eight
additive consolidations, producing seven combined recurring groups. It reduced
the recurring-group count from 1,426 to 1,418 while preserving all 8,238
recurring occurrences and the same 3,806 covered reports.

The resulting combined groups cover:

- delayed ambulance response and missed ambulance-response targets;
- general record-keeping variants;
- three fluid-balance-chart variants;
- delayed calling or summoning an ambulance;
- communication between clinical staff or professionals;
- emergency/category response-time targets; and
- inaccurate or incomplete discharge summaries.

All active regression expectations passed:

- the three established duplicate pairs merged completely;
- the private-maternity/bed-rail risk-assessment groups remained separate;
- the road-surface, road-geometry/lighting, and road-stud controls remained
  separate;
- all 11 members of `rel_00108` stayed together;
- `rel_00336` and `rel_00011` gained no additional members; and
- the ambiguous naloxone group remained unchanged.

This change addresses high-confidence false splits. It intentionally does not
auto-split possible false-positive groups. The contamination queue remains the
appropriate source of examples for developing a separate, conservative rule;
the earlier automatic refinement experiment did not generalize well enough to
be promoted.

The continuing regression set should represent both desired granularities:

- merge the confirmed ambulance-response, record-keeping, fluid-balance-chart,
  ambulance-handover, first-aid-training, and suicide-risk-assessment duplicates;
- keep distinct road surface, road geometry/lighting, road marking, and road
  stud failures;
- keep `rel_00108` broad;
- remove or split confirmed alien members of `rel_00336` and `rel_00011`.

Generic-object cross edges remain a separate correction. The additive merge
guard prevents them from consolidating groups, but it does not remove an
already accepted false-positive edge or repair an already contaminated group.

## Fifty-candidate membership review

The first 50 ranked duplicate candidates were reviewed at membership level in
`duplicate_candidate_spot_check_v1.csv`. Prototype labels alone were not used:
low-fit members, action/object fields, and representative report evidence were
examined before classifying each pair.

The results were:

- 28 pairs (56%) are suitable for whole-group consolidation;
- eight pairs (16%) have the same recurring core but at least one contaminated
  source group, so whole-group merging would preserve known false positives;
- seven pairs (14%) are related but operationally distinct;
- six pairs (12%) are generic-object contamination rather than duplicates; and
- one pair (2%) is clearly separate.

Thus 72% share a recurring core, but only 56% are immediately safe to merge.
The conservative v23 candidate merges two of the 28 safe pairs in this top-50
set. Its precision-oriented improvement is therefore real, but its observed
recall on these high-evidence analyst decisions is only 7.1%.

Among the 28 safe duplicates, the recorded backend causes are 13 object-identity
overconstraints, seven counterparty-role overconstraints, five medoid or
merge-order failures, two failure-state overconstraints, and one communication-
direction normalization error.

No cosine or coverage threshold cleanly separates the decisions. Median
action-object similarity is 0.877 for safe duplicates, 0.880 for related but
distinct pairs, and 0.906 for contaminated generic-object pairs. Generic
contamination can therefore look *more* semantically similar than a true
duplicate. Further threshold relaxation is not supported.

## Normalization v3 consequence

The smallest upstream correction is to preserve the target or qualifier of a
generic process. Normalization v3 now requires objects such as `suicide risk
assessment`, `bed rail entrapment risk assessment`, `community mental health
referral`, and `abnormal blood result follow-up` instead of bare `risk
assessment`, `referral`, or `follow-up` when the source evidence supports the
distinction. Bare process objects are deterministically flagged with
`underspecified_object_target_review`. This changes the existing normalization
request rather than adding an LLM call.

In guarded v22, 915 of 28,765 eligible occurrences (3.18%) have no
discriminative object tokens. They include 372 of 8,238 recurring occurrences
(4.52%). A targeted repair selection has been generated at
`diagnostics/generic_object_repair_selection.csv`; it allows the current corpus
to be backfilled without renormalizing the other 27,850 eligible occurrences.

This repair is expected primarily to reduce false-positive bridges and make
later consolidation safer. It does not, by itself, solve all synonym-level
false splits. Those should be reassessed after the generic-object pilot rather
than obscured by another global threshold change.

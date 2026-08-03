# Archived automatic issue-family discovery: first diagnostic

## Question

Can broad, user-trackable issues be discovered automatically by grouping the
existing precise recurring issue groups, without manually defining each parent
issue?

## Method

`discover_issue_families.py` uses the existing normalized occurrence
embeddings and guarded relational group assignments. It:

1. averages occurrence embeddings into one unit-normalized centroid per
   recurring child group;
2. applies average-linkage cosine clustering to those child centroids;
3. compares several broader family similarities;
4. leaves precise child membership unchanged;
5. counts the deduplicated union of reports represented by each family; and
6. emits model-free label hints, representative children, questionable
   children, and a review packet.

No extraction, normalization, embedding, generative-model call, or manually
specified family definition is required.

## Full-corpus result

The input contained 1,426 recurring relational child groups.

| Similarity | Families | Grouped children | Singletons | Largest family |
|---:|---:|---:|---:|---:|
| 0.78 | 64 | 1,394 | 32 | 257 |
| 0.80 | 94 | 1,381 | 45 | 215 |
| 0.82 | 118 | 1,355 | 71 | 107 |
| 0.84 | 156 | 1,314 | 112 | 91 |
| 0.86 | 188 | 1,244 | 182 | 56 |

The 0.82 result is the initial broad review view, not a selected production
threshold. Examples include:

| Automatic family hint | Child groups | Deduplicated reports | Largest child |
|---|---:|---:|---:|
| Records information — information record management | 107 | 591 | 52 |
| Communication handover — discharge | 82 | 425 | 32 |
| Clinical assessment care — assessment | 79 | 423 | 39 |
| Communication handover — handover | 88 | 381 | 16 |
| Training competence — training delivery | 62 | 288 | 40 |
| Staffing capacity — workforce management | 28 | 258 | 116 |
| Clinical assessment care — care coordination | 18 | 113 | 26 |

The records family is a strong proof of concept. Its central and peripheral
children consistently concern creating, completing, retaining, accessing,
reading, transferring, or maintaining health and care records. Its 591-report
union is 11.4 times the size of its largest precise child.

## Limitations found

One global cut of an agglomerative hierarchy is not sufficient for the final
product:

- At 0.82, some families are broad umbrellas. The discharge family also absorbs
  general family communication, follow-up, safety-netting, and a small number
  of referral concerns.
- At 0.84, those umbrellas split more cleanly, but coherent families such as
  records fragment into separate creation, access, observation-recording, and
  availability families.
- Continuity and coordination is inherently cross-cutting. A coherent
  113-report care-coordination family emerges, but related handover,
  inter-organisation information-sharing, and discharge-transition children
  sit in neighbouring families. Exclusive clustering cannot express that
  overlap.
- Family label hints reuse dominant extracted facets. They are useful for
  inspection but are not yet suitable public labels.
- Only already-recurring child groups are included. Isolated and emerging
  occurrences have not yet been retrieved, so family report counts remain
  lower bounds on what the existing extraction can support.

## Decision

The central hypothesis is supported: grouping precise child issues can recover
substantially more credible report volume without merging the child issues
themselves.

Do not select one global hierarchy cut as the production family taxonomy.
The next experiment should retain tight automatic family cores while allowing
non-exclusive attachment of neighbouring child groups at multiple resolutions.
Only after that structure is audited should isolated and emerging occurrences
be attached.

## Non-exclusive expansion follow-up

`expand_issue_families.py` retains the 0.84 families as tight, exclusive cores
and proposes secondary child memberships using existing evidence only. A
secondary attachment must:

- have cosine similarity of at least 0.92 to the alternative core centroid;
- be no more than 0.04 below its primary-core similarity; and
- either join the same broader 0.82 cluster or share both an extracted theme
  and an operational process stage with the alternative core.

At this conservative diagnostic setting, the full corpus produced 170
secondary memberships across 145 child groups. No strict singleton passed the
rule. Examples were:

- the main records core increased from 512 to 553 deduplicated reports;
- the care-coordination core increased from 93 to 126 reports, adding specific
  children for joined-up care, unclear responsibility, isolated working, and
  communication between those responsible for care; and
- the main handover core increased from 381 to 402 reports.

The result demonstrates useful non-exclusive recovery, but it is still a
review queue rather than a production assignment. Thirty-four families are
flagged: ten more than doubled in report count, two gained more attachments
than core children, and 28 obtain most of their proposed additions across the
0.82 hierarchy boundary (some families trigger more than one reason). The last
category includes potentially valuable cross-cutting families such as records
and continuity, but keeps that judgment visible. Secondary children expand
recall but do not contribute to the automatically generated family label, so
an attachment cannot silently redefine its parent family.

## Attachment audit

A deterministic, family-balanced audit sampled 60 of the 170 conservative
secondary proposals: 30 from risk-flagged families and 30 from unflagged
families, with no more than three attachments from one family. An internal
evidence-based review found:

| Stratum | Accepted | Rejected | Attachment precision | Report-weighted precision |
|---|---:|---:|---:|---:|
| Risk-flagged | 21 | 9 | 70.0% | 44.1% |
| Unflagged | 20 | 10 | 66.7% | 66.8% |
| Overall | 41 | 19 | 68.3% | 54.3% |

The family-level risk flag did not distinguish correct from incorrect
attachments. Rejections were mainly different neighbouring issues (11), plus
wrong action (3), wrong direction (2), and wrong failure state (2). The low
report-weighted result in the flagged stratum was driven particularly by
high-volume ambulance-response groups incorrectly attaching to a family about
calling emergency services.

Secondary assignments should therefore not be promoted automatically and the
pipeline should not yet expand into isolated occurrences. Similarity plus
generic facet agreement remains insufficient at family boundaries. The next
bounded experiment should adjudicate proposed attachments against the target
core examples while explicitly preserving action, direction, and failure-state
distinctions. The 60 reviewed decisions provide a fixed calibration set for
that experiment. This review remains internal and should not be described as
independent domain-expert validation.

## Core-example adjudication calibration

`adjudicate_overlapping_family_attachments.py` implements the bounded next
experiment. It enriches each proposed attachment with relational signatures
from the precise child prototype, shows the model five central tight-core
examples, emits schema-constrained decisions, and checkpoints every batch. Four
global deterministic vetoes protect distinctions exposed by the audit:

- summoning an ambulance versus ambulance response or attendance;
- performing observations versus recording observations;
- defective records versus unavailable or inaccessible records; and
- staff capacity versus training delivery.

The deterministic layer fired on seven of the 60 calibration cases and all
seven were genuine rejections. The model adjudicated the remaining 53. A first
pass retained 40 attachments, of which 33 were supported:

| Calibration stage | Retained | Correct retained | Precision | Report-weighted precision | Recall |
|---|---:|---:|---:|---:|---:|
| Deterministic veto + core-example model | 40 | 33 | 82.5% | 73.9% | 80.5% |
| Plus conservative independent model verification | 28 | 23 | 82.1% | 71.7% | 56.1% |

Neither result meets the predeclared 90% attachment and report-weighted
precision gates. The second model pass removed correct and incorrect cases at
similar rates; repeating model judgment with a more skeptical prompt is not a
useful production control.

The remaining false acceptances were not random. They again collapsed distinct
objects or actions under generic assessment/review language: mental-capacity
assessment versus mental-health assessment, care-plan review versus patient
assessment, performing observations versus assessment, and generic clinical
review versus narrower diagnostic or service-provision cores.

Therefore the 170 secondary proposals have **not** been promoted or used to
expand report counts, and isolated occurrences remain out of scope. The next
methodological step is a small global relation-compatibility layer for action,
object, direction, and failure state, learned and tested on separate calibration
and holdout samples. It should not become a list of bespoke inclusion and
exclusion rules for every public issue.

## Direct parent-recall audit

The next diagnostic changed the question from attachment precision to actual
parent recall. `build_parent_recall_audit.py` uses each selected parent's tight
core to retrieve reports from all 28,870 extracted occurrences. It combines
core-derived semantic similarity with independent report-level theme flags and
excludes reports already represented by the parent core. No bespoke issue
definition or inclusion/exclusion list is used.

Two parents were reviewed: clinical records, the strongest large family, and
continuity/coordination, the known undercounted family. The main enriched packet
sampled high-semantic, source-theme, and boundary candidates. A separate random
source-theme holdout tested whether the first packet's semantic enrichment was
hiding extraction failures. The packets were disjoint.

| Parent | Current core reports | Reports reviewed | Confirmed missing reports | Isolated/pair occurrence | Other recurring child |
|---|---:|---:|---:|---:|---:|
| Clinical records and record keeping | 512 | 50 | 40 | 30 | 10 |
| Continuity and coordination of care | 93 | 50 | 38 | 27 | 11 |

These yields are diagnostic and must not be extrapolated to the corpus: the
samples were deliberately enriched for likely misses. They nevertheless prove
that both published parent counts are material lower bounds. Adding only the
reviewed, confirmed reports would increase records from 512 to at least 552 and
continuity/coordination from 93 to at least 131.

All 78 confirmed misses had an identifiable extracted occurrence. None required
recovering a concern absent from extraction. Fifty-seven (73%) were stranded in
isolated or paired relational groups; 21 were recurring children outside the
parent core. The misses spanned 74 distinct relational groups, so this is broad
fragmentation rather than one omitted high-volume child.

The audit also corrected an important legacy-data issue: report numbers can be
blank or duplicated and are unsafe join keys for source concern text. Canonical
report URLs are used instead.

## Revised decision

Do not build the proposed relational-compatibility layer next. It protects the
boundary between neighbouring recurring children, but the dominant recall loss
occurs earlier: parent discovery excludes isolated and paired occurrences by
construction.

The next implementation should retrieve individual extracted occurrences
directly into a stable parent core, deduplicate them to reports, and leave precise
child grouping untouched. Core similarity or source-theme agreement alone is not
yet precise enough for automatic publication in this small audit (observed
positive yields remained below 90% for most combinations), so the direct
retriever still needs a calibrated acceptance/review boundary. That is a smaller
and better-targeted problem than further engineering overlapping child-family
attachments.

This was an internal evidence-based review, not independent domain-expert
validation.

## Direct report-to-parent retrieval

`adjudicate_parent_recall_candidates.py` implements the targeted follow-up. It
starts from reports outside each tight core that have both an independent source
theme signal and occurrence-to-core cosine similarity of at least 0.82. The
model sees 12 central tight-core examples, the report's extracted issues, and
the original concern text. Only a supported, high-confidence decision is
automatically accepted; lower-confidence support is reserved for review.

A fresh, disjoint 30-report holdout was reviewed before corpus assignment. It
contained 27 genuine parent matches. The automatic acceptance boundary retained
22 reports, 21 correctly:

| Parent | True matches | Auto-accepted | Correct accepted | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Clinical records and record keeping | 13 | 10 | 10 | 100.0% | 76.9% |
| Continuity and coordination of care | 14 | 12 | 11 | 91.7% | 78.6% |
| Overall | 27 | 22 | 21 | 95.5% | 77.8% |

Both parents passed the predeclared 90% precision gate. This remains internal
evidence-based validation rather than independent domain-expert review.

The same fixed rule was then applied to the full eligible candidate population.
All 1,858 candidates received a validated decision: 1,258 high-confidence
accepts and 600 rejects, with no pending or failed batches. The deterministic
merge in `build_direct_parent_assignments.py` preserves every tight-core report,
adds only accepted reports outside that core, and checks for duplicate
parent/report assignments.

| Parent | Tight-core reports | Direct additions | Final reports | Increase over core |
|---|---:|---:|---:|---:|
| Clinical records and record keeping | 512 | 799 | 1,311 | 156.1% |
| Continuity and coordination of care | 93 | 459 | 552 | 493.5% |

Of the 1,258 additions, 847 came from isolated or paired occurrences and 410
from other recurring child groups; one continuity report had no recurrence
status. This directly addresses the dominant failure found in the recall audit
without changing the precise child groups or introducing per-issue exclusion
rules.

These final counts are conservative retrieval assignments, not estimates of
total corpus prevalence. Reports outside the independent theme filter or below
0.82 similarity were not considered, and plausible lower-confidence cases were
not automatically added. The next useful validation is a domain-expert sample
from the 1,258 additions and from the 600 rejected boundary candidates before
these assignments are published on the website.

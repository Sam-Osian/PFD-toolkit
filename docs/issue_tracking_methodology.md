# Methodology for identifying recurring issues in Prevention of Future Death reports

> **Proposed next methodology:** See
> [Methodological proposal: a registry of recurring coroner concerns](issue_registry_methodological_proposal.md)
> for the proposed transition from global clustering to stable-registry
> retrieval and bounded adjudication.

## Summary

The issue tracker converts narrative Prevention of Future Death (PFD) reports
into an auditable index of concrete safety concerns and then identifies concerns
that recur across different reports.

The method is deliberately more specific than topic modelling. It does not ask
whether two reports are broadly about the same subject, such as medication,
mental health, communication, or training. It asks whether two evidence-backed
issue occurrences describe substantially the same failed obligation:

> **Who** failed to **do what**, to or for **whom**, and **what object, process,
> system, or duty** was affected?

This distinction matters. A patient declining contact and a service failing to
contact a patient may both be described using language about “disengagement”,
but they have opposite agency and are not the same issue. Likewise, failing to
use a defibrillator is not the same as being unable to access one, and failing
to make a referral is not the same as failing to respond to a referral.

The pipeline therefore combines:

1. structured large-language-model (LLM) extraction from each report;
2. exact evidence checking against the report text;
3. relational normalization into actor, action, object, counterparty, and
   direction;
4. multiple weighted semantic representations;
5. nearest-neighbour retrieval;
6. explicit compatibility bonuses, penalties, and cannot-link rules;
7. conservative prototype-anchored grouping; and
8. human review and publication controls.

The success criterion is not the largest possible number of recurring groups or
the assignment of most reports to a group. PFD reports concern heterogeneous
deaths, systems, and circumstances. It is entirely reasonable for an issue to
occur in only one report. Success means a relatively tight set of recurring
groups whose members express the same concrete concern and whose observed
prevalence is credible.

## 1. What is being indexed?

The basic unit is an **issue occurrence**, not a report and not a whole report
summary.

A report can contain:

- no extractable systemic issue;
- one issue;
- several independently actionable issues; or
- several statements that are only restatements, examples, consequences, or
  remedies for one underlying issue.

The extraction stage aims to separate independently correctable failures while
avoiding artificial fragmentation. For example, missing assessment and delayed
assessment remain separate because the failure state differs. Medication supply
and medication administration remain separate because the operational action
differs. Conversely, an example of a missing assessment and the recommendation
to introduce an assessment form should not become two issue occurrences.

A **recurring issue group** is then a set of equivalent issue occurrences found
in at least three distinct reports. Multiple occurrences in one report cannot,
by themselves, establish recurrence.

This gives the pipeline two different recall questions:

1. Did extraction recover the substantive concerns present in each report?
2. Did linkage connect genuinely equivalent occurrences across reports?

These questions are evaluated separately. A useful extraction can remain
isolated because no equivalent concern appears elsewhere, and that is not a
pipeline failure.

## 2. Input preparation and report coverage

The source dataset contains stable report identifiers and URLs together with
the report's `concerns`, `circumstances`, and, where available,
`investigation` text.

Before extraction, the pipeline:

- excludes rows with no usable source text;
- removes duplicate URLs;
- creates a stable hashed `report_key`; and
- records exclusions rather than silently dropping them.

When the combined source exceeds the model input budget, text is retained in
the following order:

1. `concerns`;
2. `circumstances`;
3. `investigation`.

The concerns section is prioritized because it contains the coroner's explicit
statement of an ongoing risk. Circumstances and investigation text still
matter: they may contain an explicitly stated reusable system gap, policy gap,
repeated practice, or clear omission. They are not used to infer a systemic
issue from chronology or an adverse outcome alone.

In the completed full-corpus extraction used to develop the relational method:

- 6,248 input rows were checked;
- 6,244 reports contained usable source text;
- all 6,244 were processed successfully;
- 6,223 produced at least one issue;
- 21 produced no issue; and
- 28,870 issue occurrences were extracted.

The fact that 21 reports produced no occurrence is not automatically an error.
Manual review found a mixture of legitimate empty, vague, or redacted reports,
some concrete misses, and a small number of upstream source omissions.

## 3. Structured issue extraction

### 3.1 Extraction instruction

The extraction model is instructed to identify every distinct substantive
safety concern, omission, unsafe condition, or service gap communicated by the
coroner. Concerns are primary, but explicit reusable system failures in the
other source sections are also eligible.

The model is told to:

- state the problem rather than merely restating a recommendation;
- extract an underlying missing provision conservatively when the report gives
  only a proposed remedy;
- avoid names, dates, local codes, organizations, and consequences in the
  reusable issue statement;
- preserve distinctions that change the issue type;
- avoid biography, chronology, unsupported causality, and isolated individual
  actions with no wider safety concern;
- optimize extraction recall while avoiding duplicates; and
- return only data conforming to a strict JSON schema.

The initial response may contain up to 24 issues. If it reaches that cap, a
continuation call can request up to 12 additional issues while supplying the
existing issue statements to prevent duplication. This avoids imposing an
artificially low ceiling on genuinely dense reports.

### 3.2 Evidence grounding

Every occurrence must contain a short `evidence_quote` copied verbatim from the
report. The quotation is not decorative provenance: it is the evidential basis
for normalization, review, and correction.

The pipeline checks whether the proposed quotation occurs in the concerns,
circumstances, or investigation text after conservative text normalization. If
the quote is not found, it attempts to recover an exact supporting sentence
using token overlap, sequence similarity, and contiguous matching. Invalid
quotes trigger a corrective model retry. Evidence validity and source section
are stored with the occurrence.

The full extraction produced 28,814 automatically valid quotations and 56
quote-matching exceptions. Manual inspection found the 56 exceptions to be
substantively supported; most failures arose from ellipses, punctuation,
section concatenation, or excerpt boundaries. This is an important distinction:
they were primarily failures of exact span matching, not hallucinated issues.

### 3.3 The extraction schema

The schema separates issue identity from descriptive facets. It avoids one
overloaded “domain” or “failure mode” field.

| Field | Purpose |
|---|---|
| `canonical_issue` | A short, reusable, neutral formulation of the problem. |
| `evidence_quote` | A verbatim supporting excerpt from the source report. |
| `failure_state` | How an action, provision, object, or duty failed. |
| `process_stage` | The direct operational activity in which the failure occurred. |
| `service_sectors` | Up to two broad institutional sectors involved. |
| `issue_themes` | Up to three cross-cutting safety themes. |
| `service_contexts` | Up to two concrete services or environments. |
| `populations_at_risk` | Up to two groups who could suffer harm. |
| `communication_direction` | The direction of information or contact, where relevant. |
| `responsible_actor_type` | A controlled, broad accountability category. |
| `responsible_actor_text` | The actor as explicitly stated in the report. |
| `concern_status` | Whether the concern is historical, unresolved, prospective, or recommendation-only. |

#### Failure state

The allowed values are:

`omitted`, `delayed`, `incomplete`, `incorrect`, `inadequate`,
`inconsistent`, `unavailable`, `inaccessible`, `excessive`, `unsafe`,
`non_compliant`, `ambiguous`, `unverified`, `uncoordinated`, and
`other_review`.

This field describes **how** something failed, rather than naming the thing
that should have happened. For example, a missing assessment is
`failure_state=omitted` and `process_stage=assessment`.

#### Process stage

The operational vocabulary covers clinical and non-clinical activity:

`prevention_risk_reduction`, `access_scheduling`, `triage`, `assessment`,
`diagnosis`, `planning`, `referral`, `admission`, `treatment_care_delivery`,
`prescribing`, `dispensing_supply`, `medication_administration`,
`procedure_surgery`, `monitoring_observation`, `escalation`, `handover`,
`transfer_transport`, `discharge`, `follow_up`, `rehabilitation`,
`care_coordination`, `emergency_call_handling`, `emergency_response`,
`search_rescue`, `safeguarding_response`, `information_record_management`,
`incident_investigation`, `death_review`, `organisational_learning`,
`workforce_management`, `training_delivery`, `commissioning_funding`,
`procurement_supply`, `design_engineering`, `manufacture_production`,
`product_labelling_warning`, `maintenance`, `inspection_enforcement`,
`licensing_certification`, `policy_development`, `protocol_implementation`,
`service_provision`, `public_information`, `research_evidence`,
`regulation_oversight`, `professional_practice`, and `other_review`.

The breadth is intentional. Most PFD reports involve healthcare, but the
method must also represent failures involving roads, workplaces, products,
custody, education, housing, regulation, water, farming, and other systems.

#### Sector, theme, context, and population

These fields answer different questions:

- **Sector** describes the broad institutional domain, such as `healthcare`,
  `social_care`, `justice_custody`, `policing`, `education`, `transport`,
  `workplace`, `consumer_products`, `utilities_infrastructure`,
  `public_health`, or `government_regulation`.
- **Theme** provides a cross-cutting analytic lens, such as
  `clinical_assessment_care`, `medication`, `communication_handover`,
  `records_information`, `staffing_capacity`, `training_competence`,
  `equipment_infrastructure`, `digital_technology`, `emergency_response`,
  `safeguarding`, `governance_learning`, `environmental_design`,
  `access_service_capacity`, or `risk_assessment_management`.
- **Context** describes the actual service or environment, such as an acute
  hospital, primary care, ambulance service, prison, school, workplace, road,
  railway, domestic home, public space, online service, farm, or maritime
  setting.
- **Population** describes who is exposed to the risk, such as a patient, care
  home resident, social-care user, person in custody, child, student, trainee,
  worker, member of staff, family carer, road user, passenger, consumer,
  recreation participant, or the general public.

The population is not copied automatically from the deceased's apparent role.
For example, `student_in_education` is reserved for an enrolled learner; it is
not a generic label for any young person or trainee. This rule was introduced
after earlier outputs overused “students”.

Each vocabulary also contains a conservative review or missingness route:
`other_review` is used when evidence is present but no option fits;
`not_stated` is used where the source does not support a value; and
`not_applicable` is available for contexts where the concept genuinely does
not apply.

#### Communication, actor, and temporal status

Communication direction distinguishes:

`within_team`, `between_teams_same_organisation`, `between_organisations`,
`professional_to_patient`, `professional_to_family_carer`,
`patient_family_to_professional`, `service_to_public`, `public_to_service`,
`bidirectional`, `not_applicable`, and `not_stated`.

Broad responsible-actor types include:

`individual_practitioner`, `team`, `provider_organisation`, `commissioner`,
`regulator_inspector`, `government_public_authority`,
`manufacturer_supplier`, `employer_operator`, `multi_organisation`,
`not_stated`, and `other_review`.

Concern status distinguishes:

- `historical_failure`: a past event only;
- `current_system_gap`: an unresolved present deficiency;
- `future_risk`: a prospective hazard; and
- `recommendation_only`: only a remedy is stated, from which the underlying
  gap has been conservatively represented.

### 3.4 Deterministic validation and deduplication

Model output is not accepted uncritically. The pipeline:

- enforces the JSON schema;
- checks controlled values;
- limits text and array lengths;
- inserts conservative missing or review values when needed;
- verifies evidence;
- flags recommendation-framed rather than failure-framed statements;
- flags local actor names leaking into canonical statements; and
- removes within-report duplicates only when their normalized statements are
  identical or extremely close and their failure state, process, direction,
  and protected distinctions agree.

Stable issue identifiers are derived from report identity, evidence, and issue
content. Extraction is append-only and checkpointed, allowing interruption and
recovery while retaining an audit trail.

## 4. Relational normalization

### 4.1 Why a second representation is necessary

A readable issue sentence alone is not a sufficiently stable object for
distance-based linkage. Natural language can make opposing relationships sound
similar, and different writing styles can make equivalent obligations sound
different.

The relational normalization stage therefore decomposes each extracted
occurrence into:

| Relational field | Question answered |
|---|---|
| `responsible_actor_role` | Who held the obligation or performed the deficient action? |
| `failed_action` | What positive action should have happened? |
| `issue_object` | What information, person, system, equipment, condition, or duty did the action operate on? |
| `counterparty_role` | Who was the recipient or other party needed to understand the relationship? |
| `canonical_issue` | How can the complete concern be expressed readably without losing agency or direction? |

Examples of `failed_action` include “contact”, “share information with”,
“assess”, “monitor”, “maintain”, “implement”, “refer”, and “respond to”. The
failure state is kept separately: the action is “contact”, not “fail to
contact”.

The `issue_object` is deliberately general across sectors. It may be a risk
assessment, discharge information, staffing levels, bridge barrier design,
product warning, custody observation policy, or another concrete target. The
pipeline uses “object” rather than “safeguard” because not every PFD concern is
medical or naturally described as a clinical safeguard.

The counterparty is used only when supported and relevant. Missing source
information remains “not stated”; the pipeline does not invent a recipient to
complete a grammatical template.

### 4.2 Schema-informed sentence generation

The canonical sentence is not generated independently of the structured
fields. The model must derive actor, action, object, and counterparty first and
then write the sentence from them. Where an actor is supported, its generic
role must appear explicitly, normally as the grammatical subject. If no actor
is supported, a passive formulation is used.

This design makes the sentence itself useful for both people and embeddings.
For example:

- “The mental health service did not contact the patient after missed
  appointments” foregrounds a service omission.
- “The patient declined further contact with the mental health service”
  foregrounds a supported individual decision.

The model remains free to produce natural language rather than being confined
to a brittle template, but the sentence must preserve the structured
relationship. It is instructed specifically not to use “disengagement from
services” as an umbrella for service non-contact, failure to follow up, failure
to respond to help-seeking, and voluntary refusal.

Code also creates a deterministic `linkage_statement` from the relational
fields. This separates the public-facing sentence, which may vary naturally,
from a stable machine-facing representation.

The current full-corpus relational run uses the earlier normalized occurrence
file as its starting input to avoid repeating completed extraction work. The
v2 normalizer rechecks its facets against `evidence_quote`; earlier
normalization is useful context but is not treated as authoritative when it
contradicts the evidence. In a clean future run, extraction can feed the v2
normalizer directly.

### 4.3 Normalization quality controls

Every input `issue_id` must receive exactly one output in the same order.
Responses with missing, additional, or duplicated identifiers are rejected or
retried. Additional checks flag:

- a supported actor being discarded;
- canonical wording that omits the extracted actor;
- generic actions or objects;
- suspiciously short objects;
- directional issues with no supported counterparty; and
- agency-sensitive language.

Generic fallback values are retained for traceability but excluded from
automatic linkage. A directional issue without a supported counterparty can
remain eligible because direction itself is evidence; it is separately routed
to review.

## 5. Semantic representation and embedding arithmetic

### 5.1 Four complementary views

Each eligible occurrence is represented in four ways:

1. **Readable canonical view**  
   The complete natural-language issue sentence.

2. **Full relational view**  
   Actor, action, object, counterparty, failure state, process stage, and
   communication direction written as labeled fields.

3. **Action–object view**  
   The failed action and its object only.

4. **Roles-and-direction view**  
   Responsible actor, counterparty, and communication direction.

Each text is independently embedded using
`Qwen/Qwen3-Embedding-8B`, producing a unit-normalized vector.

Broad themes, sectors, contexts, and populations are intentionally not part of
linkage distance. They remain valuable for filtering, stratification,
interpretation, and group summaries, but they are too broad to establish issue
identity. Two issues can both involve healthcare and communication while
describing opposite obligations; conversely, the same concrete obligation may
occur in different service settings.

### 5.2 Weighted concatenation

The four embeddings are combined using the frozen weights:

| View | Weight |
|---|---:|
| Canonical sentence | 0.10 |
| Full relational representation | 0.55 |
| Action and object | 0.25 |
| Roles, counterparty, and direction | 0.10 |

If the four unit vectors are \(e_c\), \(e_r\), \(e_o\), and \(e_d\), the
combined representation is:

\[
z =
[\sqrt{0.10}e_c;\sqrt{0.55}e_r;\sqrt{0.25}e_o;\sqrt{0.10}e_d]
\]

The concatenated vector is normalized. Because the weights sum to one, cosine
similarity between two combined vectors is effectively the weighted sum of the
four view-specific cosine similarities:

\[
\operatorname{sim}(i,j) =
0.10\,e_{c_i}\cdot e_{c_j} +
0.55\,e_{r_i}\cdot e_{r_j} +
0.25\,e_{o_i}\cdot e_{o_j} +
0.10\,e_{d_i}\cdot e_{d_j}
\]

This is preferable to embedding one long concatenated prompt. The contribution
of each methodological view is explicit, reproducible, and tunable. The
relational content dominates, but natural-language paraphrase information is
not discarded.

## 6. Candidate linkage

### 6.1 Nearest-neighbour retrieval

For each eligible occurrence, the pipeline retrieves its 80 nearest neighbours
by cosine similarity.

A candidate pair must:

- come from two different reports;
- have embedding similarity of at least 0.68; and
- appear in both occurrences' retrieved neighbourhoods.

The 0.68 threshold is a retrieval floor, not the decision threshold. Its
purpose is to avoid spending compatibility calculations on remote pairs while
retaining plausible paraphrases.

After structured scoring, a pair must also fall within both occurrences' top
40 qualifying links. Mutual retrieval reduces one-sided “hubness”, where a
generic occurrence appears moderately similar to many more specific ones.

### 6.2 Guarded compatibility arithmetic

The first relational experiment added many independent bonuses and penalties
to the combined embedding score. That was interpretable, but on the full
corpus bonuses accumulated too readily: 32.2% of accepted edges had a raw
similarity below the nominal 0.85 threshold and 8.6% of accepted scores were
clipped at 1.0. Generic agreement could therefore compensate for a weak
operational match.

The selected guarded method instead requires three pieces of evidence:

- combined similarity of at least **0.84**;
- relation-view similarity of at least **0.72**; and
- action-object-view similarity of at least **0.72**.

Structured agreement remains useful as corroboration, but all positive
adjustments together are capped at **0.03**. Actor, counterparty, direction,
failure-state, and object disagreements retain explicit penalties or vetoes.
Every view score, adjustment, and reason is written to the pair output.

Action families recognize limited operational paraphrases—for example,
“assess”, “evaluate”, “review”, and “screen” belong to an assessment family;
“contact”, “communicate”, “inform”, “notify”, “send”, and “share” belong to a
communication family.

Object comparison removes grammatical stop words and distinguishes generic
heads from discriminating modifiers. “CT scan results” and “smear-test
results” should not match simply because both contain “results”. Terms such as
assessment, risk, observation, appointment, record, information, care, policy,
and training cannot by themselves establish a concrete shared object.
Underspecified objects can link only when their broad concept and operational
relation—actor, action family, and failure family—agree.

The final pair score is:

\[
\operatorname{adjusted}(i,j) =
\operatorname{embedding\_similarity}(i,j) +
\sum_k \operatorname{compatibility}_k(i,j)
\]

clipped below 1.0. At the selected precision setting, accepted edges require
an adjusted score of at least **0.84**, subject to the view floors and
cannot-link rules.

This hybrid design lets semantic embeddings handle paraphrase while keeping
agency, action, and direction inspectable. A reviewer can see not only that two
rows linked, but also which structured agreements or disagreements changed the
score.

## 7. Hard relational constraints

Some contradictions should not be traded against high semantic similarity.
The grouping stage therefore applies cannot-link rules for:

- opposite communication directions, such as professional-to-patient versus
  patient-or-family-to-professional;
- actor/counterparty reversal;
- materially incompatible action families, including performing versus
  recording an activity and conducting versus updating it;
- summoning an ambulance versus an ambulance service responding;
- identical generic object heads with incompatible specific modifiers.

These constraints protect directional meaning. They are applied across
prospective group members, not merely to the pair that happened to initiate a
merge.

The rules are intentionally bounded rather than an exhaustive ontology. An
overly large rule system would be difficult to maintain and could reproduce the
same brittleness as a fully hand-coded taxonomy.

## 8. Conservative group construction

### 8.1 Why ordinary connected components are insufficient

If every accepted pair is treated as an edge and each connected component as a
group, similarity can chain:

- A is close to B;
- B is close to C;
- therefore A, B, and C are grouped,

even when A and C express different obligations. Generic bridge statements can
therefore merge several coherent subtypes into one broad topic.

### 8.2 Prototype-anchored groups

The selected method grows groups around a directly supported prototype.

For a proposed group:

- at least one member with a discriminating object must have a qualifying
  direct edge to every other member;
- no pair of members may trigger a hard relational conflict; and
- a generic object cannot serve as the prototype.

This is less permissive than connected components but less fragmenting than
complete-link clustering, which requires every pair of paraphrases to be close.
Two differently worded peripheral members need not be close to one another if
each has a strong, relation-preserving link to a concrete prototype.

More than one distinct occurrence from a report may belong to a group, because
a report can provide separate evidence of the same organisational issue.
Recurrence and prevalence nevertheless count distinct reports, not rows.

The initial top-40 neighbourhood imposes a practical ceiling on seed-group
size. A guarded consolidation pass therefore compares group medoids and merges
only compatible groups when at least 65% of the smaller group has direct
cross-group edge support. This removes the artificial size ceiling without
falling back to connected-component chaining.

The grouping anchor is a structural device. For output, the pipeline selects
the group medoid—the member with the greatest mean similarity to the other
members—as the readable prototype example.

Groups found in three or more distinct reports are marked `recurring`.
Singletons and two-report pairs are retained as `isolated_or_pair`, rather than
discarded or forced into a larger category. This permits future snapshots to
turn an emerging pair into a recurring group.

For downstream presentation, recurrence can be described separately from
identity:

- one report: isolated;
- two reports: emerging;
- three or four: recurring candidate;
- five to nine: established recurring; and
- ten or more: high frequency.

Frequency does not itself establish coherence or publication readiness.

## 9. What we tried, what failed, and what was learned

### 9.1 Holistic semantic grouping

The earlier index relied mainly on semantic similarity between whole issue
sentences, including a dual blend of original and normalized wording. It found
substantial recurring signal, but manual review showed that semantic similarity
was too willing to merge directionally different concerns.

In the full-corpus v1 audit:

- 33 of 60 randomly sampled groups (55.0%) were coherent as constituted;
- another 16 (26.7%) contained a valid recurring core but needed member removal
  or splitting;
- 11 (18.3%) did not support one concrete recurring issue; and
- 61 of 336 audited members were explicitly incorrect, although this
  understates directional splitting problems.

The principal errors were not random. They included:

- initiating versus responding to a referral or escalation;
- performing an action versus recording or auditing it;
- communication to different recipients or in opposite directions;
- generic shared words such as assessment, records, training, or
  investigation;
- generic members broadening a tight subgroup; and
- large groups accumulating several genuine but distinct subtypes.

The “so what” was that extraction was producing useful material, but the
representation used for linkage did not adequately encode the obligation.
This motivated relational normalization rather than simply another round of
global threshold tuning.

### 9.2 Lower similarity thresholds

Lowering thresholds recovered some true paraphrases but admitted false links at
almost the same rate. The earlier missed-link audit likewise found only a small
number of credible missed merges near and below the threshold, while linked
high-similarity controls were generally sound.

The conclusion was not that missed links are harmless, but that a global
recall increase was a poor trade for group coherence. Conservative missed
links can remain discoverable in search or be reconciled later through reviewed
registry lineage; false merges distort prevalence and the meaning of the group
itself.

### 9.3 Complete-link and constrained-density clustering

Complete-link logic controlled chaining but was too strict: legitimate
paraphrases often fail to resemble every other member even when each clearly
matches a central example.

Constrained-density grouping was then tested as a compromise. Across tested
settings:

- manually different-pair separation ranged from 78.3% to 91.1%; while
- manually same-pair retention remained only 27.0% to 32.4%.

It did not improve the precision–retention trade-off over prototype-anchored
grouping and was not selected.

### 9.4 Generic post-group degree gates

We tested removing weakly connected members after grouping. A rule requiring at
least two internal neighbours identified only 5 of 17 peripheral errors while
also removing 22 of 223 correct members in the holdout review.

This was a poor and opaque trade-off. Correct uncommon paraphrases can be
peripheral in graph terms, while an incorrect generic member can have several
moderate links. Graph degree is not equivalent to conceptual correctness.

### 9.5 A second LLM adjudication stage

A separate LLM group-adjudication process was built and exercised as a
one-off validation route. It can accept, split, exclude, or reject candidate
groups with complete membership provenance. It remains useful as a review aid,
especially for high-frequency groups, but it was not made the core linkage
method.

The reasons were methodological and practical:

- it adds substantial computation;
- it can obscure whether the underlying representation is actually sound;
- repeated adjudication risks becoming a bottomless repair layer; and
- it still requires independent evaluation rather than being treated as an
  oracle.

The relational experiment was therefore preferred as an improvement to the
underlying method. Adjudication remains an optional publication-control tool,
not a substitute for good extraction and linkage.

### 9.6 Assigning isolated occurrences to existing prototypes

A conservative second-pass prototype assignment was also tested in the older
pipeline. Fourteen of 16 retained promotions were correct (87.5%), with one
correct assignment duplicating an existing recurring type. That precision is
useful for generating a review queue but insufficient for automatic production
promotion under the current success criteria.

### 9.7 Overly broad and duplicated schema fields

Earlier schemas produced too many generic “other” values and included text
fields with unclear or duplicated purposes. The vocabulary was expanded to
cover non-medical processes and contexts. Population instructions were
tightened, particularly around “student”. Failure state and process were
separated, and responsibility and direction were made explicit.

`issue_statement_original`, `source_span`, and
`essential_qualifier_text` were retired. Their useful functions are now covered
by:

- one evidence-bearing `evidence_quote`;
- a sufficiently specific `canonical_issue`; and
- explicit relational fields.

Report provenance is represented once through report identity and URL rather
than repeated across several semantically redundant columns.

## 10. Validation of the relational method

### 10.1 Development sample

The development sample contained 475 occurrences drawn from manually reviewed
groups and candidate pairs.

At the frozen precision configuration:

- 26 of 32 manually equivalent direct pairs linked (81.3%);
- 37 of 38 manually different direct pairs remained separated (97.4%);
- direct-pair agreement was 90.0%; and
- the method produced 31 recurring groups containing 147 occurrences across
  143 reports.

The relational method did not reproduce many old accepted groups wholesale.
That was informative rather than automatically negative: some old groups were
broad semantic umbrellas, and many correct peripheral members lacked a direct
relational edge at 0.85. Lowering the threshold did not recover them cleanly.

### 10.2 Untouched stratified holdout

Twenty source groups not used in the development audit were selected
deterministically before relational normalization. They contained 394
occurrences. The frozen method was then run without threshold tuning.

It produced 36 recurring groups containing 240 occurrences. Evidence-backed
review of every group and member found:

- 25 of 36 groups fully coherent (69.4%);
- 11 groups with a coherent core plus peripheral members;
- no wholly incoherent groups;
- 223 of 240 members correctly grouped (92.9% member precision); and
- 17 peripheral members (7.1%).

Residual errors included:

- generic escalation concerns entering clinical-deterioration groups;
- appointments or reviews entering delayed-referral groups;
- defibrillator retrieval entering a failure-to-use group;
- verbal explanation entering a written-document group; and
- one reversed medication-information action.

The holdout review was conducted during pipeline development. It is auditable
internal validation, not an independent domain-expert evaluation, and the
stratified selection must not be used to estimate population prevalence.
Member precision also remained below a strict 95% target. These qualifications
are why outputs are treated as candidate recurring issues rather than
automatically authoritative classifications.

### 10.3 Guarded full-corpus linkage

The completed corpus contained 28,870 normalized occurrences, of which 28,765
passed the linkage quality gate. Applying the original arithmetic exposed two
scale effects that were not obvious in the smaller experiments: generic
objects became semantic hubs, while the mutual top-40 retrieval rule imposed
an artificial group-size ceiling. The guarded method and supported
consolidation pass were introduced in response to those observed failures.

On the frozen holdout labels, the final guardrails increased separation of
known incorrect pairs from 68.0% to 74.4%, while reducing retention of known
correct pairs from 77.7% to 73.2%. This is an explicit precision-first
trade-off, not an unqualified improvement in recall.

On the complete corpus, the final method produced:

- 40,383 accepted mutual edges;
- 1,426 recurring candidate groups;
- 8,238 recurring occurrences;
- 3,806 represented reports; and
- 239 supported post-seed consolidations.

Compared with the original full-corpus relational setting, mean within-group
minimum similarity rose from 0.773 to 0.798 and mean within-group median
similarity rose from 0.839 to 0.859. Qualitative inspection also separated the
previous large umbrellas for generic risk assessments, mixed clinical
observations, and mental-capacity versus mental-health assessments. A direct
directional test confirmed that delayed summoning of an ambulance no longer
joined delayed ambulance-service response.

These are candidate groups, not a claim that all 1,426 are independently
validated topics. A fresh audit packet contains 60 untouched random groups and
separate boundary, high-frequency, and low-cohesion diagnostic samples.
Further tuning is frozen until those samples are reviewed.

### 10.4 Group-level subtype refinement experiment

The untouched 60-group audit of the guarded full output found 36 fully
coherent groups, 19 groups with a coherent core, and five rejected groups.
Object hierarchy broadening was recurrent: generic medication could absorb
naloxone, pedestrian crossings could absorb wider infrastructure, and generic
assessment or recording frames could combine different operational duties.

A bounded model-free refinement experiment was therefore implemented. It:

- measures within-group incompatibility across actor, action, object,
  direction, and failure views;
- identifies multiple directly supported specific-object cores;
- prevents underspecified objects from being silently attached to a specific
  core;
- quarantines ambiguous or compound residual members;
- distinguishes `refined_recurring` from `review_required`; and
- emits a structured label based on fields invariant across members.

The development audit selected an incompatibility trigger of 0.25. On those
already reviewed groups, automatically retained members were 92.1% compatible
with the adjudicated core and 90.4% of known-different pairs were separated.
Across the full output, the setting produced 970 automatically refined groups
and 608 review-required groups.

This apparent development improvement did **not** generalise. A second,
untouched random sample excluded all occurrences used in the previous holdout
and random audits. Among 60 refined groups containing 252 occurrences:

- 36 groups were fully coherent (60.0%);
- 15 contained a coherent core with peripheral members (25.0%);
- nine were rejected (15.0%); and
- 199 of 252 members fitted the adjudicated core (79.0% descriptively).

Residual failures again involved generic treatment, discharge, records,
review, product-design, and referral frames. The refinement layer is therefore
not an automatic approval mechanism. It remains useful for suggesting splits,
flagging ambiguous members, constructing review strata, and prioritising human
work. Publication and stable registry promotion require group validation.

### 10.5 Recall calibration and the parent-family layer

Precise operational groups are intentionally conservative. They should not be
made broader merely to recover every report that concerns a recognisable
policy family. The pipeline therefore represents hierarchy explicitly:

- an occurrence retains its precise relational group, review-group, or
  isolated status; and
- it may additionally receive zero, one, or several broad parent-family
  assignments.

The first recall calibration covers five families:

1. continuity and coordination of care;
2. medical and care records and information;
3. staffing capacity and workforce availability;
4. discharge and care transitions; and
5. referral and required follow-up pathways.

For each family, a high-recall candidate pool is the union of:

- strict family-specific lexical retrieval;
- relevant structured process-stage and issue-theme retrieval; and
- the 1,000 occurrences nearest to the embedding centroid of strict lexical
  seeds.

Candidate selection is not treated as a family label. A stratified,
group-balanced sample is drawn separately from refined recurring groups,
review-required groups, isolated or unassigned occurrences, and semantic-only
retrieval. A local LLM adjudicates whether each sampled occurrence entails the
family definition, subject to explicit inclusions, exclusions, and
actor-direction rules.

The sample measures a different failure mode from group coherence:

- the share of supported reports already captured in operational child groups;
- false-positive family candidates within child groups;
- valid family cases stranded in review groups or isolated occurrences; and
- the number of distinct operational child groups represented by a parent
  family.

Automatic parent membership is calibrated separately for strict lexical,
schema-rule, and semantic-only channels. Threshold selection targets minimum
precision on a deterministic calibration partition and is evaluated on a held
out partition. Adjudicated decisions override calibrated predictions.
Candidates below a supported threshold remain review candidates rather than
being silently accepted.

This layer is non-exclusive by design. For example, failure to transfer a
discharge summary may belong to both `records_information` and
`discharge_transitions`, while retaining a precise child group about discharge
summary transfer. Parent families support recall and navigation; child groups
retain the operational meaning required for prevalence and review.

## 11. Interpretation: what the output does and does not mean

A recurring group supports the claim:

> At least three different PFD reports contain evidence-backed issue
> occurrences that the present method judges to express the same concrete
> failed obligation.

It does **not** by itself support the claim that:

- the issue caused each death;
- every report in the domain was equally likely to mention the issue;
- the reports are a representative sample of all deaths or all service
  failures;
- absence from a group means absence of concern;
- group size is an unbiased incidence rate; or
- the group is ready for publication without review.

PFD reports are produced through a legal and administrative process, not a
population-surveillance sampling design. Counts describe recurrence within the
available corpus. They are valuable for identifying patterns, comparing the
content of reports, and supporting investigation, but they require careful
denominators and contextual interpretation.

The method intentionally favors precision over exhaustive linkage. This means
one real-world issue may appear in more than one candidate group when wording,
actor specificity, or operational context differs too greatly. That is usually
less damaging than merging distinct obligations and presenting the result as
one prevalent issue.

## 12. Auditability, review, and future snapshots

The pipeline retains:

- source report identity;
- the exact evidence quotation;
- original and normalized issue wording;
- structured extraction and relational fields;
- normalization warnings and status;
- candidate-pair similarities;
- every compatibility bonus and penalty;
- accepted edges;
- group assignments and prototypes;
- model, prompt, schema, parameter, and embedding fingerprints; and
- append-only extraction and normalization checkpoints.

This permits an analyst to move from a group summary back to each occurrence
and then to the supporting report text.

For operational use, the recommended states are:

- isolated or emerging occurrence;
- recurring candidate;
- reviewed/approved recurring issue;
- needs review following membership change; and
- not published or rejected.

Stable issue-type registration should be based on membership overlap across
corpus snapshots, with split and merge lineage recorded explicitly. A validated
group should not silently retain its approval when its membership changes.

Before public or substantive use, the full-corpus relational output should
receive:

1. a fresh random group audit to estimate overall coherence;
2. targeted review of the largest groups, which are especially vulnerable to
   subtype accumulation;
3. review of groups with generic objects, mixed directions, or normalization
   warnings;
4. an independent domain-expert sample review; and
5. a missed-link audit that samples separated near-neighbour pairs as well as
   accepted links.

Further tuning should be driven by recurring, interpretable error patterns in
those audits. The method should not be made more complex merely to increase the
number of groups or attach more reports.

### Deterministic error localization

False negatives and false positives require different diagnostics. Separate
groups with substantial accepted cross-group edge coverage are possible false
splits; weak, conflicting, compound, or graph-bridging members inside a group
are possible false joins. Neither condition alone proves an error because the
intended issue granularity remains an analytical decision.

`build_relational_diagnostics.py` therefore produces separate duplicate-group,
contaminated-group, and contaminated-member queues using only persisted
pipeline artifacts. For possible duplicates it reports candidate and accepted
edge counts, coverage on each side, pair scores, all-pairs relational conflicts,
and the observable consolidation blocker. For possible contamination it reports
prototype fit, internal edge support, articulation points, compound relational
objects, relational conflicts, and stronger external accepted edges.

`trace_relational_case.py` provides the corresponding issue- or group-level
provenance. These tools do not alter assignments and make no LLM calls. Review
decisions should become regression fixtures before a global rule is changed.
The appropriate unit of change is the earliest component that repeatedly
causes the confirmed error: extraction atomicity, normalization, pair
compatibility, or group consolidation.

Directed retrieval ranks were not stored in the historical full-corpus run.
The trace can therefore establish that an absent pair did not enter the
candidate table, but cannot distinguish its precise retrieval rank from a
retrieval-threshold rejection. Future production runs should persist that
small amount of provenance.

The v3 consolidation revision applies any conflict-tolerant merging only after
the strict grouping has completed. This additive sequencing is important: an
earlier experiment changed the original merge order and broke a confirmed broad
group even though its aggregate results looked plausible. The additive pass can
only combine whole recurring groups. It requires accepted-edge coverage,
conflicts below a fixed cross-pair proportion, and distinctive object-token
support on both sides after removing subject-context terms. Stable issue IDs in
`relational_regression_cases_v1.json` are checked after every experiment so
renumbered group IDs cannot disguise membership changes.

A review of the first 50 high-evidence separated group pairs found that
embedding and coverage thresholds cannot reliably resolve the remaining
boundary. Related-but-distinct and generic-object-contaminated pairs had
similar or higher action-object similarity than true duplicates. The next
representation revision therefore changes normalization rather than linkage
thresholds. Version 3 requires the object to retain the assessed hazard,
monitored condition, referral destination, recorded information type, or
follow-up trigger whenever supported. This is still produced in the existing
normalization call. Existing bare process objects are flagged and can be
backfilled selectively.

## 13. The methodological “so what”

The main lesson from development is that identifying recurring PFD issues is
not principally a generic clustering problem. The difficult part is preserving
the structure of an obligation while moving from narrative legal documents to
comparable representations.

LLMs are useful here because they can interpret varied narrative expression,
but the method does not ask an LLM to make an opaque end-to-end decision. It
constrains and checks extraction, separates semantic roles, grounds claims in
evidence, performs explicit embedding arithmetic, records compatibility
decisions, and keeps human publication control.

The resulting system occupies a useful middle ground:

- more flexible than a fixed keyword taxonomy;
- more direction-sensitive than ordinary semantic clustering;
- more scalable than manually coding every report;
- more auditable than asking an LLM to invent groups directly; and
- appropriately conservative about what recurrence in a heterogeneous legal
  corpus means.

The practical objective is therefore not to prove that every concern can be
perfectly normalized. It is to create a defensible, inspectable candidate index
in which recurring groups are sufficiently precise to support serious human
analysis without disguising uncertainty or forcing heterogeneous reports into
artificial categories.

## 14. Locked topics and precise issues

The public issue tracker uses two independent classification systems:

1. **Precise issues** are extracted from coroner concerns and retain the
   concrete action, object and failure described in the report. Existing issues
   are not discarded when the topic catalogue changes.
2. **Topics** provide stable report-level navigation across broader prevention
   concerns, service settings, circumstances or conditions, and populations.

Topics are not parents of precise issues. A suicide report can therefore carry
the `Suicide and self-harm` topic while its extracted issues concern records,
risk assessment, handover, medication or continuity of care. Conversely, a
records issue is not treated as a semantic child of suicide merely because it
appears in that report.

The version 1.0.0 catalogue is locked in
`scripts/issue_tracker_mvp/locked_topics_v1.json`. It contains 40 manually
approved topics and uses multi-label assignment. Automated classification may
attach any number of existing topics when supported by report evidence, but it
must not invent or alter the catalogue. Additions, removals, merges, splits and
definition changes require a new reviewed taxonomy version.

The four internal facets are `concern`, `setting`,
`circumstance_condition`, and `population`. They support validation and analysis
without creating a public hierarchy. The catalogue is loaded through
`topic_taxonomy.py`, which rejects duplicate or malformed topics, count drift,
hierarchical fields, or a taxonomy not explicitly marked as locked.

The earlier automatic parent-family approach is superseded. Its discovery,
overlap, recall and direct-assignment code is retained only in
`scripts/issue_tracker_mvp/research/automatic_parent_families/` so that the
methodological experiments remain reproducible. Those family IDs and report
assignments are not active product data. Production must not infer a topic from
an issue's membership in an archived family; topic evidence is evaluated at the
report level against the locked catalogue.

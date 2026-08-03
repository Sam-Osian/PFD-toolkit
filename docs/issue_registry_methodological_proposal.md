# Methodological proposal: a registry of recurring coroner concerns

**Status:** Proposed methodology for the PFD Toolkit issue tracker  
**Purpose:** Track specific coroner concerns across Prevention of Future Deaths
(PFD) reports even when those concerns are phrased differently.

## 1. Objective

The issue tracker should allow a user to select a specific recurring concern
and find the PFD reports in which a coroner raised that concern.

The unit being tracked is therefore not a broad subject and not merely a
similar passage of text. It is a recurring **corrective concern**: a materially
similar failure, hazard, or unmet obligation that would call for substantially
the same corrective response.

For example, concerns that mention the Assessment, Care in Custody and
Teamwork (ACCT) process should not automatically become one issue. The
following are related, but distinct, recurring concerns:

- ACCT procedures were not followed;
- staff had not received adequate ACCT training;
- the ACCT framework or process was itself inadequate;
- ACCT documentation was incomplete or inaccurate; and
- an ACCT plan was not reviewed or updated when required.

The method must preserve distinctions like these while recognising genuine
paraphrases of the same concern.

## 2. Topics and issues are separate classifications

The tracker should maintain two independent, many-to-many classifications.

### Topics

Topics describe the wider setting, population, service, condition, or context
of a report. Examples include suicide and self-harm, maternity and perinatal
care, prisons and criminal-justice supervision, care homes, sepsis, ambulance
services, and children and young people.

Topics are assigned at report level from a fixed, pre-defined vocabulary. They
are not inferred from every extracted concern, and they are not parents of
issues. A suicide-related report may contain a medical-records concern whose
wording does not mention suicide; the report can still carry the suicide and
self-harm topic.

### Recurring issues

Issues describe the specific concern expressed by the coroner. They are
derived from evidence-backed concern occurrences and recur across reports.
Examples include failure to share discharge information, inadequate
observation, or failure to follow an ACCT procedure.

An issue may appear under several topics, and a report may contain several
issues. The website can therefore let users browse either by topic or by
specific issue without forcing issues into a topic hierarchy.

## 3. Why the current clustering approach is insufficient

The current relational pipeline extracts concerns and represents their actor,
action, object, counterparty, direction, and failure state. Its conservative
linkage rules are useful for avoiding obviously incorrect merges. Spot checks,
however, indicate that coherent groups can capture only a small fraction of
the reports containing a concern.

This is a structural limitation of treating each run as a clustering problem:

- a high similarity threshold protects precision but separates paraphrases;
- a lower threshold improves recall but joins neighbouring concerns;
- transitive clustering can allow a weak intermediate occurrence to join two
  otherwise distinct issues;
- a new corpus run can change group boundaries and identifiers; and
- the growing number of groups makes global pairwise adjudication difficult to
  explain and maintain.

Further refinement of similarity weights may still improve retrieval, but it
should not be the final authority on issue identity.

## 4. Proposed model: a stable issue registry

Recurring concerns should be represented by a versioned registry of stable
issue types. Each registry record should contain at least:

- a stable issue identifier;
- a concise public label;
- a fuller canonical description;
- the corrective obligation that unifies its occurrences;
- its structured obligation signature;
- representative evidence-backed examples;
- report and occurrence counts;
- registry status, such as candidate, active, inactive, split, or merged;
- creation and update metadata; and
- lineage linking predecessor and successor records.

Once an issue type exists, new concern occurrences should be assigned to it or
left unmatched. The whole corpus should not be reclustered every time new
reports are scraped. Stable identifiers are important for website URLs,
subscriptions, time series, citations, and reproducibility.

The current recurring groups can seed the initial registry, but should be
treated as candidate issue types rather than unquestioned ground truth.

### Concern occurrences are retained independently of recurrence

Every evidence-backed concern extracted from a report should be stored as a
durable concern occurrence, whether or not it matches a recurring issue. A
concern must not disappear from the report-level dataset because it is unique,
rare, too new to recur, or cannot yet be assigned confidently.

Each occurrence should preserve three distinct representations:

1. **Source representation:** the original wording and exact evidence span from
   the report, with source location where available;
2. **Normalised representation:** a concise, faithful restatement plus the
   structured obligation signature used for comparison; and
3. **Registry representation:** an optional link to a stable recurring issue
   and that issue's canonical label and description.

These are not successive replacements for the same text. The source wording
remains authoritative, normalisation makes differently phrased occurrences
comparable, and the registry link supports aggregation. Normalisation must not
erase report-specific detail, and linking must not overwrite either the source
or normalised representation.

An occurrence may have no registry link. Where assignment is ambiguous, the
candidate identifiers and uncertainty can be retained without presenting any
of them as a confirmed recurring issue.

## 5. Global equivalence rule

The pipeline should use one general decision rule rather than a manually
maintained set of inclusion and exclusion criteria for every issue:

> Two concern occurrences represent the same recurring issue when one accurate
> issue description and one corrective obligation can cover both without
> discarding a materially different hazard, action, object, responsibility, or
> remedy.

This test focuses on the intervention implied by the concern. Shared vocabulary
or a shared subject is not enough. Conversely, surface differences in wording,
institution names, job titles, or case details do not require separate issue
types when the underlying obligation and remedy are the same.

## 6. Structured obligation signature

Each extracted concern should be normalised into a compact signature used for
retrieval, comparison, and explanation:

| Field | Question answered |
| --- | --- |
| Failure or hazard | What was inadequate, absent, delayed, unsafe, or at risk? |
| Action family | What action was required: assess, monitor, record, communicate, refer, respond, provide, review, supervise, train, design, maintain, or enforce? |
| Object or process | What was acted on or affected? |
| Responsible function | Which function held the relevant obligation? |
| Counterparty or beneficiary | To whom, for whom, or with whom was the action required? |
| Direction | From which party or service to which other party did information or action need to flow? |
| Corrective obligation | What would have to change to answer the concern? |
| Material qualifiers | Which setting, timing, population, equipment, or named process is essential to the issue's meaning? |

The original evidence span and report context must remain attached. Structured
fields are an aid to comparison, not a replacement for the source text.

### Action families should remain distinct when the remedy differs

For a named process such as ACCT, `follow`, `train`, `design`, `record`, and
`review` point to different corrective actions. They should not be collapsed
because they share the same object.

### Responsible actors should be conditional identity features

Different wording for actors does not always imply a different issue. A
front-line practitioner and their employing organisation can express the same
operational failure at different levels of attribution. They should remain
together when one obligation and remedy accurately covers both.

Actor differences should split issues when responsibility changes the nature
of the required remedy—for example, an external contractor failing to provide
information versus a receiving clinical team failing to act on information it
already had.

## 7. Retrieval followed by bounded adjudication

Assignment should be a two-stage process.

### Stage A: retrieve plausible issue types

Use semantic embeddings of the concern, corrective obligation, and structured
signature to retrieve a small set of registry candidates. Deterministic checks
can then rerank or filter candidates using action family, object, direction,
failure state, and material actor conflicts.

A practical starting point is to retrieve approximately ten candidates and
pass the best three to five to adjudication. These numbers are parameters to be
tested, not methodological commitments.

Retrieval is deliberately recall-oriented. Its purpose is to avoid omitting
the correct registry issue from the shortlist; it does not decide membership.

### Stage B: adjudicate the shortlist

A bounded LLM adjudicator receives:

- the new concern and its evidence;
- its structured obligation signature;
- the small candidate shortlist;
- canonical descriptions and representative examples; and
- the global equivalence rule.

It returns one of:

- `existing_issue`, with the selected stable identifier;
- `no_matching_issue`;
- `ambiguous_between_candidates`; or
- `insufficient_evidence`.

The response should also include a short reason and explicit agreement or
conflict on the material identity fields. Confidence alone must not override a
materially different action or corrective obligation.

This design uses the LLM for a small semantic decision rather than exposing it
to the complete and continually growing registry.

## 8. Scaling and production operation

The adjudicator's context remains approximately constant as the registry
grows, because only the retrieved candidates are included. The full registry
is searched by embeddings and indexed metadata, not placed in the prompt.

The main scaling risk is therefore **retrieval recall**, not prompt length. It
should be measured by checking whether the correct issue appears in the top
`k` candidates. Candidate descriptions can be embedded once and recomputed
only when a registry record changes.

For each newly scraped report, the production workflow should be:

1. extract evidence-backed concern occurrences;
2. normalise each occurrence into an obligation signature;
3. retrieve candidate registry issues;
4. adjudicate assignment against the shortlist;
5. store the assignment, evidence, model and prompt versions, and rationale;
6. add unmatched occurrences to a novel-concern pool; and
7. update public counts without changing stable issue identities.

The pipeline should be deterministic around the LLM wherever possible, cache
model outputs, and permit re-adjudication when prompts or registry definitions
change.

## 9. Discovering new recurring issues

`No_matching_issue` should not immediately create a new public issue. Unmatched
occurrences should enter a separate novel-concern pool. Within that pool, the
pipeline can use the same retrieval and bounded-equivalence method to identify
recurrence.

A candidate can be promoted to the registry when it:

- occurs in the chosen minimum number of distinct reports;
- has a coherent canonical description and corrective obligation;
- has been compared against nearby existing registry issues; and
- does not depend on a single report's incidental wording.

Promotion may be automatic above conservative gates, or queued for periodic
spot checking. It should preserve the contributing occurrence identifiers and
record why the candidate was considered distinct.

## 10. Response actions and optional concern mapping

Stated actions extracted from responses should form a third classification
layer alongside topics and concerns. An action is a first-class record: it does
not need to be mapped to a concern in order to be retained and shown.

The basic relationship is:

```text
Report
  |-- all concern occurrences --> optionally linked to recurring issues
  `-- response actions ------> optionally linked to concern occurrences
```

This supports two complementary views.

### Report-level view: zoom in

For a specific report, a user should be able to see:

- every concern stated by the coroner, including concerns that are not
  recurring or are not yet assigned to the registry;
- the original wording and faithful normalised form of each concern;
- the linked recurring issue, where one has been assigned;
- each action stated in the associated response or responses;
- which actions can be linked to which concerns; and
- which concerns or actions remain unlinked.

An unlinked action is not an error. It may be a broad organisational response,
an action addressing several concerns, an action prompted by the death but not
by a particular stated concern, or too vague to map reliably.

### Recurring-issue view: zoom out

For a recurring issue, a user should be able to see:

- all report-specific concern occurrences assigned to that issue;
- the reports in which those occurrences appeared; and
- all response actions explicitly linked to those concern occurrences.

Only actions with an evidence-backed concern link should be presented as
actions taken in response to that recurring issue. Other actions from the same
reports may be displayed separately as unlinked report actions, but should not
be attributed to the issue merely because they occurred in the same case.

### Action extraction

Each action statement should retain:

- a stable action identifier and its report and response identifiers;
- the exact supporting response text;
- the responsible organisation or actor;
- the action and its object or affected process;
- any beneficiary, recipient, or counterparty;
- timing or deadline information; and
- status, distinguishing completed, underway, planned, proposed, considered,
  and unclear actions.

Extraction should not discard an action because no concern link is available.

### Optional action-to-concern linkage

Action-to-concern mapping should be many-to-many and nullable:

- one action may address several concerns;
- one concern may be addressed by several actions or respondents;
- an action may have no sufficiently supported concern mapping; and
- a concern may have no stated responsive action.

Candidate concerns should normally be retrieved only from the associated
report, using the action, object, actor, counterparty, and corrective obligation.
A bounded adjudicator can then classify a proposed relationship as:

- `directly_addresses`;
- `partially_addresses`;
- `indirectly_supports`;
- `unclear`; or
- `does_not_address`.

The stored link should contain the supporting evidence, rationale, confidence,
and pipeline version. Uncertain relationships should remain uncertain rather
than being forced to provide complete coverage.

The response must not be used retrospectively to decide what concern the
coroner raised. Concern extraction remains grounded in the PFD report, and
action extraction remains grounded in the response. The optional link connects
the two evidence-backed records after each has been extracted independently.

Keeping the complete occurrence layer also permits analyses and interface
features beyond recurrence, including report summaries, searches over original
concern language, organisation-specific analysis, examination of rare or novel
concerns, and later reprocessing with improved normalisation or assignment
methods.

## 11. Expected value, risks, and limitations

This approach is expected to be fruitful because it preserves source-level
records and treats normalisation, action mapping, and recurring-issue
assignment as revisable links. An incorrect or absent link does not remove the
underlying concern or action. The tracker can therefore provide useful
report-level analysis before every recurring-issue assignment is perfect, and
later pipeline improvements can be applied without reconstituting lost source
detail.

The principal foreseeable risks are:

- **Missed extraction.** A concern or action that was never extracted cannot be
  linked downstream. Report-to-extraction spot checks remain necessary,
  especially where one passage contains several concerns.
- **Inconsistent granularity.** One extraction may combine several actionable
  failures while another separates them. The extractor should identify
  separately actionable concerns while allowing them to retain a shared source
  passage.
- **Retrieval failure.** The correct recurring issue may be absent from the
  candidate shortlist. Top-`k` retrieval recall must therefore be measured
  independently from adjudication accuracy.
- **Confusion between neighbouring issues.** A shared object can obscure
  different actions and remedies, as with ACCT training, implementation,
  documentation, review, and process design. Material action and corrective-
  obligation conflicts must remain explicit in adjudication.
- **Registry duplication.** The novel-concern pool may generate a candidate
  that duplicates an existing issue. Promotion must include comparison against
  the closest active registry records.
- **Definition drift.** Successive borderline assignments may gradually broaden
  an issue beyond its original meaning. Stable canonical definitions,
  representative examples, and periodic inspection of boundary members are
  required.
- **Vague response language.** Statements such as "learning has been shared"
  may not identify what changed or which concern was addressed. They should be
  retained as stated actions with an unclear or absent concern link.
- **No independent verification of actions.** A response records what an
  organisation says it has completed, planned, or considered. The Toolkit
  should consistently describe these as **stated actions** or **response
  actions**, not as independently verified implementation or effectiveness.
- **Version-dependent results.** Changes to extraction, normalisation,
  retrieval, adjudication, or registry definitions may alter assignments and
  counts. Model, prompt, component, and registry versions must be recorded.

These are manageable pipeline and governance risks rather than reasons to
abandon the approach. Preserving the independent occurrence and action layers
makes most errors detectable and recoverable.

## 12. Quality assurance without a large validation sample

The absence of a fully human-labelled validation set makes traceable spot
checking essential. Every suspected error should be attributable to a pipeline
stage:

| Observed error | Likely component to inspect |
| --- | --- |
| Concern absent entirely | extraction or source-text coverage |
| Correct issue absent from shortlist | normalisation, embedding, or retrieval |
| Correct candidate present but rejected | adjudication rule or prompt |
| Wrong candidate selected despite material conflict | adjudication or deterministic conflict rules |
| Two registry types express one obligation | registry duplication or promotion logic |
| One registry type contains different remedies | issue definition or assignment contamination |
| Correct issue assigned but topic missing | separate report-level topic classifier |

Routine diagnostics should include:

- top-`k` retrieval recall on reviewed examples;
- false-positive review of issue members, especially low-similarity members;
- false-negative searches among unmatched and neighbouring occurrences;
- duplicate-registry candidate detection;
- cross-report evidence inspection for the largest issues;
- assignment and issue-size changes between pipeline versions; and
- explicit ACCT-style contrast tests where the object is shared but the action
  and remedy differ.

Spot-check decisions should be saved as small regression fixtures. This builds
an accumulating test suite without requiring an impractically large one-off
annotation exercise.

Evaluation should keep three questions separate:

1. **Extraction:** Were all material concerns and stated actions extracted from
   the source documents?
2. **Retrieval:** Where a recurring issue was appropriate, did the correct
   registry record appear in the candidate shortlist?
3. **Adjudication:** Given a shortlist containing the correct candidate, did the
   assignment decision apply the equivalence rule correctly?

Each failure points to a different component and remedy. This separation helps
prevent a local extraction, retrieval, or adjudication defect from prompting an
unnecessary redesign of the complete methodology.

## 13. Recommended implementation sequence

1. Freeze a copy of the current recurring groups as registry candidates.
2. Define the registry schema and preserve group-to-registry lineage.
3. Produce canonical descriptions and corrective obligations for candidates.
4. Implement registry retrieval and record top-`k` candidate diagnostics.
5. Implement bounded adjudication with explicit material-field comparisons.
6. Pilot on a deliberately mixed set containing paraphrases, neighbouring
   concerns, different actors, and ACCT-style action distinctions.
7. Inspect false positives, false negatives, and duplicate registry types, and
   trace each failure to extraction, normalisation, retrieval, adjudication, or
   registry maintenance.
8. Adopt incremental assignment for new reports only after the pilot meets
   agreed precision and retrieval-recall gates.
9. Extract response actions independently and pilot optional, evidence-backed
   action-to-concern links within each report.

The pilot should compare this proposal with the current grouping assignments.
It should not require replacing working extraction, evidence validation, or
normalisation components unless the diagnostics show that they are the source
of a particular error.

## 14. Methodological boundaries

This proposal does not require:

- a parent-child hierarchy of topics and issues;
- per-issue manuals of inclusion and exclusion criteria;
- placing the entire issue registry in an LLM prompt;
- rerunning global clustering whenever reports are added;
- discarding concerns that do not meet a recurrence threshold;
- replacing original concern wording with a normalised or canonical label;
- treating every difference in actor wording as a separate issue; or
- treating semantic similarity as proof that two occurrences are equivalent;
- requiring every response action to map to a concern; or
- attributing every action in a report to every recurring issue in that report.

It does require a stable registry, a global and auditable identity rule,
recall-oriented candidate retrieval, bounded semantic adjudication, and
pipeline-stage diagnostics. This keeps the method aligned with the public goal:
tracking the same specific coroner concern across differently worded reports
without collapsing concerns that require materially different corrective
responses.

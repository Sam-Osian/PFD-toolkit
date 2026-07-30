# Relational linkage experiment findings

## Development sample

The development sample contained 475 occurrences drawn from the existing
manually reviewed random groups and pair audit.

At the frozen precision setting (`edge=0.85`, `group=0.85`):

- 26 of 32 manually equivalent direct pairs linked (81.3%);
- 37 of 38 manually different direct pairs remained separate (97.4%);
- 31 recurring groups contained 147 occurrences across 143 reports.

Reproducing the old groups was much weaker because many manually accepted old
groups were not connected in the relational pair graph. At 0.85, only 10 of 42
accepted or partially accepted cores were connected. Lower thresholds admitted
negative links at almost the same rate as positive ones.

Constrained-density clustering did not improve this trade-off. Depending on the
density threshold, expected-different separation ranged from 78.3% to 91.1%,
while expected-same retention remained between 27.0% and 32.4%. The
prototype-precision baseline remained preferable.

## Untouched stratified holdout

Twenty stratified source groups not present in the random development audit
were selected deterministically before normalization. They contained 394
occurrences.

Normalization completed for 394/394 occurrences with no failures or fallbacks.
The frozen precision pipeline produced:

- 36 recurring groups;
- 240 recurring occurrences;
- 220 represented reports.

A new evidence-backed review of every group and member found:

- 25/36 groups fully coherent (69.4%);
- 11/36 groups containing a coherent core with peripheral members;
- 0 wholly incoherent groups;
- 223/240 members correctly grouped (92.9%);
- 17/240 peripheral members (7.1%).

The review was performed during pipeline development and is an auditable
internal validation, not an independent human evaluation.

The peripheral errors mainly involved:

- generic escalation members entering a clinical-deterioration group;
- appointments or reviews entering a delayed-referral group;
- access or retrieval entering a failure-to-use group;
- verbal explanation entering a written-document group;
- reversed medication-information action.

A simple degree or edge-coverage filter was not acceptable. Removing members
with fewer than two internal neighbours caught only 5/17 errors while also
removing 22/223 correct members. This avenue should not be added to production.

## Decision

Keep the v2 relational representation, modifier-sensitive objects, guarded
view floors, bounded positive adjustments, hard cannot-link constraints,
prototype grouping, and evidence-supported group consolidation. Do not use
constrained-density clustering or a generic post-group degree filter.

The final full-corpus experiment used edge/group threshold 0.84, relation and
action-object floors of 0.72, a 0.03 positive-adjustment cap, and 0.65
consolidation coverage. It produced 1,426 recurring candidate groups containing
8,238 occurrences from 3,806 reports. Mean minimum and median within-group
similarities were 0.798 and 0.859 respectively.

The guarded holdout comparison improved known-negative pair separation from
68.0% to 74.4% and reduced known-positive pair retention from 77.7% to 73.2%.
This is the selected precision-first trade-off. Full-output inspection showed
clearer separation of generic risk assessments, observation performance versus
recording, mental-capacity versus mental-health assessments, and ambulance
summoning versus ambulance-service response.

Freeze tuning until the fresh random and diagnostic full-corpus audit packets
are reviewed. Independent human review should precede publication or promotion
of groups to stable approved issue types.

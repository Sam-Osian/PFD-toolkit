# Relational linkage experiment

This experiment tests a replacement for holistic sentence similarity without
altering the completed production index.

The normalization step extracts five distinct representations:

- `responsible_actor_role`: who held the relevant duty or performed the action;
- `failed_action`: the positive action that was omitted, delayed, inadequate,
  or otherwise defective;
- `issue_object`: what the action operated on or affected;
- `counterparty_role`: the recipient or other party needed to preserve agency
  and communication direction;
- `canonical_issue`: a readable, evidence-preserving issue statement.

Code constructs `linkage_statement` deterministically from the first four
fields. This keeps public wording flexible while making the representation used
for linkage stable.

The embedding is the normalized concatenation of four independently embedded
views. Cosine similarity is therefore a weighted sum of their cosine
similarities:

- canonical issue: 0.10;
- full relation plus failure state, stage, and direction: 0.55;
- action and object: 0.25;
- actor, counterparty, and direction: 0.10.

Broad themes, sectors, settings, and populations are not embedded. They are
useful descriptors but are too broad to establish issue identity.

Nearest-neighbour retrieval is followed by explicit compatibility scoring.
Action family, object overlap, actor, counterparty, direction, stage, and
failure state can add small bonuses or penalties. This means a high semantic
similarity cannot silently erase contradictory agency or direction.

Only mutually retrieved cross-report pairs are eligible. A group must begin
with an edge above the strict seed threshold. A lower group threshold is used
only to attach a singleton to an existing seed through a qualifying direct
prototype link; lower-scoring pairs cannot seed groups or merge two established
groups. Two peripheral paraphrases need not be close to one another simply
because they express the same issue differently. Expansion is nevertheless
vetoed by hard relational conflicts, including reversed actor/counterparty
roles, opposite communication direction, incompatible action-object
relations, and objects that share only generic heads (for example, “CT scan
results” versus “smear test results”). A generic object cannot serve as a group
prototype. This blocks an ambiguous bridge from erasing agency while avoiding
the severe fragmentation caused by complete-link clustering. A group is
recurring at three distinct reports.

Before embedding, generic action/object fallbacks are excluded from automatic
linkage. A directional issue with no supported counterparty remains linkable
through its direction field, but is flagged for review; the pipeline does not
invent a party merely to satisfy the representation.

The pilot contains every occurrence in the manually reviewed random groups plus
every manually reviewed candidate pair. It compares strict, balanced, and
exploratory thresholds against those frozen decisions. The experiment should be
judged primarily on coherent groups and avoidance of false merges, with retained
true links as a secondary constraint.

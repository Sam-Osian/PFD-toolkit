# Archived automatic-parent research

Status: **superseded; not part of the active issue-tracker pipeline**.

This directory preserves the experiments that discovered broad issue families,
expanded overlapping memberships, audited parent recall, and generated direct
report-to-parent assignments. The work established that precise recurring
groups were useful, but automatically discovered parents were an unstable and
conceptually unsuitable public navigation system.

The replacement architecture has two independent, multi-label systems:

- precise issues extracted from concern text and grouped only for recurrence;
- report topics drawn from the manually curated catalogue in
  `../../locked_topics_v1.json`.

No file in this directory is imported by the production workflow or website.
Do not use its family IDs or parent assignments as current product data. The
historical scripts, findings and generated artifacts are retained only for
methodological traceability and reproducibility.

The archived regression checks are intentionally named `archived_test_*.py` so
ordinary repository test discovery ignores them. Run them explicitly from the
repository root with:

```bash
.venv/bin/pytest -q \
  scripts/issue_tracker_mvp/research/automatic_parent_families/tests/archived_test_*.py
```

Some archived command-line programs import active low-level issue-index helpers.
When reproducing an old command, expose both directories explicitly:

```bash
PYTHONPATH="scripts/issue_tracker_mvp:scripts/issue_tracker_mvp/research/automatic_parent_families" \
  .venv/bin/python \
  scripts/issue_tracker_mvp/research/automatic_parent_families/discover_issue_families.py \
  --help
```

The full experimental conclusions are recorded in
`AUTOMATIC_FAMILY_DISCOVERY_FINDINGS.md`.

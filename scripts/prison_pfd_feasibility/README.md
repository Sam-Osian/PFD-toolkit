# Prison PFD manual feasibility exercise

This directory contains the reproducible source-recovery stage for a
20-report, five-thread feasibility exercise. Generated reports, response PDFs,
extracted text and Andrew Harris's summaries are written under
`artifacts/prison_pfd_feasibility/`, which is ignored by git.

The sample is deliberately purposive rather than representative. It tests
whether the proposed concern → response action → later recurrence evidence
chain can be constructed across:

1. ACCT and suicide-risk processes;
2. observation and ligature safety;
3. mental-health access;
4. drugs, deterioration and emergency response; and
5. candour and organisational learning.

Build the reconciliation ledger without network access:

```bash
.venv/bin/python scripts/prison_pfd_feasibility/build_feasibility_sources.py \
  --docx "/path/to/AH PFD PRISON reports analysis.docx"
```

Recover official Judiciary attachments and extract their text:

```bash
.venv/bin/python scripts/prison_pfd_feasibility/build_feasibility_sources.py \
  --docx "/path/to/AH PFD PRISON reports analysis.docx" \
  --fetch
```

Build the human-review packet and one-row-per-case/thread annotation template:

```bash
.venv/bin/python scripts/prison_pfd_feasibility/build_manual_review_packet.py
```

After completing the manual annotations, validate every retained evidence quote
and generate the feasibility report:

```bash
.venv/bin/python scripts/prison_pfd_feasibility/analyse_manual_feasibility.py
```

The selected URLs have been manually confirmed against the local full-text
dataset. The source ledger retains the supplied summaries for comparison, but
they are not treated as ground truth.
